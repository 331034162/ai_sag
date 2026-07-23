"""Milvus 后端：基于 LlamaIndex MilvusVectorStore，支持元数据过滤与分布式部署。

设计要点：
  - 每个 collection 一个独立 Milvus 集合（命名为 {prefix}{name}，如 sag_chunks）
  - Milvus 原生支持 metadata 字段过滤，source_id / document_id 作为标量字段索引
  - 持久化由 Milvus 服务端负责，应用无需手动 save/load
  - 删除：调用 MilvusVectorStore.delete_nodes 按 id 删除（Milvus 物理删除）

性能特征：
  - 写入：Milvus 服务端批量写入，性能优于本地文件方案
  - 查询：原生支持 metadata filter + ANN 索引，毫秒级
  - 删除：原生物理删除，无内存标记开销
"""
from __future__ import annotations

from typing import Any

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
    VectorStoreQuery,
)
from llama_index.vector_stores.milvus import MilvusVectorStore

from ..base import Config
from ..base.logger import get_logger
from .base import BaseVectorStore, Collection

log = get_logger()

# 5 个逻辑集合名（与 chroma_store / faiss_store 保持一致）
_COLLECTIONS: tuple[str, ...] = (
    "chunks", "event_titles", "event_contents", "event_summaries", "entities",
)

# 不按 source_id / document_id 过滤的 collection（实体跨 source 共享）
_NO_SOURCE_FILTER: set[str] = {"entities"}


class MilvusVectorStoreBackend(BaseVectorStore):
    """Milvus 向量库后端。"""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._dim = cfg.vector_store.dim
        self._prefix = cfg.vector_store.milvus_collection_prefix
        self._overwrite = cfg.vector_store.milvus_overwrite

        # 每个 collection 一个 MilvusVectorStore 实例
        # uri 为空时 Milvus 客户端会用默认值，故 None/空字符串都传 None
        self._stores: dict[Collection, MilvusVectorStore] = {}
        for name in _COLLECTIONS:
            collection_name = f"{self._prefix}{name}"
            store = MilvusVectorStore(
                uri=cfg.vector_store.milvus_uri or None,
                token=cfg.vector_store.milvus_token or None,
                dim=self._dim,
                collection_name=collection_name,
                overwrite=self._overwrite,
                # Milvus 默认使用 cosine metric，与 L2 归一化后的余弦相似度一致
                # similarity_top_k 在 query 时显式传，这里仅是构造默认值
                similarity_top_k=10,
            )
            self._stores[name] = store
            # Milvus collection 创建后默认处于 released 状态，需显式 load 才能查询
            try:
                store.client.load_collection(collection_name)
            except Exception:
                pass  # 新创建的 collection 为空时 load 可能抛异常，忽略即可
        log.info("Milvus 向量库初始化完成 uri={} dim={} prefix={}",
                 cfg.vector_store.milvus_uri, self._dim, self._prefix)

    def _store(self, name: Collection) -> MilvusVectorStore:
        return self._stores[name]

    @staticmethod
    def _build_filters(source_ids: list[str] | None,
                       document_ids: list[str] | None) -> MetadataFilters | None:
        """构造 Milvus metadata 过滤器。

        - source_ids → MetadataFilter(key="source_id", operator=IN, value=source_ids)
        - document_ids → MetadataFilter(key="document_id", operator=IN, value=document_ids)
        - 同时存在 → AND 组合
        """
        filters: list[MetadataFilter] = []
        if source_ids:
            filters.append(MetadataFilter(
                key="source_id", value=source_ids, operator=FilterOperator.IN))
        if document_ids:
            filters.append(MetadataFilter(
                key="document_id", value=document_ids, operator=FilterOperator.IN))
        if not filters:
            return None
        return MetadataFilters(filters=filters, condition="and")

    # ---------------- 同步接口实现 ----------------

    def add(self, name: Collection, ids: list[str], texts: list[str],
            embeddings: list[list[float]], metadatas: list[dict] | None = None) -> None:
        if not ids:
            return
        # 1. 本次输入去重
        seen = set()
        uniq_idx = []
        for i, cid in enumerate(ids):
            if cid not in seen:
                seen.add(cid)
                uniq_idx.append(i)

        # 2. 构造 TextNode 列表
        nodes: list[TextNode] = []
        for i in uniq_idx:
            meta = (metadatas[i] if metadatas else {}) or {}
            nodes.append(TextNode(
                id_=ids[i], text=texts[i], embedding=embeddings[i], metadata=meta))

        # 3. MilvusVectorStore 内部会按 node_id upsert（同 id 覆盖）
        #    所以无需预先查询去重，直接 add 即可
        if nodes:
            self._store(name).add(nodes)
        log.info("向量写入 collection={} 传入={} 内部去重后={} 实际写入={}",
                 name, len(ids), len(uniq_idx), len(nodes))

    def query(self, name: Collection, query_embedding: list[float], top_k: int,
              similarity_threshold: float = 0.0,
              source_ids: list[str] | None = None,
              document_ids: list[str] | None = None) -> list[tuple[str, float]]:
        # entities 跨 source 共享，不按 source_id / document_id 过滤
        if name in _NO_SOURCE_FILTER:
            filters = None
        else:
            filters = self._build_filters(source_ids, document_ids)

        vsq_kwargs: dict[str, Any] = {
            "query_embedding": query_embedding,
            "similarity_top_k": top_k,
            "mode": "default",
        }
        if filters is not None:
            vsq_kwargs["filters"] = filters
        vsq = VectorStoreQuery(**vsq_kwargs)

        result = self._store(name).query(vsq)
        hits: list[tuple[str, float]] = []
        if result.nodes and result.similarities is not None:
            # 优先用 result.ids（Milvus 主键，即写入时的实体/chunk ID），
            # 因为 node.node_id 在 _node_content 未正确反序列化时是自动生成的随机 UUID
            safe_ids = result.ids or []
            for idx, (node, sim) in enumerate(zip(result.nodes, result.similarities)):
                if sim >= similarity_threshold:
                    real_id = safe_ids[idx] if idx < len(safe_ids) else node.node_id
                    hits.append((real_id, float(sim)))
        log.debug("向量查询 collection={} top_k={} 阈值={} 结果数={}",
                  name, top_k, similarity_threshold, len(hits))
        return hits

    def delete_by_source(self, source_id: str) -> None:
        """按 source_id 删除 4 个 collection 的向量（entities 不删）。"""
        for name in ("chunks", "event_titles", "event_contents", "event_summaries"):
            self._delete_by_filter(name, source_ids=[source_id])

    def delete_by_document(self, source_id: str, document_id: str) -> None:
        """按 (source_id, document_id) 删除 4 个 collection 的向量。"""
        for name in ("chunks", "event_titles", "event_contents", "event_summaries"):
            self._delete_by_filter(name, source_ids=[source_id], document_ids=[document_id])

    def _delete_by_filter(self, name: Collection, *,
                          source_ids: list[str] | None = None,
                          document_ids: list[str] | None = None) -> None:
        """按 metadata 过滤删除（先查 id 再按 id 删）。"""
        try:
            filters = self._build_filters(source_ids, document_ids)
            if filters is None:
                return
            store = self._store(name)
            # 先用查询取出所有匹配的 node_id
            # 取一个较大的 top_k 保证覆盖（受 Milvus 返回上限约束）
            vsq = VectorStoreQuery(
                query_embedding=[0.0] * self._dim,
                similarity_top_k=10000,
                mode="default",
                filters=filters,
            )
            try:
                result = store.query(vsq)
                node_ids = [n.node_id for n in (result.nodes or [])]
            except Exception:
                # 兜底：query 失败时改用 storage_context 的 delete 方法
                node_ids = []
            if node_ids:
                store.delete_nodes(node_ids=node_ids)
            log.info("向量删除 collection={} source_ids={} document_ids={} 删除数={}",
                     name, source_ids, document_ids, len(node_ids))
        except Exception as e:
            log.error("向量删除失败 collection={} source_ids={} document_ids={} err={}",
                      name, source_ids, document_ids, e)

    def delete_entities_by_ids(self, entity_ids: list[str]) -> None:
        if not entity_ids:
            return
        try:
            self._store("entities").delete_nodes(node_ids=entity_ids)
            log.info("实体向量删除 ids数量={}", len(entity_ids))
        except Exception as e:
            log.error("实体向量删除失败 ids={} err={}", entity_ids, e)

    def delete_event_ids(self, event_ids: list[str]) -> None:
        """按 event_id 删除 event_titles / event_contents / event_summaries。"""
        if not event_ids:
            return
        for name in ("event_titles", "event_contents", "event_summaries"):
            try:
                self._store(name).delete_nodes(node_ids=event_ids)
                log.info("事件向量删除 collection={} ids数量={}", name, len(event_ids))
            except Exception as e:
                log.warning("事件向量删除失败 collection={} ids={} err={}", name, event_ids, e)

    def get_embeddings(self, name: Collection, ids: list[str]) -> dict[str, list[float]]:
        """按 id 批量取已存向量。

        MilvusVectorStore 没有直接的 get 接口，通过底层 pymilvus 客户端按 pk 查询。
        """
        if not ids:
            return {}
        result: dict[str, list[float]] = {}
        try:
            client = self._store(name).client
            collection_name = f"{self._prefix}{name}"
            # pymilvus Collection.fetch / query 接口
            # 优先用 query 接口（新版统一 API）
            try:
                # 较新版本 pymilvus
                res = client.query(
                    collection_name=collection_name,
                    filter=f"id in {list(ids)}",
                    output_fields=["id", "embedding"],
                )
            except Exception:
                # 旧版接口：直接 fetch
                col = client.get_collection(collection_name)
                res = col.query(
                    expr=f"id in {list(ids)}",
                    output_fields=["id", "embedding"],
                )
            for row in res:
                row_id = str(row.get("id") or row.get("pk"))
                emb = row.get("embedding")
                if row_id and emb is not None:
                    result[row_id] = list(emb)
        except Exception as e:
            log.warning("Milvus 批量取向量失败 collection={} ids数={} err={}",
                        name, len(ids), e)
        return result

    def list_source_ids(self) -> list[str]:
        """从 chunks collection 查询所有 source_id（去重）。"""
        try:
            client = self._store("chunks").client
            collection_name = f"{self._prefix}chunks"
            res = client.query(
                collection_name=collection_name,
                filter="source_id != ''",
                output_fields=["source_id"],
                limit=100000,
            )
            ids: set[str] = set()
            for row in res:
                sid = row.get("source_id")
                if sid:
                    ids.add(str(sid))
            return list(ids)
        except Exception as e:
            log.warning("列出 Milvus source_id 失败 err={}", e)
            return []

    def list_all_entity_ids(self) -> list[str]:
        """从 entities collection 查询所有 entity_id。"""
        try:
            client = self._store("entities").client
            collection_name = f"{self._prefix}entities"
            res = client.query(
                collection_name=collection_name,
                filter="id != ''",
                output_fields=["id"],
                limit=1000000,
            )
            return [str(row.get("id")) for row in res if row.get("id") is not None]
        except Exception as e:
            log.warning("列出 Milvus entity_id 失败 err={}", e)
            return []