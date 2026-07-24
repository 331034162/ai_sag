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

异步说明：
  - 所有 a* 方法均使用 LlamaIndex 原生 AsyncMilvusClient（基于 async gRPC），
    为真正的协程实现，不经过 asyncio.to_thread 包装。
"""
from __future__ import annotations

from typing import Any

from pymilvus import DataType

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
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


def _missing_scalar_fields(client: Any, collection_name: str,
                          fields: list[str]) -> list[str] | None:
    """返回字段中尚未建索引的列表；若连接失败无法判断则返回 None。"""
    try:
        index_names = client.list_indexes(collection_name) or []
    except Exception:
        return None  # 无法判断

    indexed: set[str] = set()
    for idx_name in index_names:
        try:
            info = client.describe_index(collection_name, idx_name)
            # MilvusClient 返回 IndexInfo 对象（属性访问），旧客户端返回 dict
            field_name = (
                getattr(info, "field_name", None)
                or getattr(info, "fieldName", None)
            )
            if field_name:
                indexed.add(field_name)
        except Exception:
            pass

    return [f for f in fields if f not in indexed]


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
            # entities 为全局共享实体库，不归属任何 source/document，故不加 source_id 字段
            # 其余 collection 加 source_id 独立 VARCHAR 字段，可建 INVERTED 索引
            if name in _NO_SOURCE_FILTER:
                store = MilvusVectorStore(
                    uri=cfg.vector_store.milvus_uri or None,
                    token=cfg.vector_store.milvus_token or None,
                    dim=self._dim,
                    collection_name=collection_name,
                    overwrite=self._overwrite,
                    similarity_top_k=10,
                )
            else:
                store = MilvusVectorStore(
                    uri=cfg.vector_store.milvus_uri or None,
                    token=cfg.vector_store.milvus_token or None,
                    dim=self._dim,
                    collection_name=collection_name,
                    overwrite=self._overwrite,
                    # source_id 作为独立 VARCHAR 字段加入 schema，可建 INVERTED 索引
                    # node_to_metadata_dict 已将 metadata["source_id"] 写入 $meta，
                    # enable_dynamic_field=True 下自动匹配到 schema 同名字段
                    scalar_field_names=["source_id"],
                    scalar_field_types=[DataType.VARCHAR],
                    # Milvus 默认使用 cosine metric，与 L2 归一化后的余弦相似度一致
                    # similarity_top_k 在 query 时显式传，这里仅是构造默认值
                    similarity_top_k=10,
                )
            self._stores[name] = store
            # 为 source_id / document_id 建 INVERTED 索引（支持则建，不支持静默跳过）
            if name not in _NO_SOURCE_FILTER:
                self._ensure_scalar_index(store.client, collection_name)
            # Milvus collection 创建后默认处于 released 状态，需显式 load 才能查询
            try:
                store.client.load_collection(collection_name)
            except Exception:
                pass  # 新创建的 collection 为空时 load 可能抛异常，忽略即可

        # 打印 Milvus 运行模式与服务端版本
        _uri = (cfg.vector_store.milvus_uri or "").strip()
        is_lite = _uri.endswith(".db")
        mode = "milvus-lite（本地嵌入式）" if is_lite else "远程 Milvus 服务"
        try:
            server_ver = self._stores["chunks"].client.get_server_version()
        except Exception:
            server_ver = "未知"
        log.info("Milvus 向量库初始化完成 模式={} 服务端版本={} uri={} dim={} prefix={}",
                 mode, server_ver, cfg.vector_store.milvus_uri, self._dim, self._prefix)

    @staticmethod
    def _ensure_scalar_index(client: Any, collection_name: str) -> None:
        """为 doc_id / source_id 字段确保已有标量索引，无则建 INVERTED。

        - doc_id：llama_index 默认创建的独立 VARCHAR 字段
        - source_id：通过 scalar_field_names 加入 schema 的独立 VARCHAR 字段

        已有索引（含 Trie / INVERTED 等）直接复用，不强制重建。
        """
        # 1. 检查哪些字段已有索引（索引名不可靠，优先用字段名判断）
        fields_missing = _missing_scalar_fields(client, collection_name, ["doc_id", "source_id"])
        if not fields_missing:
            return

        # 2. 如果 list_indexes 断连失败 → 跳过建索引（不阻塞启动）
        if fields_missing is None:
            log.warning("无法查询 collection={} 现有索引（Milvus 连接不可用），跳过建索引",
                        collection_name)
            return

        try:
            from pymilvus.client.constants import LoadState
            if client.get_load_state(collection_name) == LoadState.Loaded:
                client.release_collection(collection_name)
        except Exception:
            pass

        index_params = client.prepare_index_params()
        for field_name in fields_missing:
            index_params.add_index(
                field_name=field_name,
                index_type="INVERTED",
                index_name=f"idx_{field_name}",
            )
        try:
            client.create_index(collection_name, index_params)
            log.info("创建标量索引 collection={} fields={}", collection_name, fields_missing)
        except Exception as e:
            # 1100 = index type not match（字段已有其他类型索引如 Trie，同样可用）
            log.warning("创建标量索引失败（可能字段已有索引）collection={} fields={} err={}",
                        collection_name, fields_missing, e)

    def _store(self, name: Collection) -> MilvusVectorStore:
        return self._stores[name]

    @staticmethod
    def _build_filters(source_ids: list[str] | None,
                       document_ids: list[str] | None) -> MetadataFilters | None:
        """构造 Milvus metadata 过滤器。

        - source_ids → MetadataFilter(key="source_id", operator=IN, value=source_ids)
        - document_ids → MetadataFilter(key="doc_id", operator=IN, value=document_ids)
          （doc_id 是 schema 独立 VARCHAR 字段，有 INVERTED 索引）
        - 同时存在 → AND 组合
        """
        filters: list[MetadataFilter] = []
        if source_ids:
            filters.append(MetadataFilter(
                key="source_id", value=source_ids, operator=FilterOperator.IN))
        if document_ids:
            filters.append(MetadataFilter(
                key="doc_id", value=document_ids, operator=FilterOperator.IN))
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
        # 关键：llama_index 会将 node.ref_doc_id 写入 schema 独立字段 doc_id（VARCHAR），
        # 同时 node_to_metadata_dict 也会将 ref_doc_id 写入 $meta["document_id"]。
        # 我们必须设置 SOURCE 关系让 ref_doc_id = 业务 document_id，
        # 否则 doc_id 字段会变成字符串 "None"，INVERTED 索引用不上。
        nodes: list[TextNode] = []
        for i in uniq_idx:
            meta = (metadatas[i] if metadatas else {}) or {}
            kwargs: dict[str, Any] = {
                "id_": ids[i], "text": texts[i],
                "embedding": embeddings[i], "metadata": meta,
            }
            doc_id = meta.get("document_id")
            if doc_id:
                kwargs["relationships"] = {
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id=str(doc_id))
                }
            nodes.append(TextNode(**kwargs))

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
        """按 source_id / document_id 过滤删除。

        source_id 和 doc_id 均为 schema 独立 VARCHAR 字段，各有 INVERTED 索引，
        delete 表达式可直接走索引精准命中，无需先查再删。
        """
        if not source_ids and not document_ids:
            return
        parts: list[str] = []
        if source_ids:
            ids_str = ", ".join(f"'{sid}'" for sid in source_ids)
            parts.append(f"source_id in [{ids_str}]")
        if document_ids:
            dids_str = ", ".join(f"'{did}'" for did in document_ids)
            parts.append(f"doc_id in [{dids_str}]")
        expr = " and ".join(parts)

        try:
            client = self._store(name).client
            collection_name = f"{self._prefix}{name}"
            resp = client.delete(collection_name=collection_name, filter=expr)
            deleted_count = resp.get("delete_count", 0) if isinstance(resp, dict) else 0
            log.info("向量删除 collection={} source_ids={} document_ids={} 删除数={}",
                     name, source_ids, document_ids, deleted_count)
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
            # Milvus 限制 (offset+limit) <= 16384，使用 query_iterator 游标分页
            iterator = client.query_iterator(
                collection_name=collection_name,
                filter="source_id != ''",
                output_fields=["source_id"],
                batch_size=16384,
            )
            ids: set[str] = set()
            while True:
                batch = iterator.next()
                if not batch:
                    break
                for row in batch:
                    sid = row.get("source_id")
                    if sid:
                        ids.add(str(sid))
            iterator.close()
            return list(ids)
        except Exception as e:
            log.warning("列出 Milvus source_id 失败 err={}", e)
            return []

    def list_all_entity_ids(self) -> list[str]:
        """从 entities collection 查询所有 entity_id。"""
        try:
            client = self._store("entities").client
            collection_name = f"{self._prefix}entities"
            # Milvus 限制 (offset+limit) <= 16384，使用 query_iterator 游标分页
            iterator = client.query_iterator(
                collection_name=collection_name,
                filter="id != ''",
                output_fields=["id"],
                batch_size=16384,
            )
            result: list[str] = []
            while True:
                batch = iterator.next()
                if not batch:
                    break
                for row in batch:
                    eid = row.get("id")
                    if eid is not None:
                        result.append(str(eid))
            iterator.close()
            return result
        except Exception as e:
            log.warning("列出 Milvus entity_id 失败 err={}", e)
            return []

    # ---------------- 异步接口（原生 AsyncMilvusClient，真协程）----------------

    async def aadd(self, name: Collection, ids: list[str], texts: list[str],
                   embeddings: list[list[float]], metadatas: list[dict] | None = None) -> None:
        """异步写入：通过 LlamaIndex async_add → AsyncMilvusClient 原生异步 gRPC。"""
        if not ids:
            return
        seen: set[str] = set()
        uniq_idx = [i for i, cid in enumerate(ids) if cid not in seen and not seen.add(cid)]
        nodes: list[TextNode] = []
        for i in uniq_idx:
            meta = (metadatas[i] if metadatas else {}) or {}
            kwargs: dict[str, Any] = {
                "id_": ids[i], "text": texts[i],
                "embedding": embeddings[i], "metadata": meta,
            }
            doc_id = meta.get("document_id")
            if doc_id:
                kwargs["relationships"] = {
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id=str(doc_id))
                }
            nodes.append(TextNode(**kwargs))
        if nodes:
            await self._store(name).async_add(nodes)
        log.info("向量异步写入(Milvus原生) collection={} 传入={} 内部去重后={} 实际写入={}",
                 name, len(ids), len(uniq_idx), len(nodes))

    async def aquery(self, name: Collection, query_embedding: list[float], top_k: int,
                     similarity_threshold: float = 0.0,
                     source_ids: list[str] | None = None,
                     document_ids: list[str] | None = None) -> list[tuple[str, float]]:
        """异步查询：通过 LlamaIndex aquery → AsyncMilvusClient 原生异步 gRPC。"""
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

        result = await self._store(name).aquery(vsq)
        hits: list[tuple[str, float]] = []
        if result.nodes and result.similarities is not None:
            safe_ids = result.ids or []
            for idx, (node, sim) in enumerate(zip(result.nodes, result.similarities)):
                if sim >= similarity_threshold:
                    real_id = safe_ids[idx] if idx < len(safe_ids) else node.node_id
                    hits.append((real_id, float(sim)))
        log.debug("向量异步查询(Milvus原生) collection={} top_k={} 阈值={} 结果数={}",
                  name, top_k, similarity_threshold, len(hits))
        return hits

    async def adelete_by_source(self, source_id: str) -> None:
        """异步按 source_id 删除：AsyncMilvusClient 原生删除。"""
        for name in ("chunks", "event_titles", "event_contents", "event_summaries"):
            await self._adelete_by_filter(name, source_ids=[source_id])

    async def adelete_by_document(self, source_id: str, document_id: str) -> None:
        """异步按 (source_id, document_id) 删除。"""
        for name in ("chunks", "event_titles", "event_contents", "event_summaries"):
            await self._adelete_by_filter(name, source_ids=[source_id], document_ids=[document_id])

    async def _adelete_by_filter(self, name: Collection, *,
                                 source_ids: list[str] | None = None,
                                 document_ids: list[str] | None = None) -> None:
        """异步按 source_id / document_id 过滤删除（走 INVERTED 索引精准命中）。"""
        if not source_ids and not document_ids:
            return
        parts: list[str] = []
        if source_ids:
            ids_str = ", ".join(f"'{sid}'" for sid in source_ids)
            parts.append(f"source_id in [{ids_str}]")
        if document_ids:
            dids_str = ", ".join(f"'{did}'" for did in document_ids)
            parts.append(f"doc_id in [{dids_str}]")
        expr = " and ".join(parts)

        try:
            store = self._store(name)
            collection_name = f"{self._prefix}{name}"
            resp = await store.async_client.delete(collection_name=collection_name, filter=expr)
            deleted_count = resp.get("delete_count", 0) if isinstance(resp, dict) else 0
            log.info("向量异步删除(Milvus原生) collection={} source_ids={} document_ids={} 删除数={}",
                     name, source_ids, document_ids, deleted_count)
        except Exception as e:
            log.error("向量异步删除失败 collection={} source_ids={} document_ids={} err={}",
                      name, source_ids, document_ids, e)

    async def adelete_entities_by_ids(self, entity_ids: list[str]) -> None:
        """异步按 entity_id 删除：AsyncMilvusClient 原生删除。"""
        if not entity_ids:
            return
        try:
            await self._store("entities").adelete_nodes(node_ids=entity_ids)
            log.info("实体向量异步删除 ids数量={}", len(entity_ids))
        except Exception as e:
            log.error("实体向量异步删除失败 ids={} err={}", entity_ids, e)

    async def adelete_event_ids(self, event_ids: list[str]) -> None:
        """异步按 event_id 删除 event_titles / event_contents / event_summaries。"""
        if not event_ids:
            return
        for name in ("event_titles", "event_contents", "event_summaries"):
            try:
                await self._store(name).adelete_nodes(node_ids=event_ids)
                log.info("事件向量异步删除 collection={} ids数量={}", name, len(event_ids))
            except Exception as e:
                log.warning("事件向量异步删除失败 collection={} ids={} err={}",
                            name, event_ids, e)

    async def aget_embeddings(self, name: Collection, ids: list[str]) -> dict[str, list[float]]:
        """异步按 id 批量取向量：AsyncMilvusClient 原生查询。"""
        if not ids:
            return {}
        result: dict[str, list[float]] = {}
        try:
            store = self._store(name)
            collection_name = f"{self._prefix}{name}"
            res = await store.async_client.query(
                collection_name=collection_name,
                filter=f"id in {list(ids)}",
                output_fields=["id", "embedding"],
            )
            for row in res:
                row_id = str(row.get("id") or row.get("pk"))
                emb = row.get("embedding")
                if row_id and emb is not None:
                    result[row_id] = list(emb)
        except Exception as e:
            log.warning("Milvus 异步批量取向量失败 collection={} ids数={} err={}",
                        name, len(ids), e)
        return result

    async def alist_source_ids(self) -> list[str]:
        """异步列出所有 source_id：AsyncMilvusClient 分页查询。"""
        try:
            store = self._store("chunks")
            collection_name = f"{self._prefix}chunks"
            async_client = store.async_client
            ids: set[str] = set()
            offset = 0
            limit = 16384
            while True:
                res = await async_client.query(
                    collection_name=collection_name,
                    filter="source_id != ''",
                    output_fields=["source_id"],
                    offset=offset,
                    limit=limit,
                )
                if not res:
                    break
                for row in res:
                    sid = row.get("source_id")
                    if sid:
                        ids.add(str(sid))
                if len(res) < limit:
                    break
                offset += limit
            return list(ids)
        except Exception as e:
            log.warning("列出 Milvus source_id 失败(异步) err={}", e)
            return []

    async def alist_all_entity_ids(self) -> list[str]:
        """异步列出所有 entity_id：AsyncMilvusClient 分页查询。"""
        try:
            store = self._store("entities")
            collection_name = f"{self._prefix}entities"
            async_client = store.async_client
            result: list[str] = []
            offset = 0
            limit = 16384
            while True:
                res = await async_client.query(
                    collection_name=collection_name,
                    filter="id != ''",
                    output_fields=["id"],
                    offset=offset,
                    limit=limit,
                )
                if not res:
                    break
                for row in res:
                    eid = row.get("id")
                    if eid is not None:
                        result.append(str(eid))
                if len(res) < limit:
                    break
                offset += limit
            return result
        except Exception as e:
            log.warning("列出 Milvus entity_id 失败(异步) err={}", e)
            return []