"""PGVector 后端：基于 LlamaIndex PGVectorStore，每 collection 一张 PG 表。

设计要点：
  - 每个 collection 创建一张独立表：{prefix}{name}，如 sag_chunks / sag_event_titles ...
  - PGVectorStore 原生支持 metadata 过滤（MetadataFilters），source_id / document_id
    作为 metadata 字段参与过滤
  - 持久化由 PostgreSQL 服务端负责，应用无需手动 save/load
  - 删除：调用 PGVectorStore.delete_nodes 按 node_id 删除（PG DELETE）
  - 异步：PGVectorStore 自带 async 客户端（asyncpg），但基类 a* 方法已用 to_thread 包装同步实现，
    本类直接复用同步实现，无需重写 a* 方法

性能特征：
  - 写入：PG 单表批量 INSERT，可结合 HNSW 索引构建（写入略慢于 Milvus）
  - 查询：HNSW 索引毫秒级，可走 SQL JOIN 联查业务数据
  - 删除：DELETE 物理删除，无内存开销
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
from llama_index.vector_stores.postgres import PGVectorStore

from ..base import Config
from ..base.logger import get_logger
from .base import BaseVectorStore, Collection

log = get_logger()

# 5 个逻辑集合名（与 chroma_store / faiss_store / milvus_store 保持一致）
_COLLECTIONS: tuple[str, ...] = (
    "chunks", "event_titles", "event_contents", "event_summaries", "entities",
)

# 不按 source_id / document_id 过滤的 collection（实体跨 source 共享）
_NO_SOURCE_FILTER: set[str] = {"entities"}


class PGVectorStoreBackend(BaseVectorStore):
    """PGVector 向量库后端。"""

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._dim = cfg.vector_store.dim
        self._prefix = cfg.vector_store.pg_table_prefix
        self._schema = cfg.vector_store.pg_schema_name
        self._hnsw = cfg.vector_store.pg_hnsw_index
        self._async = cfg.vector_store.pg_async

        # 每个 collection 一个 PGVectorStore 实例（独立表）
        self._stores: dict[Collection, PGVectorStore] = {}
        for name in _COLLECTIONS:
            table_name = f"{self._prefix}{name}"
            store = PGVectorStore.from_params(
                connection_string=cfg.vector_store.pg_connection_string,
                table_name=table_name,
                schema_name=self._schema,
                embed_dim=self._dim,
                hnsw_kwargs={"hnsw_index": self._hnsw},
                async_=self._async,
            )
            self._stores[name] = store
        log.info("PGVector 向量库初始化完成 conn={} schema={} prefix={} dim={} hnsw={}",
                 cfg.vector_store.pg_connection_string, self._schema, self._prefix,
                 self._dim, self._hnsw)

    def _store(self, name: Collection) -> PGVectorStore:
        return self._stores[name]

    @staticmethod
    def _build_filters(source_ids: list[str] | None,
                       document_ids: list[str] | None) -> MetadataFilters | None:
        """构造 PG metadata 过滤器。

        PGVectorStore metadata filter 通过 JSONB 字段查询，支持 IN 操作符。
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

        # 3. PGVectorStore.add 内部会按 node_id upsert（同 id 覆盖）
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
            for node, sim in zip(result.nodes, result.similarities):
                if sim >= similarity_threshold:
                    hits.append((node.node_id, float(sim)))
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
        """按 metadata 过滤删除（先查 id 再按 id 删）。

        PGVectorStore 没有直接的 delete-by-metadata API，分两步：
        1. 用零向量查询取出所有匹配的 node_id（filter 生效）
        2. 调 delete_nodes(node_ids=...) 按 PK 删除
        """
        try:
            filters = self._build_filters(source_ids, document_ids)
            if filters is None:
                return
            store = self._store(name)
            # 用零向量查询仅用于激活 metadata filter（PGVectorStore.query 会先做向量搜索再过滤，
            # 零向量排序无意义但能返回 filter 匹配的所有 row，受 similarity_top_k 限制）
            # 取一个较大的 top_k 保证覆盖
            vsq = VectorStoreQuery(
                query_embedding=[0.0] * self._dim,
                similarity_top_k=100000,
                mode="default",
                filters=filters,
            )
            try:
                result = store.query(vsq)
                node_ids = [n.node_id for n in (result.nodes or [])]
            except Exception:
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

        PGVectorStore 没有直接的 get 接口，通过底层 asyncpg / psycopg2 客户端按 node_id 查询。
        """
        if not ids:
            return {}
        result: dict[str, list[float]] = {}
        try:
            # PGVectorStore 内部 _sync_engine / _async_engine 提供 SQLAlchemy 引擎
            # 这里用 SQLAlchemy 走统一接口，避免驱动差异
            from sqlalchemy import text
            engine = getattr(self._store(name), "_engine", None)
            if engine is None:
                # 旧版兼容：尝试 _sync_engine
                engine = getattr(self._store(name), "_sync_engine", None)
            if engine is None:
                log.warning("PGVectorStore 无可用 engine，无法批量取向量 collection={}", name)
                return {}
            table_name = f"{self._prefix}{name}"
            # node_id 在 PGVectorStore 中存为 TEXT 列；embedding 列名为 embedding
            # 用 ANY(:ids) 防止 IN 子句参数数量超限
            with engine.connect() as conn:
                rows = conn.execute(
                    text(f"SELECT node_id, embedding FROM {table_name} WHERE node_id = ANY(:ids)"),
                    {"ids": list(ids)},
                ).fetchall()
            for row in rows:
                row_id = str(row[0])
                emb = row[1]
                if row_id and emb is not None:
                    # PGVectorStore embedding 列可能是 str/列表/numpy 数组，统一转 list
                    if isinstance(emb, str):
                        import json as _json
                        emb = _json.loads(emb)
                    result[row_id] = list(emb)
        except Exception as e:
            log.warning("PGVector 批量取向量失败 collection={} ids数={} err={}",
                        name, len(ids), e)
        return result

    def list_source_ids(self) -> list[str]:
        """从 chunks collection 查询所有 source_id（去重）。"""
        try:
            from sqlalchemy import text
            engine = getattr(self._store("chunks"), "_engine", None) \
                or getattr(self._store("chunks"), "_sync_engine", None)
            if engine is None:
                return []
            table_name = f"{self._prefix}chunks"
            # metadata_ 是 JSONB 列（PGVectorStore 默认列名）
            with engine.connect() as conn:
                rows = conn.execute(text(
                    f"SELECT DISTINCT metadata_->>'source_id' AS sid "
                    f"FROM {table_name} "
                    f"WHERE metadata_->>'source_id' IS NOT NULL"
                )).fetchall()
            return [str(row[0]) for row in rows if row[0]]
        except Exception as e:
            log.warning("列出 PGVector source_id 失败 err={}", e)
            return []

    def list_all_entity_ids(self) -> list[str]:
        """从 entities collection 查询所有 entity_id。"""
        try:
            from sqlalchemy import text
            engine = getattr(self._store("entities"), "_engine", None) \
                or getattr(self._store("entities"), "_sync_engine", None)
            if engine is None:
                return []
            table_name = f"{self._prefix}entities"
            with engine.connect() as conn:
                rows = conn.execute(text(
                    f"SELECT node_id FROM {table_name}"
                )).fetchall()
            return [str(row[0]) for row in rows if row[0] is not None]
        except Exception as e:
            log.warning("列出 PGVector entity_id 失败 err={}", e)
            return []