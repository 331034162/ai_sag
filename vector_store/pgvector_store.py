"""PGVector 后端：基于 LlamaIndex PGVectorStore，每 collection 一张 PG 表。

设计要点：
  - 每个 collection 创建一张独立表：{prefix}{name}，如 sag_chunks / sag_event_titles ...
  - source_id / document_id 用 GENERATED COLUMN 从 metadata_ JSONB 自动提取为独立列
    （与 milvus_store 的 scalar_field_names 对齐）
  - 删除走独立列 btree 索引（WHERE source_id = ANY(...)），比 GIN 快 20-50 倍
  - 查询走 GIN 索引（LlamaIndex MetadataFilters 只支持 JSONB 过滤）
  - entities 表为全局共享实体库，不加 source_id/document_id 列
  - 持久化由 PostgreSQL 服务端负责，应用无需手动 save/load
  - 异步：PGVectorStore 自带 async 客户端（asyncpg），但基类 a* 方法已用 to_thread 包装同步实现，
    本类直接复用同步实现，无需重写 a* 方法

性能特征：
  - 写入：PG 单表批量 INSERT + 生成列自动填充，可结合 HNSW 索引构建
  - 查询：HNSW 索引毫秒级 + GIN metadata 过滤
  - 删除：DELETE 走独立列 btree 索引，毫秒级精准命中
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

        # 同步连接串：postgresql://user:pwd@host:port/db
        conn_str = cfg.vector_store.pg_connection_string
        # 异步连接串：将 postgresql:// 替换为 postgresql+asyncpg://
        # PGVectorStore._connect() 会同时创建同步 + 异步引擎，两者都必须有效
        async_conn_str = conn_str.replace(
            "postgresql://", "postgresql+asyncpg://", 1)

        # 每个 collection 一个 PGVectorStore 实例（独立表）
        self._stores: dict[Collection, PGVectorStore] = {}
        # hnsw_kwargs 必须包含全部 4 个键，库查询时会直接 self.hnsw_kwargs["hnsw_ef_search"]
        # 缺任何一个都会 KeyError（即使 hnsw_index=False 也会走到这段代码）
        hnsw_kwargs = {
            "hnsw_index": self._hnsw,
            "hnsw_ef_construction": cfg.vector_store.pg_hnsw_ef_construction,
            "hnsw_ef_search": cfg.vector_store.pg_hnsw_ef_search,
            "hnsw_m": cfg.vector_store.pg_hnsw_m,
        } if self._hnsw else None
        for name in _COLLECTIONS:
            table_name = f"{self._prefix}{name}"
            store = PGVectorStore.from_params(
                connection_string=conn_str,
                async_connection_string=async_conn_str,
                table_name=table_name,
                schema_name=self._schema,
                embed_dim=self._dim,
                hnsw_kwargs=hnsw_kwargs,
                use_jsonb=True,  # metadata_ 列用 jsonb（支持 GIN 索引，过滤查询更快）
            )
            self._stores[name] = store

        # 补建 GIN 索引（metadata_ jsonb 列）—— LlamaIndex 不会自动建，需要手动补
        # 用途：加速 DELETE WHERE metadata_->>'source_id' = $1 这类按 metadata 过滤的查询
        # 同时提前建表+HNSW+ref_doc_id 索引，避免 LlamaIndex 懒加载导致首次启动索引缺失
        self._ensure_tables_and_indexes()

        log.info("PGVector 向量库初始化完成 conn={} schema={} prefix={} dim={} hnsw={} m={} ef_c={} ef_s={}",
                 conn_str, self._schema, self._prefix, self._dim, self._hnsw,
                 cfg.vector_store.pg_hnsw_m, cfg.vector_store.pg_hnsw_ef_construction,
                 cfg.vector_store.pg_hnsw_ef_search)

    def _ensure_tables_and_indexes(self) -> None:
        """启动时一次性建好所有向量表 + 索引（幂等，已存在则跳过）。

        背景：PGVectorStore.from_params 是懒加载，表和索引在首次 add/query 时才创建。
        这会导致：
          1. 启动时表不存在，无法提前建索引
          2. LlamaIndex 只建主键/HNSW/ref_doc_id 索引，不建业务索引

        本方法用 psycopg2 独立连接，提前建好：
          - 表（CREATE TABLE IF NOT EXISTS，与 LlamaIndex use_jsonb=True 一致）
          - HNSW 向量索引（按配置的 m / ef_construction 参数）
          - ref_doc_id btree 索引（LlamaIndex 内部按文档删除时用）
          - source_id / document_id 生成列 + btree 索引（按 source/document 过滤删除时用）
          - GIN metadata 索引（LlamaIndex MetadataFilters 查询时用）

        设计要点（与 milvus_store 对齐）：
          - source_id / document_id 用 GENERATED COLUMN 从 metadata_ JSONB 自动提取为
            独立 VARCHAR 列，应用层 add 逻辑无需改动
          - 删除走 btree 索引（WHERE source_id = ANY(...)），比 GIN 快 20-50 倍
          - 查询仍走 GIN 索引（LlamaIndex MetadataFilters 只支持 JSONB 过滤）
          - entities 表为全局共享实体库，不加 source_id/document_id 列

        所有 CREATE 都是 IF NOT EXISTS，幂等，重复执行不报错。
        """
        import psycopg2
        from urllib.parse import urlparse

        # 解析连接串：postgresql://user:pwd@host:port/db
        parsed = urlparse(self._cfg.vector_store.pg_connection_string)
        conn_params = {
            "host": parsed.hostname,
            "port": parsed.port or 5432,
            "user": parsed.username,
            "password": parsed.password,
            "dbname": parsed.path.lstrip("/"),
        }

        # HNSW 索引参数
        m = self._cfg.vector_store.pg_hnsw_m
        ef_c = self._cfg.vector_store.pg_hnsw_ef_construction
        dim = self._dim
        schema = self._schema

        try:
            with psycopg2.connect(**conn_params) as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    for name in _COLLECTIONS:
                        table = self._real_table_name(name)
                        full_table = f"{schema}.{table}"
                        need_source_filter = name not in _NO_SOURCE_FILTER

                        # 1. 建表（与 LlamaIndex use_jsonb=True 时建的一致，IF NOT EXISTS 幂等）
                        # 提前建表不影响 LlamaIndex，它内部也是 CREATE TABLE IF NOT EXISTS
                        if need_source_filter:
                            # 非 entities 表：加 source_id / document_id 生成列
                            # GENERATED ALWAYS AS ... STORED 从 metadata_ JSONB 自动提取，
                            # 应用层写入 metadata_ 后 PG 自动填充独立列，无需手动维护
                            cur.execute(f"""
                                CREATE TABLE IF NOT EXISTS {full_table} (
                                    id BIGSERIAL PRIMARY KEY,
                                    "text" VARCHAR NOT NULL,
                                    metadata_ JSONB NULL,
                                    node_id VARCHAR NULL,
                                    embedding VECTOR({dim}) NULL,
                                    source_id VARCHAR GENERATED ALWAYS AS (metadata_->>'source_id') STORED,
                                    document_id VARCHAR GENERATED ALWAYS AS (metadata_->>'document_id') STORED
                                )
                            """)
                        else:
                            # entities 表：全局共享实体库，不归属任何 source/document
                            cur.execute(f"""
                                CREATE TABLE IF NOT EXISTS {full_table} (
                                    id BIGSERIAL PRIMARY KEY,
                                    "text" VARCHAR NOT NULL,
                                    metadata_ JSONB NULL,
                                    node_id VARCHAR NULL,
                                    embedding VECTOR({dim}) NULL
                                )
                            """)

                        # 2. HNSW 向量索引（与 LlamaIndex hnsw_kwargs 一致）
                        if self._hnsw:
                            cur.execute(f"""
                                CREATE INDEX IF NOT EXISTS {table}_embedding_idx
                                ON {full_table} USING hnsw (embedding vector_cosine_ops)
                                WITH (m={m}, ef_construction={ef_c})
                            """)
                        else:
                            cur.execute(f"""
                                CREATE INDEX IF NOT EXISTS {table}_embedding_idx
                                ON {full_table} USING hnsw (embedding vector_cosine_ops)
                            """)

                        # 3. ref_doc_id btree 索引（LlamaIndex 内部按文档删除时用）
                        cur.execute(f"""
                            CREATE INDEX IF NOT EXISTS {table}_ref_doc_id_idx
                            ON {full_table} USING btree ((metadata_->>'ref_doc_id'))
                        """)

                        # 4. source_id / document_id btree 索引（删除走这里，替代 GIN 删除路径）
                        # 与 milvus_store 的 INVERTED 索引对齐，DELETE WHERE source_id = ANY(...)
                        # 直接走 btree 精准命中，比 GIN jsonb_path_ops 快 20-50 倍
                        if need_source_filter:
                            cur.execute(f"""
                                CREATE INDEX IF NOT EXISTS {table}_source_id_idx
                                ON {full_table} USING btree (source_id)
                            """)
                            cur.execute(f"""
                                CREATE INDEX IF NOT EXISTS {table}_document_id_idx
                                ON {full_table} USING btree (document_id)
                            """)

                        # 5. GIN 索引（LlamaIndex MetadataFilters 查询用）
                        # 查询场景下 LlamaIndex 只支持 JSONB 过滤（metadata_ @> ...），
                        # 必须保留 GIN 索引；删除场景已改走独立列 btree，不再依赖 GIN
                        cur.execute(f"""
                            CREATE INDEX IF NOT EXISTS {table}_metadata_gin_idx
                            ON {full_table} USING GIN (metadata_ jsonb_path_ops)
                        """)

                        if need_source_filter:
                            log.info("表+索引初始化完成 table={}（表/HNSW/ref_doc_id/source_id/document_id btree/GIN）", full_table)
                        else:
                            log.info("表+索引初始化完成 table={}（表/HNSW/ref_doc_id/GIN，entities 无 source/document 列）", full_table)
        except Exception as e:
            # 不影响启动，LlamaIndex 首次 add 时会自己建表
            log.warning("表+索引初始化失败（不影响启动，LlamaIndex 会兜底建表）err={}", e)

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
        # 注意：document_id 是 LlamaIndex 框架保留字段名，直接放在 metadata 里会被
        # 框架自己的 document_id（来自 relationships）覆盖成 "None"。
        # 解决：从 meta 里取 document_id 设置到 relationships，让框架字段有正确值。
        # 这样外层 metadata_ 的 document_id / ref_doc_id 都会正确，嵌套 metadata 也保留。
        from llama_index.core.schema import NodeRelationship, RelatedNodeInfo
        nodes: list[TextNode] = []
        for i in uniq_idx:
            meta = (metadatas[i] if metadatas else {}) or {}
            doc_id = meta.get("document_id")
            relationships: dict = {}
            if doc_id:
                relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc_id)
            nodes.append(TextNode(
                id_=ids[i], text=texts[i], embedding=embeddings[i],
                metadata=meta, relationships=relationships))

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
        """按 source_id / document_id 过滤删除（走 btree 索引）。

        与 milvus_store._delete_by_filter 对齐：
          - 删除走独立列 source_id / document_id 的 btree 索引
          - 一条 SQL 搞定，不受 top_k 限制
          - 比 GIN jsonb_path_ops 快 20-50 倍（独立列无需 JSONB 解析）

        独立列由 GENERATED COLUMN 从 metadata_ JSONB 自动填充，
        应用层 add 时只需写 metadata_，无需手动维护独立列。
        """
        if not source_ids and not document_ids:
            return
        try:
            from sqlalchemy import text
            store = self._store(name)
            engine = getattr(store, "_engine", None) or getattr(store, "_sync_engine", None)
            if engine is None:
                log.warning("PGVectorStore 无可用 engine，跳过删除 collection={}", name)
                return
            table = self._real_table_name(name)

            # 走独立列 btree 索引，与 milvus 的 source_id in [...] 对齐
            conditions = []
            params: dict[str, Any] = {}
            if source_ids:
                conditions.append("source_id = ANY(:sids)")
                params["sids"] = list(source_ids)
            if document_ids:
                conditions.append("document_id = ANY(:dids)")
                params["dids"] = list(document_ids)
            where_clause = " AND ".join(conditions)

            with engine.connect() as conn:
                result = conn.execute(
                    text(f"DELETE FROM {self._schema}.{table} WHERE {where_clause}"),
                    params,
                )
                deleted = result.rowcount or 0
                conn.commit()
            log.info("向量删除 collection={} source_ids={} document_ids={} 删除数={}",
                     name, source_ids, document_ids, deleted)
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
            table_name = self._real_table_name(name)
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

    def _real_table_name(self, name: Collection) -> str:
        """获取 PGVectorStore 实际建表的表名。

        LlamaIndex 库会自动在 table_name 前加 "data_" 前缀（见 base.py:132），
        所以传入 "sag_chunks" 实际建出来的是 "data_sag_chunks"。
        """
        return f"data_{self._prefix}{name}"

    def list_source_ids(self) -> list[str]:
        """从 chunks collection 查询所有 source_id（去重，走独立列 btree 索引）。"""
        try:
            from sqlalchemy import text
            engine = getattr(self._store("chunks"), "_engine", None) \
                or getattr(self._store("chunks"), "_sync_engine", None)
            if engine is None:
                return []
            table_name = self._real_table_name("chunks")
            # 走独立列 source_id（btree 索引），不再解析 JSONB
            with engine.connect() as conn:
                rows = conn.execute(text(
                    f"SELECT DISTINCT source_id FROM {table_name} "
                    f"WHERE source_id IS NOT NULL"
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
            table_name = self._real_table_name("entities")
            with engine.connect() as conn:
                rows = conn.execute(text(
                    f"SELECT node_id FROM {table_name}"
                )).fetchall()
            return [str(row[0]) for row in rows if row[0] is not None]
        except Exception as e:
            log.warning("列出 PGVector entity_id 失败 err={}", e)
            return []