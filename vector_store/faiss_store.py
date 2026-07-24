"""FAISS 后端：基于 faiss.IndexIDMap2 + 独立映射表反查。

设计要点（与其他后端的核心差异）：
  - FAISS IndexIDMap2 要求 id 为 int64，无法直接存 UUID
  - 通过 blake2b 把 UUID 哈希为 int64（storage.uuid_to_int64），存入 FAISS
  - 独立映射表（faiss_chunks_map / faiss_events_map / faiss_entities_map）维护
    faiss_hash ↔ uuid/source_id/document_id 映射，业务表（aisag_*）保持后端无关
  - 查询时：FAISS 返回 hash → db_store.fetch_*_by_hashes JOIN 映射表+业务表反查
  - 删除：直接 faiss.remove_ids（硬删除，不再用 JSON sidecar 软删）
  - 持久化：每个 collection 一个 .index 文件，add/delete 后写盘
  - 映射表设计便于未来把 FAISS 拆为独立微服务（迁表即可，不动业务表 schema）

性能特征：
  - 写入：O(N) 追加，触发持久化时整体重写（小数据可接受）
  - 查询：O(N) 暴力搜索（IndexFlatL2），N<100w 时毫秒级
  - 删除：faiss.remove_ids 内部需重建索引，O(N)
  - source_id / document_id 过滤：通过 db_store 反查 hash 集合做前置过滤（避免 faiss 大范围查询）

依赖：db_store（MysqlStore / PgStore，必须传入，用于 hash → UUID 反查）
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..base import Config
from ..base.logger import get_logger
from ..storage import uuid_to_int64
from .base import BaseVectorStore, Collection

if TYPE_CHECKING:
    from ..storage import MysqlStore, PgStore

log = get_logger()

# 5 个逻辑集合名（与 chroma_store 保持一致）
_COLLECTIONS: tuple[str, ...] = (
    "chunks", "event_titles", "event_contents", "event_summaries", "entities",
)

# collection → DB 业务表名映射（用于 hash 反查）
# 注意：event_titles / event_contents / event_summaries 三个 collection 都查 aisag_events 表
_COLL_TO_TABLE: dict[str, str] = {
    "chunks": "aisag_chunks",
    "event_titles": "aisag_events",
    "event_contents": "aisag_events",
    "event_summaries": "aisag_events",
    "entities": "aisag_entities",
}

# collection → FAISS 映射表名（hash ↔ uuid/source_id/document_id 映射，独立于业务表）
_COLL_TO_MAP_TABLE: dict[str, str] = {
    "chunks": "faiss_chunks_map",
    "event_titles": "faiss_events_map",
    "event_contents": "faiss_events_map",
    "event_summaries": "faiss_events_map",
    "entities": "faiss_entities_map",
}


class FaissVectorStoreBackend(BaseVectorStore):
    """FAISS 向量库后端（IndexIDMap2 + 独立映射表）。

    依赖 db_store（MysqlStore / PgStore）来完成 hash → UUID 反查（query 时）和 source_id 过滤。
    """

    def __init__(self, cfg: Config, db_store: "MysqlStore | PgStore | None" = None) -> None:
        import faiss

        self._cfg = cfg
        self._dim = cfg.vector_store.dim
        self._path = Path(cfg.vector_store.faiss_path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = db_store

        # 每个 collection 一份 IndexIDMap2（包装 IndexFlatL2 以支持 add_with_ids）
        self._stores: dict[Collection, Any] = {}
        for name in _COLLECTIONS:
            self._stores[name] = self._load_or_create_index(name)
        log.info("FAISS 向量库初始化完成 path={} dim={} db_linked={}",
                 self._path, self._dim, db_store is not None)

    # ---------------- 私有：持久化辅助 ----------------

    def _index_path(self, name: str) -> Path:
        return self._path / f"{name}.index"

    def _load_or_create_index(self, name: str):
        """加载已有 faiss 索引，不存在则新建 IndexIDMap2(IndexFlatL2)。"""
        import faiss
        path = self._index_path(name)
        if path.exists():
            try:
                idx = faiss.read_index(str(path))
                # 旧文件可能是 IndexFlatL2（无 IDMap 包装），自动迁移
                if not isinstance(idx, faiss.IndexIDMap2):
                    idx = faiss.IndexIDMap2(idx)
                return idx
            except Exception as e:
                log.warning("FAISS 索引读取失败，将重建 collection={} err={}", name, e)
        base = faiss.IndexFlatL2(self._dim)
        return faiss.IndexIDMap2(base)

    def _persist(self, name: Collection) -> None:
        """持久化 faiss 索引到磁盘。"""
        import faiss
        try:
            faiss.write_index(self._stores[name], str(self._index_path(name)))
        except Exception as e:
            log.error("FAISS 索引写入失败 collection={} err={}", name, e)

    def _require_db(self) -> "MysqlStore | PgStore":
        if self._db is None:
            raise RuntimeError(
                "FAISS 后端查询/反查需要 db_store，请在 create_vector_store 时传入")
        return self._db

    # ---------------- 同步接口：写入 / 删除（不依赖 DB 反查）----------------

    def add(self, name: Collection, ids: list[str], texts: list[str],
            embeddings: list[list[float]], metadatas: list[dict] | None = None) -> None:
        if not ids:
            return
        from ..storage import uuid_to_int64
        import faiss
        import numpy as np

        with self._lock:
            store = self._stores[name]
            # 去重（同 id 只写一次）
            seen = set()
            uniq_idx = [i for i, cid in enumerate(ids) if cid not in seen and not seen.add(cid)]
            # FAISS 不支持同 id 覆盖，需先删后加
            hashes_to_remove = [uuid_to_int64(ids[i]) for i in uniq_idx]
            selector = faiss.IDSelectorBatch(hashes_to_remove)
            try:
                store.remove_ids(selector)
            except Exception:
                pass  # 索引为空时 remove_ids 可能报错，忽略
            # 重新加入
            hashes = np.array(hashes_to_remove, dtype=np.int64)
            embs = np.array([embeddings[i] for i in uniq_idx], dtype=np.float32)
            store.add_with_ids(embs, hashes)
            self._persist(name)
            log.info("向量写入 collection={} 传入={} 实际写入={}",
                     name, len(ids), len(uniq_idx))

    def query(self, name: Collection, query_embedding: list[float], top_k: int,
              similarity_threshold: float = 0.0,
              source_ids: list[str] | None = None,
              document_ids: list[str] | None = None) -> list[tuple[str, float]]:
        """同步查询：仅做 FAISS 检索，返回 hash 字符串（非 UUID）。

        警告：FAISS 后端不推荐用同步 query，因为无法回查 DB 得到 UUID。
        请使用异步 aquery（会自动回查 DB）。
        """
        import numpy as np
        with self._lock:
            store = self._stores[name]
            q = np.array([query_embedding], dtype=np.float32)
            fetch_k = max(top_k * 3, 50)  # 多取一些，应对过滤后不足
            dists, ids = store.search(q, fetch_k)
        hits: list[tuple[str, float]] = []
        if ids is None:
            return hits
        for hid, dist in zip(ids[0], dists[0]):
            if hid < 0:
                continue  # -1 表示无结果
            # IndexFlatL2 返回的是 L2 距离（越小越相似），转为 similarity（越大越相似）
            sim = 1.0 / (1.0 + float(dist))
            if sim < similarity_threshold:
                continue
            hits.append((str(hid), sim))
            if len(hits) >= top_k:
                break
        return hits

    def delete_by_source(self, source_id: str) -> None:
        """按 source_id 删除向量。FAISS 无法直接过滤 metadata，需通过 DB 反查 hash 集合。

        此方法为同步接口，需在事件循环外调用（内部用 asyncio.run）。
        """
        self._async_run_sync(self._adelete_by_source_impl, source_id)

    def delete_by_document(self, source_id: str, document_id: str) -> None:
        """按 (source_id, document_id) 删除。同样需通过 DB 反查 hash 集合。"""
        self._async_run_sync(self._adelete_by_document_impl, source_id, document_id)

    def delete_entities_by_ids(self, entity_ids: list[str]) -> None:
        if not entity_ids:
            return
        import faiss
        with self._lock:
            store = self._stores["entities"]
            hashes = [uuid_to_int64(eid) for eid in entity_ids]
            selector = faiss.IDSelectorBatch(hashes)  # type: ignore[name-defined]
            try:
                removed = store.remove_ids(selector)
                if removed > 0:
                    self._persist("entities")
                log.info("实体向量删除 ids数量={} 实际删除={}", len(entity_ids), removed)
            except Exception as e:
                log.error("实体向量删除失败 ids={} err={}", entity_ids, e)

    def get_embeddings(self, name: Collection, ids: list[str]) -> dict[str, list[float]]:
        """按 id 批量取已存向量。通过 faiss reconstruct + hash 反查 UUID（可选）。"""
        if not ids:
            return {}
        result: dict[str, list[float]] = {}
        with self._lock:
            store = self._stores[name]
            for cid in ids:
                h = uuid_to_int64(cid)
                try:
                    vec = store.reconstruct(h)
                    result[cid] = list(vec)
                except Exception:
                    continue  # id 不存在
        return result

    async def aquery(self, name: Collection, query_embedding: list[float], top_k: int,
                     similarity_threshold: float = 0.0,
                     source_ids: list[str] | None = None,
                     document_ids: list[str] | None = None) -> list[tuple[str, float]]:
        """异步查询：FAISS 检索 → DB 反查 UUID → 返回 [(uuid, sim), ...]。

        流程：
          1. 如果 source_ids/document_ids 过滤，先从 DB 反查 hash 集合做白名单
          2. FAISS search 拿到 (hash, dist) 列表
          3. DB fetch_*_by_hashes 反查 UUID
          4. 返回 [(uuid, sim), ...]
        """
        import numpy as np
        db = self._require_db()

        # 1. source_id / document_id 过滤 → 反查 hash 集合
        # 简化处理：fetch top_k*3，然后逐个查 DB 过滤
        with self._lock:
            store = self._stores[name]
            q = np.array([query_embedding], dtype=np.float32)
            # 多取一些应对过滤后不足
            fetch_k = max(top_k * 3, 50)
            dists, ids = store.search(q, fetch_k)

        # 收集 (hash, sim) 候选
        candidates: list[tuple[int, float]] = []
        if ids is not None:
            for hid, dist in zip(ids[0], dists[0]):
                if hid < 0:
                    continue
                sim = 1.0 / (1.0 + float(dist))
                if sim < similarity_threshold:
                    continue
                candidates.append((int(hid), sim))
        if not candidates:
            return []

        # 2. 反查 UUID（按 collection 走对应映射表+业务表 JOIN）
        if name == "chunks":
            rows = await db.fetch_chunks_by_hashes([h for h, _ in candidates])
            hash_to_uuid = {int(r["faiss_hash"]): str(r["id"]) for r in rows}
            hash_to_source = {int(r["faiss_hash"]): str(r["source_id"]) for r in rows}
            hash_to_doc = {int(r["faiss_hash"]): str(r["document_id"]) for r in rows}
        elif name in ("event_titles", "event_contents", "event_summaries"):
            rows = await db.fetch_events_by_hashes([h for h, _ in candidates])
            hash_to_uuid = {int(r["faiss_hash"]): str(r["id"]) for r in rows}
            hash_to_source = {int(r["faiss_hash"]): str(r["source_id"]) for r in rows}
            hash_to_doc = {int(r["faiss_hash"]): str(r["document_id"]) for r in rows}
        else:  # entities
            rows = await db.fetch_entities_by_hashes([h for h, _ in candidates])
            hash_to_uuid = {int(r["faiss_hash"]): str(r["id"]) for r in rows}
            hash_to_source = {}  # entities 不按 source 过滤
            hash_to_doc = {}

        # 3. 过滤 + 组装结果
        src_filter = set(source_ids) if source_ids and name != "entities" else None
        doc_filter = set(document_ids) if document_ids and name != "entities" else None
        hits: list[tuple[str, float]] = []
        for h, sim in candidates:
            uid = hash_to_uuid.get(h)
            if uid is None:
                continue  # DB 中查不到（可能已被删除但 FAISS 未同步）
            if src_filter is not None and hash_to_source.get(h) not in src_filter:
                continue
            if doc_filter is not None and hash_to_doc.get(h) not in doc_filter:
                continue
            hits.append((uid, sim))
            if len(hits) >= top_k:
                break
        log.debug("向量查询 collection={} top_k={} 候选={} 命中={}",
                  name, top_k, len(candidates), len(hits))
        return hits

    async def adelete_by_source(self, source_id: str) -> None:
        await self._adelete_by_source_impl(source_id)

    async def adelete_by_document(self, source_id: str, document_id: str) -> None:
        await self._adelete_by_document_impl(source_id, document_id)

    async def aget_embeddings(self, name: Collection, ids: list[str]) -> dict[str, list[float]]:
        # FAISS reconstruct 是同步内存操作，直接走 to_thread
        if not ids:
            return {}
        return await asyncio.to_thread(self.get_embeddings, name, ids)

    # ---------------- 异步实现内部方法 ----------------

    async def _adelete_by_source_impl(self, source_id: str) -> None:
        """按 source_id 删除：从映射表反查 hash → faiss remove_ids。"""
        import faiss
        db = self._require_db()
        # chunks + events 都有 source_id，entities 不删
        for name in ("chunks", "event_titles", "event_contents", "event_summaries"):
            map_table = _COLL_TO_MAP_TABLE[name]
            hashes = await db.fetch_hashes_by_source(map_table, source_id)
            if not hashes:
                continue
            with self._lock:
                store = self._stores[name]
                selector = faiss.IDSelectorBatch(hashes)
                try:
                    removed = store.remove_ids(selector)
                    if removed > 0:
                        self._persist(name)
                    log.info("向量删除 collection={} source_id={} hash数={} 实际删除={}",
                             name, source_id, len(hashes), removed)
                except Exception as e:
                    log.error("向量删除失败 collection={} source_id={} err={}",
                              name, source_id, e)

    async def _adelete_by_document_impl(self, source_id: str, document_id: str) -> None:
        import faiss
        db = self._require_db()
        for name in ("chunks", "event_titles", "event_contents", "event_summaries"):
            map_table = _COLL_TO_MAP_TABLE[name]
            hashes = await db.fetch_hashes_by_source_document(map_table, source_id, document_id)
            if not hashes:
                continue
            with self._lock:
                store = self._stores[name]
                selector = faiss.IDSelectorBatch(hashes)
                try:
                    removed = store.remove_ids(selector)
                    if removed > 0:
                        self._persist(name)
                    log.info("向量删除 collection={} source_id={} document_id={} 实际删除={}",
                             name, source_id, document_id, removed)
                except Exception as e:
                    log.error("向量删除失败 collection={} source_id={} document_id={} err={}",
                              name, source_id, document_id, e)

    # ---------------- 一致性校验 ----------------

    async def verify_integrity(self) -> dict[str, dict[str, Any]]:
        """FAISS ↔ DB 映射表一致性对账。

        对每个 collection 做集合差集：
          - FAISS 多出的 hash = 孤儿向量（应删未删，空间泄漏）
          - 映射表多出的 hash = 丢失向量（应存未存，检索缺失）

        Returns:
            {
                collection_name: {
                    "faiss_count": int,      # FAISS 中向量总数
                    "db_count": int,      # 映射表中记录数
                    "match": int,            # 双方一致的数量
                    "orphans_in_faiss": int, # FAISS 有但映射表无（脏向量）
                    "missing_in_faiss": int, # 映射表有但 FAISS 无（丢失向量）
                    "sample_orphans": list[int],   # 孤儿 hash 样本（前 10 个）
                    "sample_missing": list[int],   # 丢失 hash 样本（前 10 个）
                },
                ...
            }
        """
        import faiss
        db = self._require_db()
        report: dict[str, dict[str, Any]] = {}

        for name in _COLLECTIONS:
            map_table = _COLL_TO_MAP_TABLE[name]

            # 1. 取 FAISS 所有 hash
            with self._lock:
                store = self._stores[name]
                faiss_hashes: set[int] = set()
                try:
                    # IndexIDMap2 的 id_map 是 DirectMap 对象，vector_to_array 取出所有 id
                    ids_arr = faiss.vector_to_array(store.id_map)
                    faiss_hashes = set(int(x) for x in ids_arr.tolist())
                except Exception as e:
                    log.error("FAISS 读取 id_map 失败 collection={} err={}", name, e)

            # 2. 取映射表所有 faiss_hash
            db_hashes_list = await db.fetch_all_hashes(map_table)
            db_hashes: set[int] = set(db_hashes_list)

            # 3. 集合差集
            orphans = faiss_hashes - db_hashes       # FAISS 有但映射表无
            missing = db_hashes - faiss_hashes       # 映射表有但 FAISS 无
            match = faiss_hashes & db_hashes

            report[name] = {
                "faiss_count": len(faiss_hashes),
                "db_count": len(db_hashes),
                "match": len(match),
                "orphans_in_faiss": len(orphans),
                "missing_in_faiss": len(missing),
                "sample_orphans": sorted(orphans)[:10],
                "sample_missing": sorted(missing)[:10],
            }

            log.info(
                "对账 collection={} faiss={} db={} 一致={} 孤儿(FAISS多)={} 丢失(FAISS缺)={}",
                name, len(faiss_hashes), len(db_hashes), len(match),
                len(orphans), len(missing))

        return report

    async def repair_orphans(self, dry_run: bool = True) -> dict[str, int]:
        """清理 FAISS 中的孤儿向量（映射表已删但 FAISS 仍存在）。

        Args:
            dry_run: True 只统计不删除，False 实际执行 remove_ids

        Returns:
            {collection_name: removed_count, ...}
        """
        import faiss
        report = await self.verify_integrity()
        result: dict[str, int] = {}

        for name in _COLLECTIONS:
            stat = report[name]
            if stat["orphans_in_faiss"] == 0:
                result[name] = 0
                continue

            # 取全部孤儿 hash（不只是 sample）
            with self._lock:
                store = self._stores[name]
                try:
                    ids_arr = faiss.vector_to_array(store.id_map)
                    faiss_hashes = set(int(x) for x in ids_arr.tolist())
                except Exception:
                    result[name] = 0
                    continue

            db = self._require_db()
            map_table = _COLL_TO_MAP_TABLE[name]
            db_hashes_list = await db.fetch_all_hashes(map_table)
            db_hashes = set(db_hashes_list)
            orphans = list(faiss_hashes - db_hashes)

            if not orphans:
                result[name] = 0
                continue

            if dry_run:
                log.info("[dry-run] 孤儿清理 collection={} 待删={}",
                         name, len(orphans))
                result[name] = len(orphans)
                continue

            with self._lock:
                selector = faiss.IDSelectorBatch(orphans)
                try:
                    removed = store.remove_ids(selector)
                    if removed > 0:
                        self._persist(name)
                    result[name] = int(removed)
                    log.info("孤儿清理完成 collection={} 删除={}",
                             name, removed)
                except Exception as e:
                    log.error("孤儿清理失败 collection={} err={}", name, e)
                    result[name] = 0

        return result

    # ---------------- 工具 ----------------

    @staticmethod
    def _async_run_sync(func, *args):
        """在同步上下文中调用异步方法（用 asyncio.run）。若已有 loop，抛错提示用异步方法。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            raise RuntimeError(
                f"FAISS 后端同步方法 {func.__name__} 在异步上下文中调用，请改用对应的异步方法（a前缀）")
        return asyncio.run(func(*args))