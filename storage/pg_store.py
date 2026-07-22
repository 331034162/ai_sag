"""PostgreSQL 异步存储层：事件-实体超边的 CRUD 与检索查询。

基于 asyncpg 连接池，全链路异步。
核心：实体按 (type, normalized_name) 全局共享，通过 event_entities 超边串联跨文档事件。

与 MysqlStore 接口完全一致，通过 create_db_store() 工厂函数按配置自动选择。
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import asyncpg

from ..base import Chunk, Entity, Event, ExtractedEntity, ExtractedEvent
from ..base.logger import get_logger

log = get_logger()

_SCHEMA_PATH = Path(__file__).parent / "schema_pg.sql"


def uuid_to_int64(uuid_str: str) -> int:
    """UUID 字符串 → int64 hash（用于 FAISS IndexIDMap2 的 id）。

    使用 blake2b 8 字节摘要，转为有符号 int64。
    冲突概率：单表 10 亿条数据下约 1.16e-10（极低），唯一索引兜底。
    """
    h = hashlib.blake2b(uuid_str.encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big", signed=True)


class PgStore:
    """PostgreSQL 异步存储，接口与 MysqlStore 完全一致。"""

    def __init__(self, host: str, port: int, user: str, password: str, database: str, *,
                 pool_size: int = 10, max_overflow: int = 5,
                 pool_timeout: float = 30.0, pool_recycle: int = 3600,
                 faiss_map_enabled: bool = False) -> None:
        self._dsn = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool_timeout = pool_timeout
        self._pool_recycle = pool_recycle
        self._pool: asyncpg.Pool | None = None
        self._faiss_map_enabled = faiss_map_enabled

    # ---------------- 连接管理 ----------------

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                dsn=self._dsn,
                min_size=1,
                max_size=self._pool_size + self._max_overflow,
                command_timeout=self._pool_timeout,
            )
        return self._pool

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[asyncpg.Connection]:
        """异步事务上下文：自动 commit/rollback，连接用完归还连接池。"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                yield conn

    async def _execute(self, sql: str, *params) -> int:
        """执行单条写 SQL，返回 rowcount。"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(sql, *params)
        # asyncpg execute 返回 "INSERT 0 1" / "UPDATE 3" 等状态字符串
        parts = result.split()
        return int(parts[-1]) if parts and parts[-1].isdigit() else 0

    async def _fetchall(self, sql: str, *params) -> list[dict]:
        """查询所有行。"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
            return [dict(r) for r in rows]

    async def _fetchone(self, sql: str, *params) -> dict | None:
        """查询单行。"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(sql, *params)
            return dict(row) if row else None

    @staticmethod
    def _ph(start: int, count: int) -> str:
        """生成 PostgreSQL 编号占位符：$start, $(start+1), ..."""
        return ",".join([f"${i}" for i in range(start, start + count)])

    async def ensure_schema(self) -> None:
        """执行 DDL 建表（幂等，IF NOT EXISTS）。"""
        sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        stmts = self._split_sql(sql)
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            for stmt in stmts:
                if stmt.strip():
                    await conn.execute(stmt)

    @staticmethod
    def _split_sql(sql: str) -> list[str]:
        """按分号分割 SQL 语句，但跳过 $$ ... $$ 函数体内的分号。"""
        # 移除注释行
        lines = []
        for line in sql.split("\n"):
            if not line.strip().startswith("--"):
                lines.append(line)
        text = "\n".join(lines)

        result = []
        in_dollar_quote = False
        current: list[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            # 追踪 $$ 定界符状态（函数体开始/结束）
            if "$$" in stripped:
                in_dollar_quote = not in_dollar_quote
            current.append(line)
            if stripped.endswith(";") and not in_dollar_quote:
                result.append("\n".join(current))
                current = []
        if current:
            result.append("\n".join(current))
        return [s for s in result if s.strip()]

    async def ping(self) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("SELECT 1")

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def mark_vector_synced(self, source_id: str, synced: bool) -> None:
        await self._execute(
            "UPDATE aisag_sources SET vector_synced=$1 WHERE id=$2",
            (1 if synced else 0), source_id,
        )

    async def find_unsynced_sources(self) -> list[str]:
        rows = await self._fetchall(
            "SELECT id FROM aisag_sources WHERE vector_synced = 0"
        )
        return [str(r["id"]) for r in rows]

    async def list_all_source_ids(self) -> list[str]:
        rows = await self._fetchall("SELECT id FROM aisag_sources")
        return [str(r["id"]) for r in rows]

    async def list_all_entity_ids(self) -> list[str]:
        rows = await self._fetchall("SELECT id FROM aisag_entities")
        return [str(r["id"]) for r in rows]

    async def check_duplicate(self, name: str, md5: str) -> str | None:
        r = await self._fetchone(
            "SELECT id FROM aisag_sources "
            "WHERE name=$1 AND md5=$2 AND md5 != '' AND archived_at IS NULL",
            name, md5,
        )
        return str(r["id"]) if r else None

    # ---------------- 写入 ----------------

    async def upsert_source(self, source_id: str, name: str, description: str = "",
                            md5: str = "") -> None:
        await self._execute(
            "INSERT INTO aisag_sources (id, name, description, md5, vector_synced) "
            "VALUES ($1,$2,$3,$4,0) "
            "ON CONFLICT (id) DO UPDATE SET "
            "name=EXCLUDED.name, description=EXCLUDED.description, "
            "md5=EXCLUDED.md5, vector_synced=EXCLUDED.vector_synced",
            source_id, name, description, md5,
        )

    async def insert_document(self, doc_id: str, source_id: str, title: str,
                              content: str, status: str = "COMPLETED") -> None:
        await self._execute(
            "INSERT INTO aisag_documents (id, source_id, title, content, status) "
            "VALUES ($1,$2,$3,$4,$5)",
            doc_id, source_id, title, content, status,
        )

    async def insert_chunk(self, chunk: Chunk) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO aisag_chunks (id, source_id, document_id, rank_index, heading, content) "
                    "VALUES ($1,$2,$3,$4,$5,$6)",
                    chunk.id, chunk.source_id, chunk.document_id, chunk.rank_index,
                    chunk.heading, chunk.content,
                )
                if self._faiss_map_enabled:
                    await conn.execute(
                        "INSERT INTO faiss_chunks_map (faiss_hash, uuid, source_id, document_id) "
                        "VALUES ($1,$2,$3,$4) "
                        "ON CONFLICT (faiss_hash) DO UPDATE SET "
                        "uuid=EXCLUDED.uuid, source_id=EXCLUDED.source_id, document_id=EXCLUDED.document_id",
                        uuid_to_int64(chunk.id), chunk.id, chunk.source_id, chunk.document_id,
                    )

    async def insert_event(self, event: Event) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO aisag_events (id, source_id, document_id, chunk_id, rank_index, "
                    "title, summary, content) "
                    "VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
                    event.id, event.source_id, event.document_id, event.chunk_id, event.rank_index,
                    event.title, event.summary, event.content,
                )
                if self._faiss_map_enabled:
                    await conn.execute(
                        "INSERT INTO faiss_events_map (faiss_hash, uuid, source_id, document_id) "
                        "VALUES ($1,$2,$3,$4) "
                        "ON CONFLICT (faiss_hash) DO UPDATE SET "
                        "uuid=EXCLUDED.uuid, source_id=EXCLUDED.source_id, document_id=EXCLUDED.document_id",
                        uuid_to_int64(event.id), event.id, event.source_id, event.document_id,
                    )

    async def upsert_entity(self, entity: ExtractedEntity) -> str:
        eid = str(uuid.uuid4())
        norm = self._normalize(entity.name)
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO aisag_entities (id, entity_type, name, normalized_name, description) "
                    "VALUES ($1,$2,$3,$4,$5) "
                    "ON CONFLICT (entity_type, normalized_name) DO UPDATE SET "
                    "name=EXCLUDED.name",
                    eid, entity.type, entity.name, norm, entity.description,
                )
                row = await conn.fetchrow(
                    "SELECT id FROM aisag_entities WHERE entity_type=$1 AND normalized_name=$2",
                    entity.type, norm,
                )
                actual_id = str(row["id"]) if row else eid
                if self._faiss_map_enabled:
                    await conn.execute(
                        "INSERT INTO faiss_entities_map (faiss_hash, uuid) "
                        "VALUES ($1,$2) "
                        "ON CONFLICT (faiss_hash) DO UPDATE SET uuid=EXCLUDED.uuid",
                        uuid_to_int64(actual_id), actual_id,
                    )
                return actual_id

    async def link_event_entity(self, event_id: str, entity_id: str,
                                weight: float = 1.0, description: str = "") -> None:
        await self._execute(
            "INSERT INTO aisag_event_entities (id, event_id, entity_id, weight, description) "
            "VALUES ($1,$2,$3,$4,$5) ON CONFLICT (event_id, entity_id) DO NOTHING",
            str(uuid.uuid4()), event_id, entity_id, weight, description,
        )

    # ---------------- 批量事务写入 ----------------

    async def persist_source(self, source_id: str, source_name: str, file_type: str,
                             document_id: str, doc_title: str, doc_content: str,
                             chunks: list[Chunk], events: list, *,
                             md5: str = "") -> tuple[list, dict[tuple[str, str], str]]:
        if len(chunks) != len(events):
            raise ValueError(
                f"chunks 和 events 数量不一致：chunks={len(chunks)} events={len(events)}，"
                f"无法保证 1:1 对应关系")
        event_records: list = []
        seen_entities: dict[tuple[str, str], str] = {}

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "INSERT INTO aisag_sources (id, name, description, md5, vector_synced) "
                    "VALUES ($1,$2,$3,$4,0) "
                    "ON CONFLICT (id) DO UPDATE SET "
                    "name=EXCLUDED.name, description=EXCLUDED.description, md5=EXCLUDED.md5",
                    source_id, source_name, f"file_type={file_type}", md5,
                )
                await conn.execute(
                    "INSERT INTO aisag_documents (id, source_id, title, content, status) "
                    "VALUES ($1,$2,$3,$4,$5)",
                    document_id, source_id, doc_title, doc_content, "COMPLETED",
                )
                # 批量插入 chunks
                await conn.executemany(
                    "INSERT INTO aisag_chunks (id, source_id, document_id, rank_index, heading, content) "
                    "VALUES ($1,$2,$3,$4,$5,$6)",
                    [(c.id, c.source_id, c.document_id, c.rank_index, c.heading, c.content)
                     for c in chunks],
                )
                if self._faiss_map_enabled:
                    await conn.executemany(
                        "INSERT INTO faiss_chunks_map (faiss_hash, uuid, source_id, document_id) "
                        "VALUES ($1,$2,$3,$4) "
                        "ON CONFLICT (faiss_hash) DO UPDATE SET "
                        "uuid=EXCLUDED.uuid, source_id=EXCLUDED.source_id, document_id=EXCLUDED.document_id",
                        [(uuid_to_int64(c.id), c.id, c.source_id, c.document_id) for c in chunks],
                    )
                # 构造 event 记录并批量插入
                for idx, (chunk, ev) in enumerate(zip(chunks, events)):
                    event = self.to_event_record(
                        ev, source_id=source_id, document_id=document_id,
                        chunk_id=chunk.id, rank_index=idx,
                    )
                    event_records.append(event)
                await conn.executemany(
                    "INSERT INTO aisag_events (id, source_id, document_id, chunk_id, rank_index, "
                    "title, summary, content) "
                    "VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
                    [(e.id, e.source_id, e.document_id, e.chunk_id, e.rank_index,
                      e.title, e.summary, e.content) for e in event_records],
                )
                if self._faiss_map_enabled:
                    await conn.executemany(
                        "INSERT INTO faiss_events_map (faiss_hash, uuid, source_id, document_id) "
                        "VALUES ($1,$2,$3,$4) "
                        "ON CONFLICT (faiss_hash) DO UPDATE SET "
                        "uuid=EXCLUDED.uuid, source_id=EXCLUDED.source_id, document_id=EXCLUDED.document_id",
                        [(uuid_to_int64(e.id), e.id, e.source_id, e.document_id) for e in event_records],
                    )

                new_entity_count = 0
                reused_entity_count = 0
                for idx, (chunk, ev) in enumerate(zip(chunks, events)):
                    event = event_records[idx]
                    for e in ev.entities:
                        norm = self._normalize(e.name)
                        key = (e.type, norm)
                        if key in seen_entities:
                            eid = seen_entities[key]
                        else:
                            new_eid = str(uuid.uuid4())
                            await conn.execute(
                                "INSERT INTO aisag_entities (id, entity_type, name, normalized_name, description) "
                                "VALUES ($1,$2,$3,$4,$5) "
                                "ON CONFLICT (entity_type, normalized_name) DO UPDATE SET "
                                "name=EXCLUDED.name",
                                new_eid, e.type, e.name, norm, e.description,
                            )
                            row = await conn.fetchrow(
                                "SELECT id FROM aisag_entities WHERE entity_type=$1 AND normalized_name=$2",
                                e.type, norm,
                            )
                            actual_id = str(row["id"]) if row else new_eid
                            if actual_id == new_eid:
                                new_entity_count += 1
                            else:
                                reused_entity_count += 1
                            eid = actual_id
                            seen_entities[key] = eid
                            if self._faiss_map_enabled:
                                await conn.execute(
                                    "INSERT INTO faiss_entities_map (faiss_hash, uuid) "
                                    "VALUES ($1,$2) "
                                    "ON CONFLICT (faiss_hash) DO UPDATE SET uuid=EXCLUDED.uuid",
                                    uuid_to_int64(eid), eid,
                                )
                        await conn.execute(
                            "INSERT INTO aisag_event_entities (id, event_id, entity_id, weight, description) "
                            "VALUES ($1,$2,$3,$4,$5) ON CONFLICT DO NOTHING",
                            str(uuid.uuid4()), event.id, eid, e.weight, e.role,
                        )
                log.info("实体去重统计 新实体={} 复用已有={} 总引用={}",
                         new_entity_count, reused_entity_count,
                         new_entity_count + reused_entity_count)

        return event_records, seen_entities

    # ---------------- 检索查询 ----------------

    async def get_event_ids_by_entity_ids(self, entity_ids: list[str], source_ids: list[str] | None,
                                          exclude: list[str] | None = None) -> list[str]:
        if not entity_ids:
            return []
        params: list = list(entity_ids)
        n = 1
        ph_e = self._ph(n, len(entity_ids))
        n += len(entity_ids)

        if source_ids:
            ph_s = self._ph(n, len(source_ids))
            n += len(source_ids)
            params.extend(source_ids)
            sql = (f"SELECT DISTINCT ee.event_id FROM aisag_event_entities ee "
                   f"INNER JOIN aisag_events ev ON ev.id = ee.event_id "
                   f"WHERE ee.entity_id IN ({ph_e}) AND ev.source_id IN ({ph_s})")
        else:
            sql = (f"SELECT DISTINCT ee.event_id FROM aisag_event_entities ee "
                   f"WHERE ee.entity_id IN ({ph_e})")
        rows = await self._fetchall(sql, *params)
        event_ids = [r["event_id"] for r in rows]
        if exclude:
            exclude_set = set(exclude) if not isinstance(exclude, set) else exclude
            event_ids = [eid for eid in event_ids if eid not in exclude_set]
        return event_ids

    async def search_entities_by_name(self, names: list[str], source_ids: list[str] | None) -> list[Entity]:
        if not names:
            return []
        normalized = [self._normalize(n) for n in names]
        params: list = list(normalized)
        n = 1
        ph_n = self._ph(n, len(normalized))
        n += len(normalized)

        if source_ids:
            ph_s = self._ph(n, len(source_ids))
            params.extend(source_ids)
            sql = (f"SELECT en.id, en.entity_type, en.name, en.normalized_name, en.description "
                   f"FROM aisag_entities en "
                   f"WHERE en.normalized_name IN ({ph_n}) "
                   f"AND EXISTS ("
                   f"SELECT 1 FROM aisag_event_entities ee "
                   f"JOIN aisag_events ev ON ev.id = ee.event_id "
                   f"WHERE ee.entity_id = en.id AND ev.source_id IN ({ph_s})"
                   f")")
        else:
            sql = (f"SELECT id, entity_type, name, normalized_name, description "
                   f"FROM aisag_entities en "
                   f"WHERE en.normalized_name IN ({ph_n}) "
                   f"AND EXISTS (SELECT 1 FROM aisag_event_entities ee WHERE ee.entity_id = en.id)")
        rows = await self._fetchall(sql, *params)
        return [self._row_to_entity(r) for r in rows]

    async def get_entity_degrees(self, entity_ids: list[str]) -> dict[str, int]:
        if not entity_ids:
            return {}
        ph = self._ph(1, len(entity_ids))
        sql = (f"SELECT entity_id, COUNT(*) AS degree "
               f"FROM aisag_event_entities "
               f"WHERE entity_id IN ({ph}) "
               f"GROUP BY entity_id")
        rows = await self._fetchall(sql, *entity_ids)
        return {r["entity_id"]: int(r["degree"]) for r in rows}

    async def filter_entity_ids_by_sources(self, entity_ids: list[str],
                                           source_ids: list[str] | None) -> list[str]:
        if not entity_ids:
            return []
        if not source_ids:
            return list(entity_ids)
        params: list = list(entity_ids)
        n = 1
        ph_e = self._ph(n, len(entity_ids))
        n += len(entity_ids)
        ph_s = self._ph(n, len(source_ids))
        params.extend(source_ids)
        sql = (f"SELECT DISTINCT ee.entity_id FROM aisag_event_entities ee "
               f"WHERE ee.entity_id IN ({ph_e}) "
               f"AND EXISTS (SELECT 1 FROM aisag_events e "
               f"            WHERE e.id = ee.event_id AND e.source_id IN ({ph_s}))")
        rows = await self._fetchall(sql, *params)
        return [r["entity_id"] for r in rows]

    async def get_events_by_ids(self, event_ids: list[str],
                                source_ids: list[str] | None = None) -> list[Event]:
        if not event_ids:
            return []
        params: list = list(event_ids)
        n = 1
        ph = self._ph(n, len(event_ids))
        n += len(event_ids)
        where = [f"id IN ({ph})"]
        if source_ids:
            ph_s = self._ph(n, len(source_ids))
            params.extend(source_ids)
            where.append(f"source_id IN ({ph_s})")
        sql = (f"SELECT id, source_id, document_id, chunk_id, rank_index, title, summary, content "
               f"FROM aisag_events WHERE " + " AND ".join(where))
        rows = await self._fetchall(sql, *params)
        row_map = {r["id"]: r for r in rows}
        ordered = [row_map[eid] for eid in event_ids if eid in row_map]
        matched_ids = [r["id"] for r in ordered]
        roles_map = await self._entity_roles_of_events(matched_ids) if matched_ids else {}
        result = []
        for r in ordered:
            e = self._row_to_event(r)
            role_weight_map = roles_map.get(e.id, {})
            e.entity_ids = list(role_weight_map.keys())
            e.entity_roles = {eid: role for eid, (role, _) in role_weight_map.items()}
            e.entity_weights = {eid: weight for eid, (_, weight) in role_weight_map.items()}
            result.append(e)
        return result

    async def _entity_roles_of_events(self, event_ids: list[str]) -> dict[str, dict[str, tuple[str, float]]]:
        if not event_ids:
            return {}
        ph = self._ph(1, len(event_ids))
        rows = await self._fetchall(
            f"SELECT event_id, entity_id, description, weight "
            f"FROM aisag_event_entities WHERE event_id IN ({ph})",
            *event_ids,
        )
        result: dict[str, dict[str, tuple[str, float]]] = {}
        for r in rows:
            weight = r["weight"]
            weight = float(weight) if weight is not None else 0.5
            result.setdefault(str(r["event_id"]), {})[str(r["entity_id"])] = (
                r["description"] or "",
                weight,
            )
        return result

    async def get_entity_names_by_ids(self, entity_ids: list[str]) -> dict[str, str]:
        if not entity_ids:
            return {}
        ph = self._ph(1, len(entity_ids))
        rows = await self._fetchall(
            f"SELECT id, name FROM aisag_entities WHERE id IN ({ph})",
            *entity_ids,
        )
        return {str(r["id"]): str(r["name"]) for r in rows}

    async def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        ph = self._ph(1, len(chunk_ids))
        sql = (f"SELECT id, source_id, document_id, rank_index, heading, content "
               f"FROM aisag_chunks WHERE id IN ({ph})")
        rows = await self._fetchall(sql, *chunk_ids)
        row_map = {r["id"]: r for r in rows}
        return [self._row_to_chunk(row_map[cid]) for cid in chunk_ids if cid in row_map]

    async def get_chunk_ids_by_event_ids(self, event_ids: list[str]) -> dict[str, str]:
        if not event_ids:
            return {}
        ph = self._ph(1, len(event_ids))
        sql = f"SELECT id, chunk_id FROM aisag_events WHERE id IN ({ph})"
        rows = await self._fetchall(sql, *event_ids)
        return {r["id"]: r["chunk_id"] for r in rows}

    async def delete_by_source(self, source_id: str) -> tuple[int, list[str]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                rows = await conn.fetch(
                    "SELECT DISTINCT ee.entity_id FROM aisag_event_entities ee "
                    "JOIN aisag_events e ON e.id = ee.event_id "
                    "WHERE e.source_id=$1",
                    source_id,
                )
                affected_entity_ids = [str(r["entity_id"]) for r in rows]

                await conn.execute(
                    "DELETE FROM aisag_event_entities WHERE event_id IN "
                    "(SELECT id FROM aisag_events WHERE source_id=$1)",
                    source_id,
                )
                result = await conn.execute(
                    "DELETE FROM aisag_events WHERE source_id=$1",
                    source_id,
                )
                n = int(result.split()[-1]) if result else 0
                await conn.execute("DELETE FROM aisag_chunks WHERE source_id=$1", source_id)
                await conn.execute("DELETE FROM aisag_documents WHERE source_id=$1", source_id)
                await conn.execute("DELETE FROM aisag_sources WHERE id=$1", source_id)
                if self._faiss_map_enabled:
                    await conn.execute("DELETE FROM faiss_chunks_map WHERE source_id=$1", source_id)
                    await conn.execute("DELETE FROM faiss_events_map WHERE source_id=$1", source_id)

                orphan_ids: list[str] = []
                if affected_entity_ids:
                    ph_aff = self._ph(1, len(affected_entity_ids))
                    orphan_rows = await conn.fetch(
                        f"SELECT e.id, e.entity_type, e.name FROM aisag_entities e "
                        f"WHERE e.id IN ({ph_aff}) "
                        f"AND NOT EXISTS ("
                        f"  SELECT 1 FROM aisag_event_entities ee WHERE ee.entity_id = e.id"
                        f")",
                        *affected_entity_ids,
                    )
                    orphan_ids = [str(r["id"]) for r in orphan_rows]
                    if orphan_ids:
                        ph_o = self._ph(1, len(orphan_ids))
                        await conn.execute(
                            f"DELETE FROM aisag_entities WHERE id IN ({ph_o})",
                            *orphan_ids,
                        )
                        if self._faiss_map_enabled:
                            await conn.execute(
                                f"DELETE FROM faiss_entities_map WHERE uuid IN ({ph_o})",
                                *orphan_ids,
                            )

                return n, orphan_ids

    async def hard_delete_soft_deleted_events(self) -> tuple[list[str], list[str]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                rows = await conn.fetch(
                    "SELECT id FROM aisag_events WHERE deleted_at IS NOT NULL"
                )
                event_ids = [str(r["id"]) for r in rows]
                if not event_ids:
                    return [], []

                log.info("硬删除扫描：发现 {} 条软删除事件，开始物理清理", len(event_ids))

                ph_ev = self._ph(1, len(event_ids))
                affected = await conn.fetch(
                    f"SELECT DISTINCT ee.entity_id FROM aisag_event_entities ee "
                    f"WHERE ee.event_id IN ({ph_ev})",
                    *event_ids,
                )
                affected_entity_ids = [str(r["entity_id"]) for r in affected]

                await conn.execute(
                    f"DELETE FROM aisag_event_entities WHERE event_id IN ({ph_ev})",
                    *event_ids,
                )
                result = await conn.execute(
                    f"DELETE FROM aisag_events WHERE id IN ({ph_ev})",
                    *event_ids,
                )
                n = int(result.split()[-1]) if result else 0
                if self._faiss_map_enabled:
                    await conn.execute(
                        f"DELETE FROM faiss_events_map WHERE uuid IN ({ph_ev})",
                        *event_ids,
                    )

                orphan_ids: list[str] = []
                if affected_entity_ids:
                    ph_aff = self._ph(1, len(affected_entity_ids))
                    orphans = await conn.fetch(
                        f"SELECT e.id FROM aisag_entities e WHERE e.id IN ({ph_aff}) "
                        f"AND NOT EXISTS ("
                        f"  SELECT 1 FROM aisag_event_entities ee WHERE ee.entity_id = e.id"
                        f")",
                        *affected_entity_ids,
                    )
                    orphan_ids = [str(r["id"]) for r in orphans]
                    if orphans:
                        ph_o = self._ph(1, len(orphan_ids))
                        await conn.execute(
                            f"DELETE FROM aisag_entities WHERE id IN ({ph_o})",
                            *orphan_ids,
                        )
                        if self._faiss_map_enabled:
                            await conn.execute(
                                f"DELETE FROM faiss_entities_map WHERE uuid IN ({ph_o})",
                                *orphan_ids,
                            )

                log.info("硬删除完成 events={} 孤儿entities={}", n, len(orphan_ids))
                return event_ids, orphan_ids

    # ---------------- 管理类查询 ----------------

    async def list_sources(self, *, include_archived: bool = False, keyword: str | None = None,
                           limit: int = 100, offset: int = 0) -> list[dict]:
        where = [] if include_archived else ["archived_at IS NULL"]
        params: list = []
        param_idx = 1
        if keyword:
            params.append(f"%{keyword}%")
            where.append(f"name LIKE ${param_idx}")
            param_idx += 1
        where_clause = (" WHERE " + " AND ".join(where)) if where else ""
        params.extend([limit, offset])
        sql = (f"SELECT id, name, description, md5, created_at, updated_at, archived_at "
               f"FROM aisag_sources{where_clause} ORDER BY created_at DESC "
               f"LIMIT ${param_idx} OFFSET ${param_idx + 1}")
        rows = await self._fetchall(sql, *params)
        return [self._row_to_source(r) for r in rows]

    async def get_source(self, source_id: str) -> dict | None:
        r = await self._fetchone(
            "SELECT id, name, description, md5, created_at, updated_at, archived_at "
            "FROM aisag_sources WHERE id=$1", source_id)
        return self._row_to_source(r) if r else None

    async def get_source_names_by_ids(self, source_ids: list[str]) -> dict[str, str]:
        if not source_ids:
            return {}
        ph = self._ph(1, len(source_ids))
        rows = await self._fetchall(
            f"SELECT id, name FROM aisag_sources WHERE id IN ({ph})",
            *source_ids,
        )
        return {str(r["id"]): str(r["name"]) for r in rows}

    async def get_document_by_source(self, source_id: str) -> dict | None:
        r = await self._fetchone(
            "SELECT id, source_id, title, content, status, created_at "
            "FROM aisag_documents WHERE source_id=$1 ORDER BY created_at DESC LIMIT 1",
            source_id)
        if not r:
            return None
        return {
            "id": str(r["id"]), "source_id": r["source_id"], "title": r["title"],
            "content": r["content"], "status": r["status"],
            "created_at": str(r["created_at"]) if r.get("created_at") else None,
        }

    async def count_sources(self, *, include_archived: bool = False) -> int:
        where = "" if include_archived else " WHERE archived_at IS NULL"
        r = await self._fetchone(f"SELECT COUNT(*) AS c FROM aisag_sources{where}")
        return int(r["c"]) if r else 0

    async def update_source(self, source_id: str, *, name: str | None = None,
                            description: str | None = None) -> bool:
        sets, params = [], []
        idx = 1
        if name is not None:
            params.append(name)
            sets.append(f"name=${idx}")
            idx += 1
        if description is not None:
            params.append(description)
            sets.append(f"description=${idx}")
            idx += 1
        if not sets:
            return False
        params.append(source_id)
        return await self._execute(
            f"UPDATE aisag_sources SET {', '.join(sets)} WHERE id=${idx}", *params) > 0

    async def count_chunks_by_source(self, source_id: str) -> int:
        r = await self._fetchone(
            "SELECT COUNT(*) AS c FROM aisag_chunks WHERE source_id=$1", source_id)
        return int(r["c"]) if r else 0

    async def count_events_by_source(self, source_id: str) -> int:
        r = await self._fetchone(
            "SELECT COUNT(*) AS c FROM aisag_events WHERE source_id=$1",
            source_id)
        return int(r["c"]) if r else 0

    @staticmethod
    def _row_to_source(r) -> dict:
        return {
            "id": str(r["id"]), "name": r["name"], "description": r.get("description") or "",
            "md5": r.get("md5") or "",
            "created_at": str(r["created_at"]) if r.get("created_at") else None,
            "updated_at": str(r["updated_at"]) if r.get("updated_at") else None,
            "archived_at": str(r["archived_at"]) if r.get("archived_at") else None,
        }

    async def search_documents(self, keyword: str, *, source_ids: list[str] | None = None,
                               limit: int = 20, context_size: int = 80) -> list[dict]:
        kw = (keyword or "").strip()
        if not kw:
            return []
        like = f"%{kw}%"
        params: list = [like]
        sql = (
            "SELECT d.source_id, d.title, d.content, s.name AS source_name, d.created_at "
            "FROM aisag_documents d "
            "JOIN aisag_sources s ON s.id = d.source_id "
            "WHERE d.content LIKE $1 AND s.archived_at IS NULL"
        )
        idx = 2
        if source_ids:
            ph = self._ph(idx, len(source_ids))
            params.extend(source_ids)
            sql += f" AND d.source_id IN ({ph})"
            idx += len(source_ids)
        sql += f" ORDER BY d.created_at DESC LIMIT ${idx}"
        params.append(limit)
        rows = await self._fetchall(sql, *params)
        results: list[dict] = []
        for r in rows:
            content = r.get("content") or ""
            results.append({
                "source_id": r["source_id"], "title": r["title"],
                "source_name": r.get("source_name") or "",
                "snippet": self._snippet(content, kw, context_size),
                "created_at": str(r["created_at"]) if r.get("created_at") else None,
            })
        return results

    @staticmethod
    def _snippet(content: str, keyword: str, context_size: int) -> str:
        idx = content.lower().find(keyword.lower())
        if idx < 0:
            return content[: context_size * 2]
        start = max(0, idx - context_size)
        end = min(len(content), idx + len(keyword) + context_size)
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(content) else ""
        return prefix + content[start:end] + suffix

    # ---------------- 工具 ----------------

    @staticmethod
    def _normalize(name: str) -> str:
        return re.sub(r"\s+", "", name or "").strip().lower()

    @staticmethod
    def _row_to_entity(r) -> Entity:
        return Entity(id=str(r["id"]), type=r["entity_type"], name=r["name"],
                      normalized_name=r["normalized_name"], description=r.get("description") or "")

    @staticmethod
    def _row_to_event(r) -> Event:
        return Event(
            id=str(r["id"]), source_id=r["source_id"], document_id=r["document_id"],
            chunk_id=r["chunk_id"], rank_index=r["rank_index"], title=r["title"],
            summary=r.get("summary") or "", content=r["content"],
            entity_ids=[], score=0.0,
        )

    @staticmethod
    def _row_to_chunk(r) -> Chunk:
        return Chunk(
            id=str(r["id"]), document_id=r["document_id"], source_id=r["source_id"],
            rank_index=r["rank_index"], heading=r.get("heading") or "Introduction",
            content=r["content"],
        )

    # ---------------- FAISS hash 反查（仅 FAISS 后端使用）----------------

    async def fetch_chunks_by_hashes(self, hashes: list[int]) -> list[dict]:
        if not hashes:
            return []
        ph = self._ph(1, len(hashes))
        sql = (
            f"SELECT c.id, c.source_id, c.document_id, c.content, m.faiss_hash "
            f"FROM faiss_chunks_map m JOIN aisag_chunks c ON c.id = m.uuid "
            f"WHERE m.faiss_hash IN ({ph})"
        )
        return await self._fetchall(sql, *hashes)

    async def fetch_events_by_hashes(self, hashes: list[int]) -> list[dict]:
        if not hashes:
            return []
        ph = self._ph(1, len(hashes))
        sql = (
            f"SELECT e.id, e.source_id, e.document_id, e.title, e.summary, e.content, m.faiss_hash "
            f"FROM faiss_events_map m JOIN aisag_events e ON e.id = m.uuid "
            f"WHERE m.faiss_hash IN ({ph}) AND e.deleted_at IS NULL"
        )
        return await self._fetchall(sql, *hashes)

    async def fetch_entities_by_hashes(self, hashes: list[int]) -> list[dict]:
        if not hashes:
            return []
        ph = self._ph(1, len(hashes))
        sql = (
            f"SELECT e.id, e.entity_type, e.name, e.normalized_name, e.description, m.faiss_hash "
            f"FROM faiss_entities_map m JOIN aisag_entities e ON e.id = m.uuid "
            f"WHERE m.faiss_hash IN ({ph})"
        )
        return await self._fetchall(sql, *hashes)

    async def fetch_source_ids_by_hashes(self, table: str, hashes: list[int]) -> dict[int, str]:
        if not hashes:
            return {}
        ph = self._ph(1, len(hashes))
        if table == "aisag_chunks":
            sql = f"SELECT faiss_hash, source_id FROM faiss_chunks_map WHERE faiss_hash IN ({ph})"
        elif table == "aisag_events":
            sql = f"SELECT faiss_hash, source_id FROM faiss_events_map WHERE faiss_hash IN ({ph})"
        else:
            raise ValueError(f"fetch_source_ids_by_hashes 不支持表 {table}（entities 无 source_id）")
        rows = await self._fetchall(sql, *hashes)
        return {int(r["faiss_hash"]): str(r["source_id"]) for r in rows}

    async def fetch_hashes_by_source(self, map_table: str, source_id: str) -> list[int]:
        """按 source_id 从 FAISS 映射表查 faiss_hash 列表（FAISS 后端专用）。"""
        rows = await self._fetchall(
            f"SELECT faiss_hash FROM {map_table} WHERE source_id=$1", source_id)
        return [int(r["faiss_hash"]) for r in rows]

    async def fetch_hashes_by_source_document(self, map_table: str, source_id: str,
                                              document_id: str) -> list[int]:
        """按 (source_id, document_id) 从 FAISS 映射表查 faiss_hash 列表（FAISS 后端专用）。"""
        rows = await self._fetchall(
            f"SELECT faiss_hash FROM {map_table} WHERE source_id=$1 AND document_id=$2",
            source_id, document_id)
        return [int(r["faiss_hash"]) for r in rows]

    async def fetch_all_hashes(self, map_table: str) -> list[int]:
        """查 FAISS 映射表全部 faiss_hash（一致性对账用，FAISS 后端专用）。"""
        rows = await self._fetchall(f"SELECT faiss_hash FROM {map_table}")
        return [int(r["faiss_hash"]) for r in rows]

    async def fetch_distinct_source_ids(self) -> list[str]:
        """查 aisag_chunks 表所有去重 source_id（FAISS 后端对账用）。"""
        rows = await self._fetchall("SELECT DISTINCT source_id FROM aisag_chunks")
        return [str(r["source_id"]) for r in rows]

    async def fetch_all_entity_ids(self) -> list[str]:
        """查 aisag_entities 表所有 id（FAISS 后端对账用）。"""
        rows = await self._fetchall("SELECT id FROM aisag_entities")
        return [str(r["id"]) for r in rows]

    def to_event_record(self, event: ExtractedEvent, *, source_id: str, document_id: str,
                        chunk_id: str, rank_index: int) -> Event:
        return Event(
            id=str(uuid.uuid4()), source_id=source_id, document_id=document_id,
            chunk_id=chunk_id, rank_index=rank_index, title=event.title,
            summary=event.summary, content=event.content,
            entity_ids=[], score=0.0,
        )