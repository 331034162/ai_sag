"""MySQL 异步存储层：事件-实体超边的 CRUD 与检索查询。

基于 aiomysql 连接池，全链路异步。
核心：实体按 (type, normalized_name) 全局共享，通过 event_entities 超边串联跨文档事件。
"""
from __future__ import annotations

import json
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import aiomysql

from ..base import Chunk, Entity, Event, ExtractedEntity, ExtractedEvent
from ..base.logger import get_logger

log = get_logger()

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class MysqlStore:
    def __init__(self, host: str, port: int, user: str, password: str, database: str, *,
                 pool_size: int = 10, max_overflow: int = 5,
                 pool_timeout: float = 30.0, pool_recycle: int = 3600) -> None:
        self._conn_kwargs = dict(host=host, port=port, user=user, password=password,
                                 db=database, charset="utf8mb4",
                                 autocommit=False)
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool_timeout = pool_timeout
        self._pool_recycle = pool_recycle
        self._pool: aiomysql.Pool | None = None

    async def _get_pool(self) -> aiomysql.Pool:
        if self._pool is None:
            self._pool = await aiomysql.create_pool(
                **self._conn_kwargs,
                minsize=1, maxsize=self._pool_size + self._max_overflow,
                pool_recycle=self._pool_recycle,
            )
        return self._pool

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiomysql.Connection]:
        """异步事务上下文：自动 commit/rollback，连接用完归还连接池。"""
        pool = await self._get_pool()
        conn = await pool.acquire()
        try:
            await conn.begin()
            yield conn
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
        finally:
            pool.release(conn)

    async def _execute(self, sql: str, params: list | None = None) -> int:
        """执行单条写 SQL，返回 rowcount。"""
        async with self.transaction() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(sql, params or [])
                return cur.rowcount

    async def _fetchall(self, sql: str, params: list | None = None) -> list[dict]:
        """查询所有行。"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(sql, params or [])
                return await cur.fetchall()

    async def _fetchone(self, sql: str, params: list | None = None) -> dict | None:
        """查询单行。"""
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(sql, params or [])
                return await cur.fetchone()

    async def ensure_schema(self) -> None:
        sql = _SCHEMA_PATH.read_text(encoding="utf-8")
        stmts = self._split_sql(sql)
        # 所有建表/改表均为 DDL，MySQL 会自动隐式提交，不应在显式事务中执行
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.cursor() as cur:
                for stmt in stmts:
                    if stmt.strip():
                        await cur.execute(stmt)
                # 兼容旧表：添加 md5 列（列已存在则跳过，错误码 1060 = Duplicate column name）
                try:
                    await cur.execute(
                        "ALTER TABLE aisag_sources ADD COLUMN md5 VARCHAR(32) NOT NULL DEFAULT ''")
                    log.info("ensure_schema: 已添加 md5 列")
                except Exception as e:
                    if getattr(e, 'args', [None])[0] == 1060:
                        log.info("ensure_schema: md5 列已存在，跳过")
                    else:
                        raise
                # 兼容旧表：删除旧普通索引
                try:
                    await cur.execute(
                        "ALTER TABLE aisag_sources DROP INDEX idx_aisag_source_name_md5")
                    log.info("ensure_schema: 已删除旧索引 idx_aisag_source_name_md5")
                except Exception:
                    pass
                # 兼容旧表：创建唯一索引（索引已存在则跳过，错误码 1061 = Duplicate key name）
                try:
                    await cur.execute(
                        "ALTER TABLE aisag_sources ADD UNIQUE KEY uq_aisag_source_name_md5 (name(128), md5)")
                    log.info("ensure_schema: 已创建唯一索引 uq_aisag_source_name_md5")
                except Exception as e:
                    code = getattr(e, 'args', [None])[0]
                    if code == 1061:
                        log.info("ensure_schema: 唯一索引已存在，跳过")
                    elif code == 1062:
                        log.warning(
                            "ensure_schema: 表中存在重复的 (name, md5) 数据，无法创建唯一索引。"
                            "请先手动清理重复行后重试。")
                    else:
                        raise
                # 兼容旧表：为事件-实体关联表添加 created_at 列（列已存在则跳过，错误码 1060）
                try:
                    await cur.execute(
                        "ALTER TABLE aisag_event_entities "
                        "ADD COLUMN created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP")
                    log.info("ensure_schema: 已添加 aisag_event_entities.created_at 列")
                except Exception as e:
                    if getattr(e, 'args', [None])[0] == 1060:
                        log.info("ensure_schema: aisag_event_entities.created_at 列已存在，跳过")
                    else:
                        raise
                # 兼容旧表：添加 (entity_id, event_id) 覆盖索引（索引已存在则跳过，错误码 1061）
                # 用于 get_event_ids_by_entity_ids 避免 idx_aisag_ee_entity 单列索引回表。
                try:
                    await cur.execute(
                        "ALTER TABLE aisag_event_entities "
                        "ADD INDEX idx_aisag_ee_entity_event (entity_id, event_id)")
                    log.info("ensure_schema: 已添加覆盖索引 idx_aisag_ee_entity_event")
                except Exception as e:
                    if getattr(e, 'args', [None])[0] == 1061:
                        log.info("ensure_schema: 覆盖索引 idx_aisag_ee_entity_event 已存在，跳过")
                    else:
                        raise
                # 兼容旧表：删除冗余单列索引（被联合索引覆盖，删除提升写入性能）
                # idx_aisag_ee_entity 被 idx_aisag_ee_entity_event (entity_id, event_id) 覆盖
                # idx_aisag_ee_event 被 uq_aisag_ee (event_id, entity_id) 覆盖
                for old_idx in ("idx_aisag_ee_entity", "idx_aisag_ee_event"):
                    try:
                        await cur.execute(
                            f"ALTER TABLE aisag_event_entities DROP INDEX {old_idx}")
                        log.info("ensure_schema: 已删除冗余索引 {}", old_idx)
                    except Exception as e:
                        if getattr(e, 'args', [None])[0] == 1091:  # Can't DROP INDEX
                            log.info("ensure_schema: 索引 {} 不存在，跳过", old_idx)
                        else:
                            raise

    @staticmethod
    def _split_sql(sql: str) -> list[str]:
        return [s.strip() for s in re.split(r";\s*\n", sql) if s.strip()]

    async def ping(self) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.ping(reconnect=True)

    async def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None

    async def mark_vector_synced(self, source_id: str, synced: bool) -> None:
        await self._execute(
            "UPDATE aisag_sources SET metadata = JSON_SET("
            "  COALESCE(metadata, '{}'), '$.vector_synced', %s) WHERE id = %s",
            (1 if synced else 0, source_id),
        )

    async def find_unsynced_sources(self) -> list[str]:
        rows = await self._fetchall(
            "SELECT id FROM aisag_sources "
            "WHERE COALESCE(JSON_EXTRACT(metadata, '$.vector_synced'), 0) = 0"
        )
        return [str(r["id"]) for r in rows]

    async def list_all_source_ids(self) -> list[str]:
        """列出 MySQL 中所有 source_id（含已软删除事件的）。"""
        rows = await self._fetchall("SELECT id FROM aisag_sources")
        return [str(r["id"]) for r in rows]

    async def list_all_entity_ids(self) -> list[str]:
        """列出 MySQL 中所有 entity_id（供对账比对孤儿实体向量）。"""
        rows = await self._fetchall("SELECT id FROM aisag_entities")
        return [str(r["id"]) for r in rows]

    async def check_duplicate(self, name: str, md5: str) -> str | None:
        """检查是否存在同名且同 MD5 的未归档文档。

        返回已存在的 source_id，若不存在则返回 None。
        """
        r = await self._fetchone(
            "SELECT id FROM aisag_sources WHERE name=%s AND md5=%s AND md5 != '' AND archived_at IS NULL",
            (name, md5),
        )
        return str(r["id"]) if r else None

    # ---------------- 写入 ----------------

    async def upsert_source(self, source_id: str, name: str, description: str = "",
                            md5: str = "", metadata: dict | None = None) -> None:
        await self._execute(
            "INSERT INTO aisag_sources (id, name, description, md5, metadata) VALUES (%s,%s,%s,%s,%s) "
            "AS new_src ON DUPLICATE KEY UPDATE name=new_src.name, description=new_src.description, "
            "md5=new_src.md5, metadata=new_src.metadata",
            (source_id, name, description, md5, json.dumps(metadata or {}, ensure_ascii=False)),
        )

    async def insert_document(self, doc_id: str, source_id: str, title: str,
                              content: str, status: str = "COMPLETED",
                              metadata: dict | None = None) -> None:
        await self._execute(
            "INSERT INTO aisag_documents (id, source_id, title, content, status, metadata) "
            "VALUES (%s,%s,%s,%s,%s,%s)",
            (doc_id, source_id, title, content, status,
             json.dumps(metadata or {}, ensure_ascii=False)),
        )

    async def insert_chunk(self, chunk: Chunk) -> None:
        await self._execute(
            "INSERT INTO aisag_chunks (id, source_id, document_id, rank_index, heading, content, metadata) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s)",
            (chunk.id, chunk.source_id, chunk.document_id, chunk.rank_index,
             chunk.heading, chunk.content,
             json.dumps(chunk.metadata or {}, ensure_ascii=False)),
        )

    async def insert_event(self, event: Event) -> None:
        await self._execute(
            "INSERT INTO aisag_events (id, source_id, document_id, chunk_id, rank_index, "
            "title, summary, content, metadata) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (event.id, event.source_id, event.document_id, event.chunk_id, event.rank_index,
             event.title, event.summary, event.content,
             json.dumps({}, ensure_ascii=False)),
        )

    async def upsert_entity(self, entity: ExtractedEntity) -> str:
        eid = str(uuid.uuid4())
        norm = self._normalize(entity.name)
        async with self.transaction() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    "INSERT INTO aisag_entities (id, entity_type, name, normalized_name, description) "
                    "VALUES (%s,%s,%s,%s,%s) "
                    "AS new_ent ON DUPLICATE KEY UPDATE "
                    "name=new_ent.name",
                    (eid, entity.type, entity.name, norm, entity.description),
                )
                await cur.execute(
                    "SELECT id FROM aisag_entities WHERE entity_type=%s AND normalized_name=%s",
                    (entity.type, norm),
                )
                row = await cur.fetchone()
                return str(row["id"]) if row else eid

    async def link_event_entity(self, event_id: str, entity_id: str,
                                weight: float = 1.0, description: str = "") -> None:
        await self._execute(
            "INSERT IGNORE INTO aisag_event_entities (id, event_id, entity_id, weight, description) "
            "VALUES (%s,%s,%s,%s,%s)",
            (str(uuid.uuid4()), event_id, entity_id, weight, description),
        )

    # ---------------- 批量事务写入 ----------------

    async def persist_source(self, source_id: str, source_name: str, file_type: str,
                             document_id: str, doc_title: str, doc_content: str,
                             chunks: list[Chunk], events: list, *,
                             md5: str = "") -> tuple[list, dict[tuple[str, str], str]]:
        # 长度校验：chunks 和 events 必须严格 1:1（P0 修复）
        if len(chunks) != len(events):
            raise ValueError(
                f"chunks 和 events 数量不一致：chunks={len(chunks)} events={len(events)}，"
                f"无法保证 1:1 对应关系")
        event_records: list = []
        seen_entities: dict[tuple[str, str], str] = {}

        async with self.transaction() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    "INSERT INTO aisag_sources (id, name, description, md5, metadata) VALUES (%s,%s,%s,%s,%s)",
                    (source_id, source_name, f"file_type={file_type}", md5,
                     json.dumps({"vector_synced": False}, ensure_ascii=False)),
                )
                await cur.execute(
                    "INSERT INTO aisag_documents (id, source_id, title, content, status, metadata) "
                    "VALUES (%s,%s,%s,%s,%s,%s)",
                    (document_id, source_id, doc_title, doc_content, "COMPLETED",
                     json.dumps({}, ensure_ascii=False)),
                )
                # 批量插入 chunks（P1 性能优化）
                chunk_rows = [
                    (c.id, c.source_id, c.document_id, c.rank_index,
                     c.heading, c.content,
                     json.dumps(c.metadata or {}, ensure_ascii=False))
                    for c in chunks
                ]
                await cur.executemany(
                    "INSERT INTO aisag_chunks (id, source_id, document_id, rank_index, heading, content, metadata) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    chunk_rows,
                )
                # 构造 event 记录并批量插入
                for idx, (chunk, ev) in enumerate(zip(chunks, events)):
                    event = self.to_event_record(
                        ev, source_id=source_id, document_id=document_id,
                        chunk_id=chunk.id, rank_index=idx,
                    )
                    event_records.append(event)
                event_rows = [
                    (e.id, e.source_id, e.document_id, e.chunk_id, e.rank_index,
                     e.title, e.summary, e.content, json.dumps({}, ensure_ascii=False))
                    for e in event_records
                ]
                await cur.executemany(
                    "INSERT INTO aisag_events (id, source_id, document_id, chunk_id, rank_index, "
                    "title, summary, content, metadata) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    event_rows,
                )

                # 实体去重 + 关联写入（需逐条查询去重，无法批量）
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
                            await cur.execute(
                                "INSERT INTO aisag_entities (id, entity_type, name, normalized_name, description) "
                                "VALUES (%s,%s,%s,%s,%s) "
                                "AS new_ent ON DUPLICATE KEY UPDATE "
                                "name=new_ent.name",
                                (new_eid, e.type, e.name, norm, e.description),
                            )
                            await cur.execute(
                                "SELECT id FROM aisag_entities WHERE entity_type=%s AND normalized_name=%s",
                                (e.type, norm),
                            )
                            row = await cur.fetchone()
                            actual_id = str(row["id"]) if row else new_eid
                            if actual_id == new_eid:
                                new_entity_count += 1
                            else:
                                reused_entity_count += 1
                            eid = actual_id
                            seen_entities[key] = eid
                        await cur.execute(
                            "INSERT IGNORE INTO aisag_event_entities (id, event_id, entity_id, weight, description) "
                            "VALUES (%s,%s,%s,%s,%s)",
                            (str(uuid.uuid4()), event.id, eid, e.weight, e.role),
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
        ph_e = ",".join(["%s"] * len(entity_ids))
        params: list = list(entity_ids)
        if source_ids:
            # JOIN 替代 EXISTS：让 MySQL 优化器一次性规划，避免每行 ee 触发子查询。
            # ee.entity_id IN 走 idx_aisag_ee_entity，ev.source_id IN 走 idx_aisag_event_source，
            # JOIN 走 ev.id 主键，DISTINCT 走 uq_aisag_ee(event_id, entity_id)。
            ph_s = ",".join(["%s"] * len(source_ids))
            params.extend(source_ids)
            sql = (f"SELECT DISTINCT ee.event_id FROM aisag_event_entities ee "
                   f"INNER JOIN aisag_events ev ON ev.id = ee.event_id "
                   f"WHERE ee.entity_id IN ({ph_e}) AND ev.source_id IN ({ph_s})")
        else:
            sql = (f"SELECT DISTINCT ee.event_id FROM aisag_event_entities ee "
                   f"WHERE ee.entity_id IN ({ph_e})")
        rows = await self._fetchall(sql, params)
        event_ids = [r["event_id"] for r in rows]
        if exclude:
            # Python 端过滤 tracked_events（不进 SQL，避免 NOT IN 长度膨胀）。
            # exclude 直接迭代，无需 list() 拷贝。
            exclude_set = set(exclude) if not isinstance(exclude, set) else exclude
            event_ids = [eid for eid in event_ids if eid not in exclude_set]
        return event_ids

    async def search_entities_by_name(self, names: list[str], source_ids: list[str] | None) -> list[Entity]:
        if not names:
            return []
        # 入库时实体名经 _normalize 处理（去空格+小写），查询前需一致化
        normalized = [self._normalize(n) for n in names]
        ph = ",".join(["%s"] * len(normalized))
        if source_ids:
            ph_s = ",".join(["%s"] * len(source_ids))
            params = list(normalized) + list(source_ids)
            sql = (f"SELECT en.id, en.entity_type, en.name, en.normalized_name, en.description "
                   f"FROM aisag_entities en "
                   f"WHERE en.normalized_name IN ({ph}) "
                   f"AND EXISTS ("
                   f"SELECT 1 FROM aisag_event_entities ee "
                   f"JOIN aisag_events ev ON ev.id = ee.event_id "
                   f"WHERE ee.entity_id = en.id AND ev.source_id IN ({ph_s})"
                   f")")
            rows = await self._fetchall(sql, params)
            return [self._row_to_entity(r) for r in rows]
        else:
            sql = (f"SELECT id, entity_type, name, normalized_name, description "
                   f"FROM aisag_entities en "
                   f"WHERE en.normalized_name IN ({ph}) "
                   f"AND EXISTS (SELECT 1 FROM aisag_event_entities ee WHERE ee.entity_id = en.id)")
            rows = await self._fetchall(sql, list(normalized))
            return [self._row_to_entity(r) for r in rows]

    async def get_entity_degrees(self, entity_ids: list[str]) -> dict[str, int]:
        """批量查实体被多少事件引用（度数），用于 BFS 边界实体的 IDF 评分。
        度数高 = 高频枢纽实体（如"众邦银行"），扩展价值低。"""
        if not entity_ids:
            return {}
        ph = ",".join(["%s"] * len(entity_ids))
        sql = (f"SELECT entity_id, COUNT(*) AS degree "
               f"FROM aisag_event_entities "
               f"WHERE entity_id IN ({ph}) "
               f"GROUP BY entity_id")
        rows = await self._fetchall(sql, list(entity_ids))
        return {r["entity_id"]: int(r["degree"]) for r in rows}

    async def filter_entity_ids_by_sources(self, entity_ids: list[str],
                                           source_ids: list[str] | None) -> list[str]:
        if not entity_ids:
            return []
        if not source_ids:
            return list(entity_ids)
        ph = ",".join(["%s"] * len(entity_ids))
        ph_s = ",".join(["%s"] * len(source_ids))
        params = list(entity_ids) + list(source_ids)
        sql = (f"SELECT DISTINCT ee.entity_id FROM aisag_event_entities ee "
               f"WHERE ee.entity_id IN ({ph}) "
               f"AND EXISTS (SELECT 1 FROM aisag_events e "
               f"            WHERE e.id = ee.event_id AND e.source_id IN ({ph_s}))")
        rows = await self._fetchall(sql, params)
        return [r["entity_id"] for r in rows]

    async def get_events_by_ids(self, event_ids: list[str],
                                source_ids: list[str] | None = None) -> list[Event]:
        if not event_ids:
            return []
        ph = ",".join(["%s"] * len(event_ids))
        where = [f"id IN ({ph})"]
        params: list = list(event_ids)
        if source_ids:
            ph_s = ",".join(["%s"] * len(source_ids))
            where.append(f"source_id IN ({ph_s})")
            params.extend(source_ids)
        sql = (f"SELECT id, source_id, document_id, chunk_id, rank_index, title, summary, content "
               f"FROM aisag_events WHERE " + " AND ".join(where))
        rows = await self._fetchall(sql, params)
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
        ph = ",".join(["%s"] * len(event_ids))
        rows = await self._fetchall(
            f"SELECT event_id, entity_id, description, weight "
            f"FROM aisag_event_entities WHERE event_id IN ({ph})",
            list(event_ids),
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
        ph = ",".join(["%s"] * len(entity_ids))
        rows = await self._fetchall(
            f"SELECT id, name FROM aisag_entities WHERE id IN ({ph})",
            list(entity_ids),
        )
        return {str(r["id"]): str(r["name"]) for r in rows}

    async def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        ph = ",".join(["%s"] * len(chunk_ids))
        sql = (f"SELECT id, source_id, document_id, rank_index, heading, content, metadata "
               f"FROM aisag_chunks WHERE id IN ({ph})")
        rows = await self._fetchall(sql, list(chunk_ids))
        row_map = {r["id"]: r for r in rows}
        return [self._row_to_chunk(row_map[cid]) for cid in chunk_ids if cid in row_map]

    async def get_chunk_ids_by_event_ids(self, event_ids: list[str]) -> dict[str, str]:
        if not event_ids:
            return {}
        ph = ",".join(["%s"] * len(event_ids))
        sql = f"SELECT id, chunk_id FROM aisag_events WHERE id IN ({ph})"
        rows = await self._fetchall(sql, list(event_ids))
        return {r["id"]: r["chunk_id"] for r in rows}

    async def delete_by_source(self, source_id: str) -> tuple[int, list[str]]:
        async with self.transaction() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                await cur.execute(
                    "SELECT DISTINCT ee.entity_id FROM aisag_event_entities ee "
                    "JOIN aisag_events e ON e.id = ee.event_id "
                    "WHERE e.source_id=%s",
                    (source_id,),
                )
                affected = await cur.fetchall()
                affected_entity_ids = [str(r["entity_id"]) for r in affected]

                await cur.execute(
                    "DELETE FROM aisag_event_entities WHERE event_id IN "
                    "(SELECT id FROM aisag_events WHERE source_id=%s)",
                    (source_id,),
                )
                await cur.execute(
                    "DELETE FROM aisag_events WHERE source_id=%s",
                    (source_id,),
                )
                n = cur.rowcount
                await cur.execute("DELETE FROM aisag_chunks WHERE source_id=%s", (source_id,))
                await cur.execute("DELETE FROM aisag_documents WHERE source_id=%s", (source_id,))
                await cur.execute("DELETE FROM aisag_sources WHERE id=%s", (source_id,))

                orphan_ids: list[str] = []
                if affected_entity_ids:
                    placeholders = ",".join(["%s"] * len(affected_entity_ids))
                    await cur.execute(
                        f"SELECT e.id, e.entity_type, e.name FROM aisag_entities e "
                        f"WHERE e.id IN ({placeholders}) "
                        f"AND NOT EXISTS ("
                        f"  SELECT 1 FROM aisag_event_entities ee WHERE ee.entity_id = e.id"
                        f")",
                        affected_entity_ids,
                    )
                    orphans = await cur.fetchall()
                    orphan_ids = [str(r["id"]) for r in orphans]
                    if orphans:
                        ph = ",".join(["%s"] * len(orphan_ids))
                        await cur.execute(f"DELETE FROM aisag_entities WHERE id IN ({ph})", orphan_ids)

                return n, orphan_ids

    async def hard_delete_soft_deleted_events(self) -> tuple[list[str], list[str]]:
        """硬删除所有 deleted_at IS NOT NULL 的软删除事件。

        删除链路：event_entities → events → 检测孤儿 entities → 删除孤儿 entities。
        返回 (deleted_event_ids, orphan_entity_ids)，供上层清理向量库。
        """
        async with self.transaction() as conn:
            async with conn.cursor(aiomysql.DictCursor) as cur:
                # 1. 查所有软删除的事件
                await cur.execute(
                    "SELECT id FROM aisag_events WHERE deleted_at IS NOT NULL"
                )
                rows = await cur.fetchall()
                event_ids = [str(r["id"]) for r in rows]
                if not event_ids:
                    return [], []

                log.info("硬删除扫描：发现 {} 条软删除事件，开始物理清理", len(event_ids))

                # 2. 查受影响 entity_id（用于后续孤儿检测）
                ph_ev = ",".join(["%s"] * len(event_ids))
                await cur.execute(
                    f"SELECT DISTINCT ee.entity_id FROM aisag_event_entities ee "
                    f"WHERE ee.event_id IN ({ph_ev})",
                    event_ids,
                )
                affected = await cur.fetchall()
                affected_entity_ids = [str(r["entity_id"]) for r in affected]

                # 3. 删除 event_entities 关联
                await cur.execute(
                    f"DELETE FROM aisag_event_entities WHERE event_id IN ({ph_ev})",
                    event_ids,
                )

                # 4. 硬删除 events
                await cur.execute(
                    f"DELETE FROM aisag_events WHERE id IN ({ph_ev})",
                    event_ids,
                )
                n = cur.rowcount

                # 5. 检测孤儿 entities（无任何 event_entities 引用的）
                orphan_ids: list[str] = []
                if affected_entity_ids:
                    ph_aff = ",".join(["%s"] * len(affected_entity_ids))
                    await cur.execute(
                        f"SELECT e.id FROM aisag_entities e WHERE e.id IN ({ph_aff}) "
                        f"AND NOT EXISTS ("
                        f"  SELECT 1 FROM aisag_event_entities ee WHERE ee.entity_id = e.id"
                        f")",
                        affected_entity_ids,
                    )
                    orphans = await cur.fetchall()
                    orphan_ids = [str(r["id"]) for r in orphans]
                    if orphans:
                        ph_o = ",".join(["%s"] * len(orphan_ids))
                        await cur.execute(
                            f"DELETE FROM aisag_entities WHERE id IN ({ph_o})",
                            orphan_ids,
                        )

                log.info("硬删除完成 events={} 孤儿entities={}", n, len(orphan_ids))
                return event_ids, orphan_ids

    # ---------------- 管理类查询 ----------------

    async def list_sources(self, *, include_archived: bool = False, keyword: str | None = None,
                           limit: int = 100, offset: int = 0) -> list[dict]:
        where = [] if include_archived else ["archived_at IS NULL"]
        params: list = []
        if keyword:
            where.append("name LIKE %s")
            params.append(f"%{keyword}%")
        where_clause = (" WHERE " + " AND ".join(where)) if where else ""
        sql = (f"SELECT id, name, description, md5, created_at, updated_at, archived_at "
               f"FROM aisag_sources{where_clause} ORDER BY created_at DESC LIMIT %s OFFSET %s")
        params.extend([limit, offset])
        rows = await self._fetchall(sql, params)
        return [self._row_to_source(r) for r in rows]

    async def get_source(self, source_id: str) -> dict | None:
        r = await self._fetchone(
            "SELECT id, name, description, md5, created_at, updated_at, archived_at "
            "FROM aisag_sources WHERE id=%s", (source_id,))
        return self._row_to_source(r) if r else None

    async def get_source_names_by_ids(self, source_ids: list[str]) -> dict[str, str]:
        """批量查 source_id → name 映射，供 API 返回溯源信息。"""
        if not source_ids:
            return {}
        ph = ",".join(["%s"] * len(source_ids))
        rows = await self._fetchall(
            f"SELECT id, name FROM aisag_sources WHERE id IN ({ph})",
            list(source_ids),
        )
        return {str(r["id"]): str(r["name"]) for r in rows}

    async def get_document_by_source(self, source_id: str) -> dict | None:
        r = await self._fetchone(
            "SELECT id, source_id, title, content, status, created_at "
            "FROM aisag_documents WHERE source_id=%s ORDER BY created_at DESC LIMIT 1",
            (source_id,))
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
        if name is not None:
            sets.append("name=%s")
            params.append(name)
        if description is not None:
            sets.append("description=%s")
            params.append(description)
        if not sets:
            return False
        params.append(source_id)
        return await self._execute(
            f"UPDATE aisag_sources SET {', '.join(sets)} WHERE id=%s", params) > 0

    async def count_chunks_by_source(self, source_id: str) -> int:
        r = await self._fetchone(
            "SELECT COUNT(*) AS c FROM aisag_chunks WHERE source_id=%s", (source_id,))
        return int(r["c"]) if r else 0

    async def count_events_by_source(self, source_id: str) -> int:
        r = await self._fetchone(
            "SELECT COUNT(*) AS c FROM aisag_events WHERE source_id=%s",
            (source_id,))
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
        sql = (
            "SELECT d.source_id, d.title, d.content, s.name AS source_name, d.created_at "
            "FROM aisag_documents d "
            "JOIN aisag_sources s ON s.id = d.source_id "
            "WHERE d.content LIKE %s AND s.archived_at IS NULL"
        )
        params: list = [like]
        if source_ids:
            ph = ",".join(["%s"] * len(source_ids))
            sql += f" AND d.source_id IN ({ph})"
            params.extend(source_ids)
        sql += " ORDER BY d.created_at DESC LIMIT %s"
        params.append(limit)
        rows = await self._fetchall(sql, params)
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
        meta = r.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        return Chunk(
            id=str(r["id"]), document_id=r["document_id"], source_id=r["source_id"],
            rank_index=r["rank_index"], heading=r.get("heading") or "Introduction",
            content=r["content"], metadata=meta or {},
        )

    def to_event_record(self, event: ExtractedEvent, *, source_id: str, document_id: str,
                        chunk_id: str, rank_index: int) -> Event:
        return Event(
            id=str(uuid.uuid4()), source_id=source_id, document_id=document_id,
            chunk_id=chunk_id, rank_index=rank_index, title=event.title,
            summary=event.summary, content=event.content,
            entity_ids=[], score=0.0,
        )