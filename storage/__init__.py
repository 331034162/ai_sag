"""存储层：关系型数据库事件-实体超边（向量库已独立到 ai_sag.vector_store 包）。

支持 MySQL（aiomysql）和 PostgreSQL（asyncpg）两种后端，
通过 create_db_store() 工厂函数按配置自动选择。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .mysql_store import MysqlStore, uuid_to_int64
from .pg_store import PgStore

if TYPE_CHECKING:
    from ..base import Config

__all__ = ["MysqlStore", "PgStore", "uuid_to_int64", "create_db_store"]


def create_db_store(cfg: "Config", *, faiss_map_enabled: bool = False) -> MysqlStore | PgStore:
    """按配置创建数据库存储实例。

    Args:
        cfg: 全局配置（通过 cfg.db_backend 选择后端）
        faiss_map_enabled: 是否启用 FAISS 映射表（仅 FAISS 向量后端需要）

    Returns:
        MysqlStore 或 PgStore 实例（接口完全一致）
    """
    backend = cfg.db_backend.lower()
    if backend == "postgresql":
        return PgStore(
            host=cfg.pg.host, port=cfg.pg.port,
            user=cfg.pg.user, password=cfg.pg.password,
            database=cfg.pg.database,
            pool_size=cfg.pg.pool_size,
            max_overflow=cfg.pg.max_overflow,
            pool_timeout=cfg.pg.pool_timeout,
            pool_recycle=cfg.pg.pool_recycle,
            faiss_map_enabled=faiss_map_enabled,
        )
    # 默认 mysql
    return MysqlStore(
        host=cfg.mysql.host, port=cfg.mysql.port,
        user=cfg.mysql.user, password=cfg.mysql.password,
        database=cfg.mysql.database,
        pool_size=cfg.mysql.pool_size,
        max_overflow=cfg.mysql.max_overflow,
        pool_timeout=cfg.mysql.pool_timeout,
        pool_recycle=cfg.mysql.pool_recycle,
        faiss_map_enabled=faiss_map_enabled,
    )