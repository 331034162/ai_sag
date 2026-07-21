"""向量库工厂：按配置后端创建对应实现，便于切换 ChromaDB / FAISS / Milvus / PGVector。

faiss / milvus / pgvector 后端依赖（faiss-cpu / pymilvus / pgvector / asyncpg / sqlalchemy）
按需导入，未安装时给出明确安装提示，不影响仅使用 chroma 后端的用户。

FAISS 后端依赖 MySQL 反查（faiss_id_hash → UUID），需通过 mysql_store 参数传入 MysqlStore。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ..base import Config
from .base import BaseVectorStore
from .chroma_store import ChromaVectorStoreBackend

if TYPE_CHECKING:
    from ..storage.mysql_store import MysqlStore


def create_vector_store(cfg: Config, mysql_store: "MysqlStore | None" = None) -> BaseVectorStore:
    """创建向量库后端实例。

    Args:
        cfg: 全局配置
        mysql_store: FAISS 后端必需（用于 faiss_id_hash → UUID 反查）；
                    其他后端（chroma/milvus/pgvector）可不传。
    """
    backend = cfg.vector_store.backend.lower()
    if backend in ("chroma", "chromadb"):
        return ChromaVectorStoreBackend(cfg)
    if backend in ("faiss",):
        try:
            from .faiss_store import FaissVectorStoreBackend
        except ImportError as e:
            raise ImportError(
                f"使用 faiss 后端需先安装依赖：pip install faiss-cpu "
                f"llama-index-vector-stores-faiss（原错误：{e}）"
            ) from e
        if mysql_store is None:
            raise ValueError(
                "FAISS 后端必须传入 mysql_store（用于 faiss_id_hash → UUID 反查），"
                "请调用 create_vector_store(cfg, mysql_store=db)")
        return FaissVectorStoreBackend(cfg, mysql_store=mysql_store)
    if backend in ("milvus",):
        try:
            from .milvus_store import MilvusVectorStoreBackend
        except ImportError as e:
            raise ImportError(
                f"使用 milvus 后端需先安装依赖：pip install pymilvus "
                f"llama-index-vector-stores-milvus（原错误：{e}）"
            ) from e
        return MilvusVectorStoreBackend(cfg)
    if backend in ("pgvector", "pg", "postgres", "postgresql"):
        try:
            from .pgvector_store import PGVectorStoreBackend
        except ImportError as e:
            raise ImportError(
                f"使用 pgvector 后端需先安装依赖：pip install llama-index-vector-stores-postgres "
                f"asyncpg sqlalchemy（原错误：{e}）"
            ) from e
        return PGVectorStoreBackend(cfg)
    raise ValueError(f"未知向量库后端: {backend}（支持: chroma / faiss / milvus / pgvector）")