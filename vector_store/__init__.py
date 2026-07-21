"""向量库模块：提供统一抽象与工厂，支持 ChromaDB / FAISS / Milvus / PGVector 四种后端。

用法：
    from ai_sag.vector_store import create_vector_store
    vs = create_vector_store(cfg)
    vs.add_chunks([(cid, text, emb), ...])
    hits = vs.query_chunks(query_vec, top_k=10)

后端切换：在 .env 中设置 AISAG_VECTOR_STORE_BACKEND=chroma|faiss|milvus|pgvector
  - chroma：默认，本地轻量持久化（无需外部服务，适合开发/小规模）
  - faiss：单机内存索引 + 本地持久化（检索速度极快，适合本地大规模）
  - milvus：分布式向量库（需启动 Milvus 服务，适合生产部署）
  - pgvector：基于 PostgreSQL + pgvector 扩展（与业务数据共存，支持 SQL 联查）
"""
from __future__ import annotations

from .base import BaseVectorStore, Collection
from .chroma_store import ChromaVectorStoreBackend
from .factory import create_vector_store

# faiss / milvus / pgvector 后端依赖较重（faiss-cpu / pymilvus / asyncpg / sqlalchemy），
# 未安装时不阻塞 import。create_vector_store() 在选中对应后端时才按需导入，并给出明确的安装提示。
__all__ = [
    "BaseVectorStore",
    "Collection",
    "ChromaVectorStoreBackend",
    "create_vector_store",
]