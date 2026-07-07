"""向量库模块：提供统一抽象与工厂，支持 ChromaDB（可扩展 Milvus/FAISS）。

用法：
    from ai_sag.vector_store import create_vector_store
    vs = create_vector_store(cfg)
    vs.add_chunks([(cid, text, emb), ...])
    hits = vs.query_chunks(query_vec, top_k=10)
"""
from __future__ import annotations

from .base import BaseVectorStore, Collection
from .chroma_store import ChromaVectorStoreBackend
from .factory import create_vector_store

__all__ = ["BaseVectorStore", "Collection", "ChromaVectorStoreBackend", "create_vector_store"]