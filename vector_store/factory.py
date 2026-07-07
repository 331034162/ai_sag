"""向量库工厂：按配置后端创建对应实现，便于切换 ChromaDB/Milvus/FAISS。"""
from __future__ import annotations

from ..base import Config
from .base import BaseVectorStore
from .chroma_store import ChromaVectorStoreBackend


def create_vector_store(cfg: Config) -> BaseVectorStore:
    backend = cfg.vector_store.backend.lower()
    if backend in ("chroma", "chromadb"):
        return ChromaVectorStoreBackend(cfg)
    if backend in ("milvus",):
        raise NotImplementedError("Milvus 后端尚未实现，可参照 chroma_store.py 扩展")
    if backend in ("faiss",):
        raise NotImplementedError("FAISS 后端尚未实现，可参照 chroma_store.py 扩展")
    raise ValueError(f"未知向量库后端: {backend}（支持: chroma）")