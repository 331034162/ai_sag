"""Embedding 模块：提供统一抽象与工厂，支持 bge / qwen3 后端切换。

用法：
    from ai_sag.embeddings import create_embedder
    embedder = create_embedder(cfg)
    vec = embedder.embed_text("你好")
"""
from __future__ import annotations

from .base import BaseEmbedder
from .bge import BgeEmbedder
from .factory import create_embedder
from .qwen3 import Qwen3Embedder

__all__ = ["BaseEmbedder", "BgeEmbedder", "Qwen3Embedder", "create_embedder"]