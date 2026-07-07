"""Embedding 工厂：按配置后端创建对应实现，便于切换。"""
from __future__ import annotations

from ..base import Config
from .base import BaseEmbedder
from .bge import BgeEmbedder
from .qwen3 import Qwen3Embedder


def create_embedder(cfg: Config) -> BaseEmbedder:
    backend = cfg.embedding.backend.lower()
    if backend == "bge":
        return BgeEmbedder(cfg)
    if backend in ("qwen3", "qwen"):
        return Qwen3Embedder(cfg)
    raise ValueError(f"未知 embedding 后端: {backend}（支持: bge / qwen3）")