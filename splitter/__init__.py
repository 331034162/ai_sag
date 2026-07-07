"""文档切分模块：提供统一抽象与工厂，支持 auto/markdown/sentence/token/code 切分。

推荐使用 auto 模式：根据文档类型自动选择最适配的策略。
用法：
    from ai_sag.splitter import create_splitter
    splitter = create_splitter(cfg)  # cfg.splitter.mode 默认 "auto"
    chunks = splitter.split(doc, source_id, document_id)
"""
from __future__ import annotations

from .auto_splitter import AutoSplitter
from .base import BaseSplitter
from .chunk_splitter import ChunkSplitter
from .factory import create_splitter

__all__ = ["BaseSplitter", "ChunkSplitter", "AutoSplitter", "create_splitter"]