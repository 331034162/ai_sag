"""文档切分抽象基类：定义统一接口，便于切换不同切分策略。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..base import Chunk, LoadedDocument


class BaseSplitter(ABC):
    """文档切分器抽象。实现 split 即可，返回 Chunk 列表供 extractor 抽取事件。"""

    @abstractmethod
    def split(self, doc: LoadedDocument, source_id: str, document_id: str) -> list[Chunk]:
        ...