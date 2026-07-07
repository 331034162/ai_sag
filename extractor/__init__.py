"""事件抽取：基于 LlamaIndex LLM + Pydantic，从 chunk 抽取事件与实体。"""
from __future__ import annotations

from .event_extractor import EventExtractor

__all__ = ["EventExtractor"]