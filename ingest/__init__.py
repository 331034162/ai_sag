"""ingest 子包：入库编排，封装 loader → cleaner → splitter → extractor → storage 完整流程。"""
from __future__ import annotations

from .pipeline import IngestPipeline

__all__ = ["IngestPipeline"]