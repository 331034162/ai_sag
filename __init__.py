"""ai_sag：基于 LlamaIndex 实现的 SAG 事件-实体关联知识库。

模块划分：
- loader       文档加载（.md/.txt/.docx/.pdf → 纯文本）
- cleaner      文本清洗（去噪、规整）
- splitter     文档切分（LlamaIndex NodeParser，支持 Markdown/句子/Token/代码等）
- extractor    事件抽取（LLM 抽取事件与实体，Pydantic 结构化输出）
- embeddings   Embedding 后端（bge/qwen3，工厂可切换）
- llm          LLM 后端（openai_like/openai/ollama，工厂可切换）
- vector_store 向量库后端（chroma/可扩展 milvus/faiss，工厂可切换）
- storage      MySQL 事件-实体超边存储
- ingest       入库编排（loader→cleaner→splitter→extractor→storage）
- retrieval    文档问答检索（SAG 多跳 BFS + 粗排/重排 + 双路融合）
- api          FastAPI 接口（文档上传/下载/删除/更新/检索/问答）
"""
from __future__ import annotations

from .base import Chunk, Config, Entity, Event, LoadedDocument
from .embeddings import BaseEmbedder, create_embedder
from .ingest import IngestPipeline
from .llm import LlmFactory
from .retrieval.qa_engine import QAEngine
from .vector_store import BaseVectorStore, create_vector_store

__all__ = [
    "Config",
    "IngestPipeline",
    "QAEngine",
    "LoadedDocument",
    "Chunk",
    "Event",
    "Entity",
    "BaseEmbedder",
    "create_embedder",
    "LlmFactory",
    "BaseVectorStore",
    "create_vector_store",
]


def create_app():
    """懒加载创建 FastAPI 应用，避免 import 时连接 DB。"""
    from .api import create_app as _create_app
    return _create_app()