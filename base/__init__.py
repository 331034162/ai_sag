"""base 子包：基础设施层，提供配置与数据模型。

集中管理全局共享的数据结构（LoadedDocument/Chunk/Event/Entity 等）与组件配置（Config），
被 loader/cleaner/splitter/extractor/storage/vector_store/embeddings/llm/ingest/retrieval 复用。
"""
from __future__ import annotations

from .config import (
    CleanerConfig,
    Config,
    EmbeddingConfig,
    MysqlConfig,
    SearchConfig,
    SplitterConfig,
    VectorStoreConfig,
)
from .logger import (
    LogConfig,
    generate_trace_id,
    get_logger,
    get_trace_id,
    init_logger,
    reset_trace_id,
    set_trace_id,
)
from .models import (
    Chunk,
    Entity,
    Event,
    ExtractedEntity,
    ExtractedEvent,
    ExtractionError,
    LoadedDocument,
    RetrievedSection,
    SearchResult,
    SearchTrace,
)
from .prompts import (
    CHAT_SYSTEM_PROMPT,
    EXTRACT_TEMPLATE,
    QA_EMPTY_ANSWER,
    QA_SECTION_FORMAT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    QUERY_REWRITE_USER_PROMPT,
    RERANK_CANDIDATE_FORMAT,
    RERANK_PROMPT_TEMPLATE,
    RERANK_SYSTEM_PROMPT,
    SUPPORTED_GENRES,
    extract_system_prompt,
    query_extract_system_prompt,
)

__all__ = [
    "Config",
    "MysqlConfig",
    "EmbeddingConfig",
    "VectorStoreConfig",
    "SplitterConfig",
    "LogConfig",
    "SearchConfig",
    "init_logger",
    "get_logger",
    "get_trace_id",
    "set_trace_id",
    "reset_trace_id",
    "generate_trace_id",
    "LoadedDocument",
    "Chunk",
    "ExtractedEntity",
    "ExtractedEvent",
    "ExtractionError",
    "Entity",
    "Event",
    "RetrievedSection",
    "SearchTrace",
    "SearchResult",
    "extract_system_prompt",
    "query_extract_system_prompt",
    "SUPPORTED_GENRES",
    "EXTRACT_TEMPLATE",
    "QUERY_REWRITE_SYSTEM_PROMPT",
    "QUERY_REWRITE_USER_PROMPT",
    "QA_EMPTY_ANSWER",
    "QA_SECTION_FORMAT",
    "CHAT_SYSTEM_PROMPT",
    "RERANK_PROMPT_TEMPLATE",
    "RERANK_SYSTEM_PROMPT",
    "RERANK_CANDIDATE_FORMAT",
]