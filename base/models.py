"""数据模型：贯穿加载→清洗→切分→抽取→入库→检索各阶段。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ExtractionError(Exception):
    """LLM 事件抽取失败（重试耗尽后仍失败）。

    对应场景：
    - API 持续不可用（rate limit / 服务端 503 / 网络中断）
    - structured_predict JSON 解析反复失败
    - chunk 内容触发安全过滤器
    """
    def __init__(self, chunk_id: str, retries: int, last_error: str) -> None:
        super().__init__(f"抽取失败 chunk={chunk_id}（重试{retries}次后仍失败）: {last_error}")
        self.chunk_id = chunk_id
        self.retries = retries
        self.last_error = last_error


@dataclass
class LoadedDocument:
    """loader 产物：原始文档解析后的纯文本。"""
    title: str
    content: str
    source_path: str = ""
    file_type: str = ""


@dataclass
class Chunk:
    """splitter 产物：文档切片，对应 LlamaIndex 的 Node。"""
    id: str
    document_id: str
    source_id: str
    rank_index: int
    heading: str
    content: str


@dataclass
class ExtractedEntity:
    """extractor 产物：从事件中抽取的实体。"""
    type: str
    name: str
    description: str = ""   # 实体固有属性（如"互联网银行"），不随事件变化
    role: str = ""          # 实体在该事件中的角色（如"贷款方"/"担保方"）
    weight: float = 1.0     # 实体在当前事件中的重要性/关联度，0.1-1.0，默认1.0


@dataclass
class ExtractedEvent:
    """extractor 产物：从 chunk 中抽取的融合事件。"""
    title: str
    summary: str
    content: str
    entities: list[ExtractedEntity] = field(default_factory=list)


@dataclass
class Entity:
    """storage 产物：已入库的实体（跨文档全局共享）。"""
    id: str
    type: str
    name: str
    normalized_name: str
    description: str = ""
    score: float = 0.0


@dataclass
class Event:
    """storage 产物：已入库的事件。"""
    id: str
    source_id: str
    document_id: str
    chunk_id: str
    rank_index: int
    title: str
    summary: str
    content: str
    entity_ids: list[str] = field(default_factory=list)
    entity_roles: dict[str, str] = field(default_factory=dict)
    entity_weights: dict[str, float] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class RetrievedSection:
    """retrieval 产物：检索返回的切片。"""
    chunk_id: str
    source_id: str
    document_id: str
    heading: str
    content: str
    source_name: str = ""
    rank: int = 0
    score: float = 0.0


@dataclass
class SearchTrace:
    """检索审计链路：q → Uq → Ûq → ER → Ecand → Ê → Cout"""
    query: str
    query_entities: list[str] = field(default_factory=list)          # Uq：LLM 抽取的查询实体
    expanded_query_entities: list[str] = field(default_factory=list) # Ûq：向量扩展后的实体 ID（0.9 阈值）
    seed_event_ids: list[str] = field(default_factory=list)          # ER：种子事件 ID
    expanded_event_ids: list[str] = field(default_factory=list)      # Ecand：BFS 扩展后的事件池
    rerank_candidate_ids: list[str] = field(default_factory=list)    # Ê：送入 LLM 精排的候选事件
    reranked_ids: list[str] = field(default_factory=list)            # E*：LLM 精排输出 (top 5)
    fallback: str | None = None


@dataclass
class SearchResult:
    sections: list[RetrievedSection]
    trace: SearchTrace | None = None