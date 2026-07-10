"""事件抽取器：用 LlamaIndex LLM 把 chunk 抽成结构化事件 + 实体。

忠实移植 SAG 的抽取原则：
- 单事件融合：每个 chunk 融合为一个顶层事件，各 chunk 独立抽取互不污染
- 事件+实体并行输出：一次 LLM 调用同时产出事件与实体，非串行级联
- 11 类实体定义
"""
from __future__ import annotations

import time
from typing import Any

from llama_index.core.llms import LLM
from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from ..base import (
    EXTRACT_TEMPLATE,
    Chunk,
    ExtractedEntity,
    ExtractedEvent,
    ExtractionError,
    extract_system_prompt,
)
from ..base.logger import get_logger

log = get_logger()


class _Entity(BaseModel):
    type: str
    name: str
    description: str = ""   # 实体固有属性
    role: str = ""           # 在该事件中的角色
    weight: float = 1.0      # 实体在当前事件中的关联度，0.1-1.0


class _Event(BaseModel):
    title: str
    summary: str
    content: str
    entities: list[_Entity] = Field(default_factory=list)


_EXTRACT_TEMPLATE = PromptTemplate(EXTRACT_TEMPLATE)


class EventExtractor:
    def __init__(self, llm: LLM, *, max_retries: int = 2,
                 summary_max_chars: int = 500) -> None:
        self.llm = llm
        self._max_retries = max_retries
        self._summary_max_chars = summary_max_chars

    def extract(self, chunk: Chunk, doc_title: str,
                previous_context: str = "") -> ExtractedEvent:
        """抽取 chunk 中的结构化事件。

        previous_context 为前一个 chunk 的事件摘要，用于消解代词和省略主语。
        瞬时故障（rate limit / 网络抖动 / JSON 解析偶发失败）自动重试 self._max_retries 次。
        重试耗尽后抛 ExtractionError 终止入库，不再静默返回 entities=[] 的 fallback。
        """
        last_error = ""
        for attempt in range(self._max_retries + 1):
            try:
                result = self.llm.structured_predict(
                    _Event,
                    _EXTRACT_TEMPLATE,
                    system_prompt=extract_system_prompt(self._summary_max_chars),
                    doc_title=doc_title,
                    heading=chunk.heading,
                    previous_context=previous_context or "（无，这是文档的开头）",
                    content=chunk.content,
                )
                ev = self._to_model(result)
                log.info("抽取成功 chunk={} 实体数={} 重试={}", chunk.id, len(ev.entities), attempt)
                return ev
            except Exception as e:
                last_error = str(e)[:200]
                if attempt < self._max_retries:
                    wait = 2 ** attempt  # 指数退避：1s → 2s
                    log.warning("LLM 抽取失败，{:.0f}s 后重试 chunk={} attempt={}/{} err={}",
                                wait, chunk.id, attempt + 1, self._max_retries, last_error)
                    time.sleep(wait)
        raise ExtractionError(chunk.id, self._max_retries, last_error)

    def extract_batch(self, chunks: list[Chunk], doc_title: str, *,
                      parallel: bool = False, max_workers: int = 4) -> list[ExtractedEvent]:
        """批量抽取。顺序模式逐 chunk 传递前文摘要用于代词消解；
        并行模式下无法传递跨 chunk 上下文（各 chunk 独立并发）。

        parallel=True 时并发抽取，max_workers 控制并发数。"""
        if not parallel:
            results: list[ExtractedEvent] = []
            prev = ""
            for chunk in chunks:
                ev = self.extract(chunk, doc_title, previous_context=prev)
                results.append(ev)
                prev = ev.summary or ""
            return results
        # 并行抽取：任一 chunk 的重试耗尽均向上抛 ExtractionError，不再静默 fallback。
        from concurrent.futures import ThreadPoolExecutor, as_completed
        indexed: list[ExtractedEvent] = [None] * len(chunks)  # type: ignore[list-item]
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(self.extract, chunk, doc_title): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    indexed[idx] = future.result()
                except ExtractionError as e:
                    errors.append(str(e))
                except Exception as e:
                    errors.append(f"并行抽取异常 chunk_idx={idx}: {e}")
        if errors:
            raise ExtractionError(
                chunks[0].id if chunks else "unknown", self._max_retries,
                "; ".join(errors[:3]))
        return indexed

    @staticmethod
    def _to_model(ev: _Event) -> ExtractedEvent:
        return ExtractedEvent(
            title=ev.title, summary=ev.summary, content=ev.content,
            entities=[ExtractedEntity(type=e.type, name=e.name,
                                     description=e.description, role=e.role,
                                     weight=e.weight) for e in ev.entities],
        )