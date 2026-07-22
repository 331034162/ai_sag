"""异步问答引擎：检索 + LLM 答案生成。对外暴露 search / ask / chat / chat_stream 接口。

全链路异步：MySQL 用 aiomysql，LLM 用 LlamaIndex 原生异步方法，
Embedder/VectorStore 是 CPU 密集型同步组件用 asyncio.to_thread 包装。
"""
from __future__ import annotations

from collections.abc import AsyncGenerator

from llama_index.core.llms import ChatMessage, ChatResponse, LLM, MessageRole

from ..base import (
    CHAT_SYSTEM_PROMPT,
    Config,
    QA_EMPTY_ANSWER,
    QA_SECTION_FORMAT,
    SearchResult,
)
from ..base.logger import get_logger
from ..embeddings import create_embedder
from ..llm import LlmFactory
from ..storage import create_db_store, MysqlStore, PgStore
from ..vector_store import BaseVectorStore, create_vector_store
from .sag_retriever import SagRetriever

log = get_logger()

MAX_HISTORY_ROUNDS = 5


class QAEngine:
    def __init__(self, cfg: Config | None = None, *,
                 db: MysqlStore | PgStore | None = None,
                 vectors: BaseVectorStore | None = None) -> None:
        self.cfg = cfg or Config()
        self.embedder = create_embedder(self.cfg)
        # LlmFactory 支持按场景返回不同 LLM 实例（答案生成用 ANSWER，其他场景独立配置）
        self._llm_factory = LlmFactory(self.cfg)
        # self.llm 作为检索器兜底 LLM：用 ANSWER 场景的实例（检索器内部会用 _llm_for 按场景选取）
        self.llm: LLM = self._llm_factory.get("ANSWER")
        self._owns_db = db is None
        self.db = db or create_db_store(
            self.cfg,
            faiss_map_enabled=(self.cfg.vector_store.backend.lower() == "faiss"),
        )
        self.vectors = vectors or create_vector_store(self.cfg, db_store=self.db)
        self.retriever = SagRetriever(self.cfg, self.db, self.vectors, self.embedder, self.llm,
                                      llm_factory=self._llm_factory)

    async def search(self, query: str, source_ids: list[str] | None = None,
                     document_ids: list[str] | None = None, *,
                     fusion: str | None = None,
                     history: list[dict] | None = None) -> SearchResult:
        return await self.retriever.search(query, source_ids, document_ids, fusion=fusion, history=history)

    async def ask(self, query: str, source_ids: list[str] | None = None,
                  document_ids: list[str] | None = None, *,
                  fusion: str | None = None) -> tuple[str, SearchResult]:
        result = await self.search(query, source_ids, document_ids, fusion=fusion)
        answer = await self._generate(query, result)
        return answer, result

    async def chat(self, query: str, history: list[dict] | None = None,
                   source_ids: list[str] | None = None,
                   document_ids: list[str] | None = None, *,
                   fusion: str | None = None) -> tuple[str, SearchResult]:
        result = await self.search(query, source_ids, document_ids, fusion=fusion, history=history)
        answer = await self._chat_generate(query, result, history or [])
        return answer, result

    async def chat_stream(self, query: str, history: list[dict] | None = None,
                          source_ids: list[str] | None = None,
                          document_ids: list[str] | None = None, *,
                          fusion: str | None = None) -> tuple[SearchResult, AsyncGenerator[str, None]]:
        result = await self.search(query, source_ids, document_ids, fusion=fusion, history=history)
        stream = self._stream_chat_generate(query, result, history or [])
        return result, stream

    async def _generate(self, query: str, result: SearchResult) -> str:
        """单轮问答：统一走 _build_messages（history=[] 等同单轮）。"""
        if not result.sections:
            return QA_EMPTY_ANSWER
        messages = self._build_messages(query, result, [])
        try:
            # 答案生成使用 ANSWER 场景（可独立配置思考模式 / temperature）
            llm_answer = self._llm_factory.get("ANSWER")
            resp: ChatResponse = await llm_answer.achat(messages)
            return str(resp).strip()
        except Exception as e:
            log.error("单轮问答生成失败 query={!r} err={}", query, e)
            return "抱歉，生成答案时出现错误，请稍后重试。"

    async def _chat_generate(self, query: str, result: SearchResult,
                             history: list[dict]) -> str:
        if not result.sections:
            return QA_EMPTY_ANSWER
        messages = self._build_messages(query, result, history)
        try:
            llm_answer = self._llm_factory.get("ANSWER")
            resp: ChatResponse = await llm_answer.achat(messages)
            return str(resp).strip()
        except Exception as e:
            log.error("多轮对话生成失败 query={!r} err={}", query, e)
            return "抱歉，对话生成时出现错误，请稍后重试。"

    def _build_messages(self, query: str, result: SearchResult,
                        history: list[dict]) -> list[ChatMessage]:
        """组装多轮对话消息列表：system(含资料) + 历史 + 当前 query。

        片段编号采用全局递增（P1 修复：避免多轮对话中片段编号引用错乱）。
        """
        context = self._build_context(result) if result.sections else "（未检索到相关资料）"
        system_content = CHAT_SYSTEM_PROMPT.format(context=context)

        messages: list[ChatMessage] = [ChatMessage(role=MessageRole.SYSTEM, content=system_content)]

        truncated = self._truncate_history(history, MAX_HISTORY_ROUNDS)
        role_map = {"user": MessageRole.USER, "assistant": MessageRole.ASSISTANT}
        for h in truncated:
            role = role_map.get(h.get("role", "user"), MessageRole.USER)
            content = h.get("content", "")
            if content:
                messages.append(ChatMessage(role=role, content=content))

        messages.append(ChatMessage(role=MessageRole.USER, content=query))
        return messages

    async def _stream_chat_generate(self, query: str, result: SearchResult,
                                    history: list[dict]) -> AsyncGenerator[str, None]:
        """流式生成：用 llm.astream_chat 原生异步流式，async for 直接迭代增量 token。"""
        messages = self._build_messages(query, result, history)
        try:
            llm_answer = self._llm_factory.get("ANSWER")
            async for resp in await llm_answer.astream_chat(messages):
                delta = getattr(resp, "delta", None)
                if delta:
                    yield delta
        except Exception as e:
            log.error("流式对话生成失败 query={!r} err={}", query, e)
            yield "\n[生成中断，请稍后重试]"

    @staticmethod
    def _build_context(result: SearchResult) -> str:
        """构建上下文：每个片段带编号和相关度分数，LLM 引用片段编号时可直接对应。"""
        return "\n\n".join(
            QA_SECTION_FORMAT.format(
                i=i + 1,
                source_name=s.source_name or s.source_id[:12],
                heading=s.heading,
                score=s.score,
                content=s.content,
            )
            for i, s in enumerate(result.sections)
        )

    @staticmethod
    def _truncate_history(history: list[dict], max_rounds: int) -> list[dict]:
        valid = [h for h in history
                 if h.get("role") in ("user", "assistant") and h.get("content", "").strip()]
        max_msgs = max_rounds * 2
        if len(valid) > max_msgs:
            valid = valid[-max_msgs:]
        return valid

    async def close(self) -> None:
        if self._owns_db:
            await self.db.close()