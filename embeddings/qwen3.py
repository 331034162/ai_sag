"""Qwen3 后端：基于 LlamaIndex HuggingFaceEmbedding 实现。

设计要点：
  - 复用 LlamaIndex 的 HuggingFaceEmbedding，自动适配 LlamaIndex 生态
    （SemanticSplitterNodeParser、VectorStoreIndex 等需要 BaseEmbedding 的组件可直接复用）
  - 查询指令前缀（Qwen3 官方推荐）通过 model_kwargs 或 query_instruction 注入
  - last_token pooling：Qwen3-Embedding-0.6B 原生采用 last_token（非 CLS）pooling，
    在 sentence_transformers 配置（1_Pooling/config.json）中已声明 pooling_mode_lasttoken=True，
    HuggingFaceEmbedding 底层用 sentence-transformers 加载时会自动按该配置 pooling
"""
from __future__ import annotations

import asyncio
import os

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from ..base import Config
from .base import BaseEmbedder


# 默认查询任务描述（可被 SAG_EMBEDDING_QUERY_INSTRUCTION 覆盖）
_DEFAULT_QUERY_INSTRUCTION = (
    "Instruct: Given a query, retrieve relevant passages from the knowledge base\nQuery: "
)


class Qwen3Embedder(BaseEmbedder):
    """Qwen3-Embedding-0.6B 后端（基于 LlamaIndex HuggingFaceEmbedding）。

    - pooling: last_token（由模型 sentence_transformers 配置自动决定）
    - 查询加 "Instruct: ...\\nQuery: " 前缀
    - 文档不加前缀，直接 embed
    - max_length 默认 8192，支持长文本

    暴露两个 embedder：
      - self._llama_embed：LlamaIndex BaseEmbedding 实例，供 LlamaIndex 组件复用
        （如 SemanticSplitterNodeParser）
      - self：BaseEmbedder 子类，供项目内部统一接口使用
    """

    def __init__(self, cfg: Config) -> None:
        # 无 CUDA 时降级到 CPU（HuggingFaceEmbedding 默认会用 CUDA，需显式指定）
        import torch
        device_str = cfg.embedding.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if "cuda" in (device_str or "").lower() and not torch.cuda.is_available():
            device_str = "cpu"

        # 查询指令前缀（Qwen3 官方推荐：Instruct: ... \nQuery: ）
        # HuggingFaceEmbedding 的 query_instruction 仅在 aget_query_embedding 中追加
        self._query_instruction = os.environ.get(
            "SAG_EMBEDDING_QUERY_INSTRUCTION", _DEFAULT_QUERY_INSTRUCTION
        )

        # LlamaIndex HuggingFaceEmbedding：
        #   - 内部用 sentence-transformers 加载模型，自动按 1_Pooling/config.json 做 pooling
        #   - Qwen3-Embedding-0.6B 的 pooling_mode_lasttoken=True，自动 last_token pooling
        #   - normalize 参数控制是否 L2 归一化（透传给 SentenceTransformer.encode(normalize_embeddings=...)）
        #     注意：参数名是 normalize（不是 normalize_embeddings），后者会被 pydantic 当成
        #     model_kwargs 透传给 SentenceTransformer.__init__() 导致 TypeError
        self._llama_embed: BaseEmbedding = HuggingFaceEmbedding(
            model_name=cfg.embedding.model_path,
            max_length=cfg.embedding.max_length,
            normalize=cfg.embedding.normalize,
            trust_remote_code=True,
            device=device_str,
            query_instruction=self._query_instruction,
        )

    # ---------------- 项目内部统一接口（BaseEmbedder） ----------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """入库文档：不加指令前缀，直接 embed。"""
        if not texts:
            return []
        safe = [self._safe_text(t) for t in texts]
        # HuggingFaceEmbedding._get_text_embedding 是同步方法，单条调用
        # 批量调用走 _get_text_embeddings（注意复数）效率更高
        return [list(e) for e in self._llama_embed._get_text_embeddings(safe)]

    def embed_query(self, query: str) -> list[float]:
        """同步查询：加指令前缀（HuggingFaceEmbedding 自动追加 query_instruction）。"""
        if not (query or "").strip():
            return []
        return list(self._llama_embed._get_query_embedding(self._safe_text(query)))

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        """异步批量 embed：LlamaIndex BaseEmbedding 没有公开的复数异步 API，
        用 asyncio.to_thread 包装同步的批量方法 _get_text_embeddings（复数），
        避免逐条 await 造成的串行开销（Qwen3 单条 embed 50-200ms，10 条串行就要 1-2s）。
        """
        if not texts:
            return []
        safe = [self._safe_text(t) for t in texts]
        # 走线程池执行同步批量调用，不阻塞事件循环
        embs = await asyncio.to_thread(self._llama_embed._get_text_embeddings, safe)
        return [list(e) for e in embs]

    async def aembed_query(self, query: str) -> list[float]:
        """异步查询。"""
        if not (query or "").strip():
            return []
        return list(await self._llama_embed.aget_query_embedding(self._safe_text(query)))

    def embed_text(self, text: str) -> list[float]:
        """同步单条 embed。"""
        return list(self._llama_embed._get_text_embedding(self._safe_text(text)))

    async def aembed_text(self, text: str) -> list[float]:
        """异步单条 embed。"""
        return list(await self._llama_embed.aget_text_embedding(self._safe_text(text)))