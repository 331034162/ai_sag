"""Embedding 抽象基类：定义统一接口，便于切换 bge/qwen3/其他后端。

同步方法 embed_texts 供入库等同步场景使用；
异步方法 aembed_texts 供异步检索流程使用，默认用 asyncio.to_thread 包装同步实现。
子类若有原生异步能力可重写 aembed_texts。

查询接口 embed_query / aembed_query 默认走文档路径；
指令微调模型（如 Qwen3）应重写这两个方法以加查询指令前缀。
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Embedding 后端抽象。所有后端实现 embed_texts / embed_text。"""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_text(self, text: str) -> list[float]:
        out = self.embed_texts([text])
        if out is None or len(out) == 0:
            return []
        return out[0]

    # ---------------- 查询接口（默认走文档路径，指令微调模型可重写）----------------

    def embed_query(self, query: str) -> list[float]:
        """同步查询 embed。默认走文档路径；指令微调模型应重写加前缀。"""
        return self.embed_text(query)

    async def aembed_query(self, query: str) -> list[float]:
        """异步查询 embed。默认走文档路径；指令微调模型应重写加前缀。"""
        return await self.aembed_text(query)

    # ---------------- 异步接口（默认用 to_thread 包装同步实现）----------------

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        """异步批量 embed。默认包装同步 embed_texts，子类可重写为原生异步。"""
        if not texts:
            return []
        return await asyncio.to_thread(self.embed_texts, texts)

    async def aembed_text(self, text: str) -> list[float]:
        """异步单个 embed。"""
        out = await self.aembed_texts([text])
        if out is None or len(out) == 0:
            return []
        return out[0]

    @staticmethod
    def _safe_text(text: str) -> str:
        text = (text or "").strip()
        return text if text else "empty"