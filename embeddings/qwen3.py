"""Qwen3 后端：基于 transformers 手写实现（llamaindex 新版 HuggingFaceEmbedding 不支持 pooling 配置）。

参考 Qwen3-Embedding 官方推荐用法：
- pooling: last_token（取最后一个非 padding token 的向量）
- 文档（入库）：不加前缀，直接 embed
- 查询（检索）：加 "Instruct: ${task}\\nQuery: " 前缀
  默认 task = "Given a query, retrieve relevant passages from the knowledge base"
"""
from __future__ import annotations

import asyncio
import os

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from ..base import Config
from .base import BaseEmbedder


# 默认查询任务描述（可被 SAG_EMBEDDING_QUERY_INSTRUCTION 覆盖）
_DEFAULT_QUERY_INSTRUCTION = (
    "Instruct: Given a query, retrieve relevant passages from the knowledge base\nQuery: "
)


class Qwen3Embedder(BaseEmbedder):
    """Qwen3-Embedding-0.6B 等指令微调模型。

    - pooling: last_token（官方推荐）
    - 查询加 "Instruct: ...\\nQuery: " 前缀
    - 文档不加前缀，直接 embed
    - max_length 默认 8192，支持长文本
    """
    def __init__(self, cfg: Config) -> None:
        # 无 CUDA 时降级到 CPU
        device_str = cfg.embedding.device
        if device_str and "cuda" in device_str.lower() and not torch.cuda.is_available():
            device_str = "cpu"
        self._device = torch.device(device_str or "cpu")

        self._tokenizer = AutoTokenizer.from_pretrained(
            cfg.embedding.model_path, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            cfg.embedding.model_path, trust_remote_code=True
        ).to(self._device).eval()

        self._max_length = cfg.embedding.max_length
        self._normalize = cfg.embedding.normalize
        self._batch_size = cfg.embedding.batch_size if cfg.embedding.batch_size > 0 else 32
        self._query_instruction = os.environ.get(
            "SAG_EMBEDDING_QUERY_INSTRUCTION", _DEFAULT_QUERY_INSTRUCTION
        )

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """内部批量 embed（不带指令前缀）。"""
        if not texts:
            return []
        all_embs: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.no_grad():
                outputs = self._model(**inputs)
            # last_token pooling：取每个样本最后一个非 padding token 的向量
            # left padding 时取 [:, 0, :]；right padding 时取 eos token 位置
            attention_mask = inputs["attention_mask"]
            last_idx = attention_mask.sum(dim=1) - 1  # [B]
            hidden = outputs.last_hidden_state  # [B, L, D]
            embs = hidden[torch.arange(hidden.size(0)), last_idx]  # [B, D]
            if self._normalize:
                embs = F.normalize(embs, p=2, dim=1)
            all_embs.extend([list(e.float().cpu().numpy()) for e in embs])
        return all_embs

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """入库文档：不加指令前缀，直接 embed。"""
        if not texts:
            return []
        safe = [self._safe_text(t) for t in texts]
        return self._embed_batch(safe)

    def embed_query(self, query: str) -> list[float]:
        """同步查询：加指令前缀。"""
        if not (query or "").strip():
            return []
        return self._embed_batch([self._query_instruction + self._safe_text(query)])[0]

    async def aembed_query(self, query: str) -> list[float]:
        """异步查询。"""
        return await asyncio.to_thread(self.embed_query, query)