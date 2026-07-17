"""Qwen3 后端：基于 transformers 原生实现，精度高但较重。"""
from __future__ import annotations

from ..base import Config
from .base import BaseEmbedder


class Qwen3Embedder(BaseEmbedder):
    def __init__(self, cfg: Config) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._device = torch.device(
            cfg.embedding.device if torch.cuda.is_available() and "cuda" in cfg.embedding.device else "cpu"
        )
        self._normalize = cfg.embedding.normalize
        self._max_length = cfg.embedding.qwen3_max_length
        self._batch_size = cfg.embedding.batch_size
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.embedding.qwen3_model_path)
        self._model = AutoModel.from_pretrained(cfg.embedding.qwen3_model_path).to(self._device).eval()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import torch
        safe = [self._safe_text(t) for t in texts]
        embs: list[list[float]] = []
        # batch_size<=0 时退化为单条循环（保留原行为）
        batch = self._batch_size if self._batch_size > 0 else 1
        for i in range(0, len(safe), batch):
            chunk = safe[i:i + batch]
            inputs = self._tokenizer(
                chunk, return_tensors="pt", truncation=True,
                max_length=self._max_length, padding=True,
            ).to(self._device)
            with torch.no_grad():
                out = self._model(**inputs)
            emb = out.last_hidden_state[:, 0, :]
            if self._normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            embs.extend(e.cpu().tolist() for e in emb)
        return embs