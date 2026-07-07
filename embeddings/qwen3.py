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
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.embedding.qwen3_model_path)
        self._model = AutoModel.from_pretrained(cfg.embedding.qwen3_model_path).to(self._device).eval()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        import torch
        embs: list[list[float]] = []
        for t in texts:
            safe = self._safe_text(t)
            inputs = self._tokenizer(safe, return_tensors="pt", truncation=True, max_length=8192).to(self._device)
            with torch.no_grad():
                out = self._model(**inputs)
            emb = out.last_hidden_state[:, 0, :]
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            embs.append(emb[0].cpu().tolist())
        return embs