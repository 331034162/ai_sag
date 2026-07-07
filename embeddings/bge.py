"""BGE 后端：基于 LlamaIndex HuggingFaceEmbedding，轻量稳定。"""
from __future__ import annotations

from ..base import Config
from .base import BaseEmbedder


class BgeEmbedder(BaseEmbedder):
    def __init__(self, cfg: Config) -> None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        self._model = HuggingFaceEmbedding(
            model_name=cfg.embedding.bge_model_path, device=cfg.embedding.device, normalize=True,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        safe = [self._safe_text(t) for t in texts]
        embs = self._model._get_text_embeddings(safe)
        # 确保返回 list[list[float]]，避免 numpy 数组导致下游布尔判断出错
        return [list(e) for e in embs]