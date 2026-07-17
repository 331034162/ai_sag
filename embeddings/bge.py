"""BGE 后端：基于 LlamaIndex HuggingFaceEmbedding，轻量稳定。"""
from __future__ import annotations

from ..base import Config
from .base import BaseEmbedder


class BgeEmbedder(BaseEmbedder):
    def __init__(self, cfg: Config) -> None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        self._normalize = cfg.embedding.normalize
        self._batch_size = cfg.embedding.batch_size
        self._model = HuggingFaceEmbedding(
            model_name=cfg.embedding.bge_model_path, device=cfg.embedding.device, normalize=self._normalize,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        safe = [self._safe_text(t) for t in texts]
        # batch_size<=0 时一次性送入（保留原行为）
        if self._batch_size <= 0:
            embs = self._model._get_text_embeddings(safe)
            return [list(e) for e in embs]
        # 按批送入，防大批量 OOM
        out: list[list[float]] = []
        for i in range(0, len(safe), self._batch_size):
            batch = safe[i:i + self._batch_size]
            embs = self._model._get_text_embeddings(batch)
            out.extend(list(e) for e in embs)
        return out