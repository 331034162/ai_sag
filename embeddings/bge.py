"""BGE 后端：基于 llamaindex HuggingFaceEmbedding，CLS pooling（默认），轻量稳定。"""
from __future__ import annotations

from ..base import Config
from .base import BaseEmbedder


class BgeEmbedder(BaseEmbedder):
    """bge-small-zh-v1.5 等 BGE 模型。

    - pooling: CLS（HuggingFaceEmbedding 默认值，BGE 训练时即用 CLS）
    - 无指令前缀（BGE 不是指令微调模型）
    - 自身 max_length 上限 512（bge-small-zh-v1.5 的位置编码硬上限）
      配置 SAG_EMBEDDING_MAX_LENGTH 再大也会被压缩到模型实际容量，
      防止超长文本输入导致 "The size of tensor a (1110) must match the size of tensor b (512)" 报错。
    """
    def __init__(self, cfg: Config) -> None:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from transformers import AutoConfig

        # 读取模型自身的位置编码上限（bge-small-zh-v1.5 = 512），对 max_length 做上限保护
        model_cfg = AutoConfig.from_pretrained(cfg.embedding.model_path, trust_remote_code=True)
        model_max = getattr(model_cfg, "max_position_embeddings", 512)
        max_length = min(cfg.embedding.max_length, model_max)

        self._model = HuggingFaceEmbedding(
            model_name=cfg.embedding.model_path,
            device=cfg.embedding.device,
            max_length=max_length,
            normalize=cfg.embedding.normalize,
            embed_batch_size=cfg.embedding.batch_size if cfg.embedding.batch_size > 0 else 32,
            trust_remote_code=True,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        safe = [self._safe_text(t) for t in texts]
        embs = self._model._get_text_embeddings(safe)
        return [list(e) for e in embs]

    def embed_query(self, query: str) -> list[float]:
        """BGE 不是指令微调模型，查询和文档走同一路径。"""
        return self.embed_text(query)

    async def aembed_query(self, query: str) -> list[float]:
        return await self.aembed_text(query)