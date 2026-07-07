"""切分器工厂：按配置 mode 创建对应实现，便于切换 markdown/sentence/token/code/semantic/auto。

mode 说明：
- auto     自动按文档类型选择（md→markdown，其余→sentence），适配性最好，推荐默认
- markdown 按 Markdown 标题切分（保留层级 + 超长兜底）
- sentence 按句子边界 + chunk_size 窗口切分
- token    按 token 精确切分
- code     按语法结构切分（需配合 language）
- semantic 按语义相似度切分（chunk 语义完整，适合 SAG 事件抽取，但入库较慢）
"""
from __future__ import annotations

from ..base import Config
from .auto_splitter import AutoSplitter
from .base import BaseSplitter
from .chunk_splitter import ChunkSplitter


def create_splitter(cfg: Config, embed_model=None) -> BaseSplitter:
    mode = cfg.splitter.mode.lower()
    if mode == "auto":
        return AutoSplitter(
            chunk_size=cfg.splitter.chunk_size,
            chunk_overlap=cfg.splitter.chunk_overlap,
            language=cfg.splitter.language,
        )
    if mode in ("markdown", "sentence", "token", "code"):
        return ChunkSplitter(
            mode=mode, chunk_size=cfg.splitter.chunk_size,
            chunk_overlap=cfg.splitter.chunk_overlap, language=cfg.splitter.language,
        )
    if mode == "semantic":
        # semantic 模式需要 embed_model，优先用传入的，否则从 config 创建
        if embed_model is None:
            from ..embeddings import create_embedder
            embedder = create_embedder(cfg)
            embed_model = getattr(embedder, '_model', None)
            if embed_model is None:
                raise ValueError("semantic 模式需要 embed_model，但 embedder 无 _model 属性")
        return ChunkSplitter(
            mode="semantic", chunk_size=cfg.splitter.chunk_size,
            chunk_overlap=cfg.splitter.chunk_overlap, language=cfg.splitter.language,
            embed_model=embed_model,
            breakpoint_percentile_threshold=cfg.splitter.breakpoint_percentile_threshold,
        )
    raise ValueError(f"未知切分模式: {mode}（支持: auto / markdown / sentence / token / code / semantic）")