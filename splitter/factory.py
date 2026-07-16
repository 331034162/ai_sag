"""切分器工厂：按配置 mode 创建对应实现，便于切换 markdown/sentence/token/code/semantic/auto。

mode 说明：
- auto     自动按文档类型选择（md→markdown，xlsx→table，其余→sentence），适配性最好
- semantic 自动按文档类型选择（md→semantic，xlsx→table，其余→semantic），语义完整适合 SAG（默认）
- markdown 按 Markdown 标题切分（保留层级 + 超长兜底）
- sentence 按句子边界 + chunk_size 窗口切分
- token    按 token 精确切分
- code     按语法结构切分（需配合 language）

注意：auto/semantic 都走 AutoSplitter，xlsx 始终路由到 TableSplitter（按数据行切分，
每行带表头列名），避免表格实体（如人名）因列名-值分离而漏抽。
"""
from __future__ import annotations

from ..base import Config
from .auto_splitter import AutoSplitter
from .base import BaseSplitter
from .chunk_splitter import ChunkSplitter


def create_splitter(cfg: Config, embed_model=None) -> BaseSplitter:
    mode = cfg.splitter.mode.lower()
    if mode in ("auto", "semantic"):
        # auto/semantic 都走 AutoSplitter，按文件类型路由：
        # xlsx→TableSplitter，md→markdown(auto)或semantic，其余→semantic或sentence
        resolved_embed = embed_model
        if mode == "semantic" and resolved_embed is None:
            from ..embeddings import create_embedder
            embedder = create_embedder(cfg)
            resolved_embed = getattr(embedder, '_model', None)
            if resolved_embed is None:
                raise ValueError("semantic 模式需要 embed_model，但 embedder 无 _model 属性")
        default_mode = "semantic" if mode == "semantic" else "sentence"
        return AutoSplitter(
            chunk_size=cfg.splitter.chunk_size,
            chunk_overlap=cfg.splitter.chunk_overlap,
            language=cfg.splitter.language,
            default_mode=default_mode,
            embed_model=resolved_embed,
            breakpoint_percentile_threshold=cfg.splitter.breakpoint_percentile_threshold,
            table_chunk_size=cfg.splitter.table_chunk_size,
        )
    if mode in ("markdown", "sentence", "token", "code"):
        return ChunkSplitter(
            mode=mode, chunk_size=cfg.splitter.chunk_size,
            chunk_overlap=cfg.splitter.chunk_overlap, language=cfg.splitter.language,
        )
    raise ValueError(f"未知切分模式: {mode}（支持: auto / markdown / sentence / token / code / semantic）")