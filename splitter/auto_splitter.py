"""自动切分器：根据文档类型自动选择最适配的切分策略。

适配策略：
- md / markdown  → MarkdownNodeParser（按标题切，保留层级）
                   （semantic 模式下 md 也走 semantic，保持原行为）
- xlsx / xls / csv → TableSplitter（按数据行切分，每行带表头列名，确保表格实体不丢失）
- txt / docx / pdf / 其它 → 根据 default_mode 选 semantic 或 sentence

表格类型必须走 TableSplitter 的原因：Excel/CSV 都是 CSV 文本格式，
若走 sentence/semantic 切分，表格行会被打散，导致"创建人"等列名与值分离，
LLM 无法正确识别人名实体（如"汪晨"被漏抽）。TableSplitter 用 csv.reader 解析，
按"列名: 值"格式保留每行的列名上下文，确保表格实体可被准确抽取。
"""
from __future__ import annotations

from ..base import Chunk, LoadedDocument
from .base import BaseSplitter
from .chunk_splitter import ChunkSplitter
from .table_splitter import TableSplitter

_MARKDOWN_TYPES = {"md", "markdown"}
# 表格类文档：走 TableSplitter（按数据行切分，每行带表头列名，确保表格实体不丢失）
# xlsx/xls：V6 解析产出 CSV 文本（带结构识别前缀）
# csv：CSVReader 直接产出 CSV 原始文本（无前缀，走兼容模式）
_TABLE_TYPES = {"xlsx", "xls", "csv"}


class AutoSplitter(BaseSplitter):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 100,
                 language: str = "python",
                 default_mode: str = "sentence",
                 embed_model=None,
                 breakpoint_percentile_threshold: int = 95,
                 table_chunk_size: int = 0) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.language = language
        self._default_mode = default_mode
        # 表格专用 chunk_size：0 时回退到通用 chunk_size
        effective_table_size = table_chunk_size if table_chunk_size > 0 else chunk_size
        self._markdown = ChunkSplitter(
            mode="markdown", chunk_size=chunk_size, chunk_overlap=chunk_overlap, language=language)
        self._sentence = ChunkSplitter(
            mode="sentence", chunk_size=chunk_size, chunk_overlap=chunk_overlap, language=language)
        self._table = TableSplitter(
            chunk_size=effective_table_size, chunk_overlap=chunk_overlap)
        self._semantic = None
        if default_mode == "semantic":
            if embed_model is None:
                raise ValueError("default_mode=semantic 需要 embed_model 参数")
            self._semantic = ChunkSplitter(
                mode="semantic", chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                language=language, embed_model=embed_model,
                breakpoint_percentile_threshold=breakpoint_percentile_threshold)

    def split(self, doc: LoadedDocument, source_id: str, document_id: str) -> list[Chunk]:
        splitter = self._select(doc.file_type)
        return splitter.split(doc, source_id, document_id)

    def _select(self, file_type: str) -> BaseSplitter:
        ft = (file_type or "").lower()
        if ft in _TABLE_TYPES:
            return self._table
        if ft in _MARKDOWN_TYPES and self._default_mode != "semantic":
            return self._markdown
        if self._default_mode == "semantic" and self._semantic is not None:
            return self._semantic
        return self._sentence