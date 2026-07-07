"""自动切分器：根据文档类型自动选择最适配的切分策略。

适配策略（默认）：
- md / markdown  → MarkdownNodeParser（按标题切，保留层级）+ SentenceSplitter 兜底超长
- txt / docx / pdf / xlsx / 其它 → SentenceSplitter（按句子+窗口，对纯文本最稳妥）

理由：docx/pdf 经 loader 解析后为纯文本，标题层级已丢失；只有原生 markdown
才有可靠的标题结构。因此"适配性最好"的策略是：仅对 markdown 用结构化切分，
其余统一按句子切，保证语义完整且不依赖可能缺失的标题。
"""
from __future__ import annotations

from ..base import Chunk, LoadedDocument
from .base import BaseSplitter
from .chunk_splitter import ChunkSplitter

_MARKDOWN_TYPES = {"md", "markdown"}


class AutoSplitter(BaseSplitter):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 100,
                 language: str = "python") -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.language = language
        self._markdown = ChunkSplitter(
            mode="markdown", chunk_size=chunk_size, chunk_overlap=chunk_overlap, language=language)
        self._sentence = ChunkSplitter(
            mode="sentence", chunk_size=chunk_size, chunk_overlap=chunk_overlap, language=language)

    def split(self, doc: LoadedDocument, source_id: str, document_id: str) -> list[Chunk]:
        splitter = self._select(doc.file_type)
        return splitter.split(doc, source_id, document_id)

    def _select(self, file_type: str) -> BaseSplitter:
        ft = (file_type or "").lower()
        if ft in _MARKDOWN_TYPES:
            return self._markdown
        return self._sentence