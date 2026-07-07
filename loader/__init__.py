"""文档加载：支持 .md/.txt/.docx/.pdf/.xlsx，统一产出 LoadedDocument。

设计：每个格式一个 Reader，统一注册到 DocumentLoader。
基于 LlamaIndex 的 SimpleDirectoryReader 思路，但产出更轻量的 LoadedDocument。
"""
from __future__ import annotations

from .base import BaseReader, DocumentLoader, LoadError
from .readers import DocxReader, ExcelReader, MarkdownReader, PDFReader, TextReader

__all__ = [
    "BaseReader",
    "DocumentLoader",
    "LoadError",
    "MarkdownReader",
    "TextReader",
    "DocxReader",
    "PDFReader",
    "ExcelReader",
]