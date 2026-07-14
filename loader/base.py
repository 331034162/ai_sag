"""加载器抽象与调度入口。"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path

from ..base import LoadedDocument


class LoadError(Exception):
    """文档加载异常。"""


class BaseReader(ABC):
    """格式 Reader 抽象基类。"""

    suffixes: tuple[str, ...] = ()

    @abstractmethod
    def read(self, path: str, title: str | None = None,
             ocr_images: bool | None = None,
             ocr_backend: str | None = None) -> LoadedDocument:
        ...


class DocumentLoader:
    """根据文件后缀分发到对应 Reader。"""

    def __init__(self) -> None:
        self._readers: dict[str, BaseReader] = {}

    def register(self, reader: BaseReader) -> None:
        for suf in reader.suffixes:
            self._readers[suf.lower()] = reader

    @classmethod
    def default(cls, config=None) -> "DocumentLoader":
        from .readers import CSVReader, DocxReader, ExcelReader, MarkdownReader, PDFReader, TextReader
        loader = cls()
        loader.register(MarkdownReader())
        loader.register(TextReader())
        # 透传 PdfDocParserConfig 给 Docx/PDF/Excel Reader，未提供时用各自 parse_* 默认值
        # Excel 也接收 config，用于注入 upload_tmp_dir 给样式表降级副本
        doc_parser_config = getattr(config, "doc_parser", None) if config is not None else None
        loader.register(DocxReader(doc_parser_config=doc_parser_config))
        loader.register(PDFReader(doc_parser_config=doc_parser_config))
        loader.register(ExcelReader(doc_parser_config=doc_parser_config))
        loader.register(CSVReader())
        return loader

    def load(self, path: str, title: str | None = None,
             ocr_images: bool | None = None,
             ocr_backend: str | None = None) -> LoadedDocument:
        if not os.path.exists(path):
            raise LoadError(f"文件不存在: {path}")
        suf = Path(path).suffix.lower().lstrip(".")
        reader = self._readers.get(suf)
        if reader is None:
            raise LoadError(f"不支持的文件类型: .{suf}（支持 .md/.txt/.docx/.pdf/.xlsx/.csv）"
                            + (f"；.xls 请先转换为 .xlsx 或 .csv" if suf == "xls" else ""))
        if title is None:
            title = Path(path).stem
        return reader.read(path, title=title, ocr_images=ocr_images, ocr_backend=ocr_backend)

    def load_text(self, title: str, content: str) -> LoadedDocument:
        """直接由文本构造文档（跳过文件解析）。"""
        if not content or not content.strip():
            raise LoadError("文档内容为空")
        return LoadedDocument(title=title, content=content, file_type="text")