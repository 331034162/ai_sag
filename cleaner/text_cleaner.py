"""文本清洗器：把 loader 产出的原始文本规整为适合切分与抽取的干净文本。

清洗步骤（可按需开关）：
1. 去 HTML/XML 标签
2. 统一换行符（\r\n / \r → \n）
3. 合并硬断行（非段落边界的单换行）
4. 规整空白（全角空格、连续空格、行首尾空白）
5. 去除多余空行（连续空行压缩为一个）
6. 规整常见全角标点（可选）
"""
from __future__ import annotations

import re

from ..base import LoadedDocument


class TextCleaner:
    def __init__(self, *, strip_html: bool = True, merge_hard_breaks: bool = True,
                 collapse_blank_lines: bool = True, normalize_whitespace: bool = True) -> None:
        self.strip_html = strip_html
        self.merge_hard_breaks = merge_hard_breaks
        self.collapse_blank_lines = collapse_blank_lines
        self.normalize_whitespace = normalize_whitespace

    _HTML_TAG_RE = re.compile(r"<[^>]+>")
    _MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
    _MULTI_SPACE_RE = re.compile(r"[ \t\u3000]+")
    _HARD_BREAK_RE = re.compile(r"(?<!\n)\n(?!\n)")

    def clean(self, doc: LoadedDocument) -> LoadedDocument:
        text = doc.content
        if self.strip_html:
            text = self._HTML_TAG_RE.sub("", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if self.normalize_whitespace:
            lines = [self._MULTI_SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
            text = "\n".join(lines)
        # Markdown 文档跳过硬断行合并：列表项/代码块依赖单换行保持结构，
        # 合并且会将 "- 条目A\n- 条目B" 变成 "- 条目A - 条目B"，破坏独立语义。
        if self.merge_hard_breaks and doc.file_type != "md":
            text = self._HARD_BREAK_RE.sub(" ", text)
        if self.collapse_blank_lines:
            text = self._MULTI_NEWLINE_RE.sub("\n\n", text)
            text = text.strip()
        return LoadedDocument(
            title=doc.title, content=text,
            source_path=doc.source_path, file_type=doc.file_type,
            metadata={**doc.metadata, "cleaned": True},
        )

    def clean_text(self, text: str) -> str:
        return self.clean(LoadedDocument(title="", content=text)).content