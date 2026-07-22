"""文本清洗器：把 loader 产出的原始文本规整为适合切分与抽取的干净文本。

清洗步骤（可按需开关）：
1. 去 HTML/XML 标签
2. 统一换行符（\r\n / \r → \n）
3. 合并硬断行（非段落边界的单换行）
4. 规整空白（全角空格、连续空格、行首尾空白）
5. 去除多余空行（连续空行压缩为一个）
"""
from __future__ import annotations

import re

from ..base import LoadedDocument


# 表格类文档：单换行是行分隔符，合并会破坏表格结构（行压成一行 → TableSplitter 失效 → 实体漏抽）
# 包含 xls 是为未来扩展预留（当前 loader 尚不支持 .xls，需转 xlsx/csv）
_TABLE_FILE_TYPES = frozenset({"xlsx", "xls", "csv"})

# 结构化文档：单换行承载结构语义（列表项/代码块/表格行），不能合并
_STRUCTURED_FILE_TYPES = frozenset({"md"}) | _TABLE_FILE_TYPES

# 列表行起始标记（markdown 有序列表/无序列表）
# 用于在非结构化文档中识别列表项，避免把列表项合并成一行
_LIST_LINE_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")


class TextCleaner:
    def __init__(self, *, strip_html: bool = False, merge_hard_breaks: bool = True,
                 collapse_blank_lines: bool = True, normalize_whitespace: bool = True,
                 protect_list_items: bool = True) -> None:
        self.strip_html = strip_html
        self.merge_hard_breaks = merge_hard_breaks
        self.collapse_blank_lines = collapse_blank_lines
        self.normalize_whitespace = normalize_whitespace
        # 在非结构化文档（docx/pdf/txt）中识别列表项并跳过合并
        # 关闭后所有单换行都会被合并（旧行为，不推荐）
        self.protect_list_items = protect_list_items

    # HTML 标签正则：要求 < 后紧跟字母（标签名开头）或 /（闭合标签），
    # 避免误伤数学公式 a<b、代码 List<String>、文本 金额<1000 等
    _HTML_TAG_RE = re.compile(r"</?\s*[a-z][a-z0-9]*\b[^>]*>")
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
        # 结构化文档（md/xlsx/xls/csv）整体跳过硬断行合并：
        # - md：列表项/代码块依赖单换行保持结构
        # - xlsx/xls/csv：单换行是行分隔符，合并会把数据行压成一行，破坏 TableSplitter 切分
        if self.merge_hard_breaks and doc.file_type not in _STRUCTURED_FILE_TYPES:
            if self.protect_list_items:
                # 按行扫描，列表项之间的单换行不合并（保留列表结构）
                text = self._merge_hard_breaks_preserving_lists(text)
            else:
                text = self._HARD_BREAK_RE.sub(" ", text)
        if self.collapse_blank_lines:
            text = self._MULTI_NEWLINE_RE.sub("\n\n", text)
            text = text.strip()
        return LoadedDocument(
            title=doc.title, content=text,
            source_path=doc.source_path, file_type=doc.file_type,
        )

    def clean_text(self, text: str, *, file_type: str = "md") -> str:
        """便捷接口：清洗纯文本。

        file_type 默认 md（保留单换行结构），调用方可按实际语义传入：
        - "md"/"xlsx"/"csv"：跳过硬断行合并
        - "txt"/"docx"/"pdf"：合并硬断行（保留列表项）
        """
        return self.clean(LoadedDocument(title="", content=text, file_type=file_type)).content

    @staticmethod
    def _merge_hard_breaks_preserving_lists(text: str) -> str:
        """合并硬断行，但保留列表项和段落边界的换行。

        规则：逐行扫描，满足以下任一条件时保留换行：
        - 前一行或当前行是列表项（- * + 或 数字. )）
        - 前一行或当前行为空行（段落分隔，即 \\n\\n 边界）
        否则用空格合并（硬断行）。这样 docx/pdf 转出的列表结构和段落结构都不会被压成一行。
        """
        lines = text.split("\n")
        if len(lines) <= 1:
            return text
        result: list[str] = [lines[0]]
        for i in range(1, len(lines)):
            prev = lines[i - 1]
            curr = lines[i]
            # 前一行或当前行是列表项 → 保留换行（保护列表结构）
            # 前一行或当前行为空 → 保留换行（保护段落分隔 \n\n）
            if (_LIST_LINE_RE.match(prev) or _LIST_LINE_RE.match(curr)
                    or prev == "" or curr == ""):
                result.append("\n")
            else:
                # 都不是列表项、都不是空行 → 合并为空格（硬断行）
                result.append(" ")
            result.append(curr)
        return "".join(result)