"""
Word 文档解析子包
================
提供完整的 Word (.docx) 文档解析能力：
- 数据模型: WordResult, WordParagraph, WordTable, WordImage
- 配置: DEFAULT_OCR_BACKEND, IMAGE_MIN_AREA_FOR_OCR
- 核心解析: WordParser
- 表格处理: 合并单元格（gridSpan/vMerge）、多段表格、表单字段、签章检测
- 图片处理: 嵌入图片提取、OCR 识别、水印/签章检测
- 结果输出: print_summary, save_text, save_markdown
- 便捷函数: parse_word, parse_directory

默认使用 V1 版本，也可通过 doc_parser.word.v2 显式使用 V2 版本。
"""

from .v1.models import WordResult, WordParagraph, WordTable, WordImage
from .v1.config import (
    DEFAULT_OCR_BACKEND,
    IMAGE_MIN_AREA_FOR_OCR,
    ENABLE_SIGNING_DETECTION,
    SIGNING_KEYWORDS,
)
from .v1.parser import WordParser
from .v1.formatter import print_summary, save_markdown, save_text
from .v1.parser import parse_word, parse_directory

__all__ = [
    "WordResult",
    "WordParagraph",
    "WordTable",
    "WordImage",
    "DEFAULT_OCR_BACKEND",
    "IMAGE_MIN_AREA_FOR_OCR",
    "ENABLE_SIGNING_DETECTION",
    "SIGNING_KEYWORDS",
    "WordParser",
    "parse_word",
    "parse_directory",
    "print_summary",
    "save_text",
    "save_markdown",
]
