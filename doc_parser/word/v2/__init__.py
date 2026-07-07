"""
Word 文档解析器 V2
=================
"""

from .models import WordResult, WordParagraph, WordTable, WordImage
from .config import (
    DEFAULT_OCR_BACKEND,
    IMAGE_MIN_AREA_FOR_OCR,
    ENABLE_SIGNING_DETECTION,
    SIGNING_KEYWORDS,
)
from .parser import WordParser, parse_word, parse_directory
from .formatter import print_summary, save_markdown, save_text

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
