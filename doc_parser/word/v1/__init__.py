"""
Word 文档解析器 V1
=================
"""

from ai_sag.doc_parser.word.v1.models import WordResult, WordParagraph, WordTable, WordImage
from ai_sag.doc_parser.word.v1.config import (
    DEFAULT_OCR_BACKEND,
    IMAGE_MIN_AREA_FOR_OCR,
    ENABLE_SIGNING_DETECTION,
    SIGNING_KEYWORDS,
)
from ai_sag.doc_parser.word.v1.parser import WordParser, parse_word, parse_directory
from ai_sag.doc_parser.word.v1.formatter import print_summary, save_markdown, save_text

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
