"""
doc_parser - 文档解析工具包
==========================
支持多种文档格式的解析，当前包含：
- pdf: PDF 文档解析（支持文本/混合/扫描版）
- excel: Excel 文档解析（支持合并单元格/货币格式/分组表头/签章检测）
- word: Word 文档解析（支持段落样式/合并表格/嵌入图片OCR/水印签章检测）
- image: 图像 OCR 解析（支持水印检测/签章检测/多种 OCR 后端，可被 PDF/Word/Excel 复用）
"""

from ai_sag.doc_parser.pdf import (
    PDFParser,
    PDFTypeDetector,
    ImageProcessor,
    WatermarkHandler,
    StampDetector,
    ImagePreprocessor,
    ImageInfo,
    PageContent,
    PDFResult,
    parse_pdf,
    parse_directory as parse_pdf_directory,
    print_summary as print_pdf_summary,
    save_text as save_pdf_text,
    save_markdown as save_pdf_markdown,
)

from ai_sag.doc_parser.excel import (
    ExcelParser,
    SheetContent,
    ExcelResult,
    ENABLE_SIGNING_DETECTION,
    SIGNING_KEYWORDS,
    parse_excel,
    parse_directory as parse_excel_directory,
    print_summary as print_excel_summary,
    save_text as save_excel_text,
    save_markdown as save_excel_markdown,
)

from ai_sag.doc_parser.word import (
    WordParser,
    WordResult,
    WordParagraph,
    WordTable,
    WordImage,
    parse_word,
    parse_directory as parse_word_directory,
    print_summary as print_word_summary,
    save_text as save_word_text,
    save_markdown as save_word_markdown,
)

from ai_sag.doc_parser.image import (
    ImageParser,
    ImageOCRResult,
    parse_image,
    parse_directory as parse_image_directory,
    print_summary as print_image_summary,
    print_ocr_text as print_image_ocr_text,
    save_text as save_image_text,
    save_summary as save_image_summary,
)

__all__ = [
    # PDF
    "PDFParser",
    "PDFTypeDetector",
    "ImageProcessor",
    "WatermarkHandler",
    "StampDetector",
    "ImagePreprocessor",
    "ImageInfo",
    "PageContent",
    "PDFResult",
    "parse_pdf",
    "parse_pdf_directory",
    "print_pdf_summary",
    "save_pdf_text",
    "save_pdf_markdown",
    # Excel
    "ExcelParser",
    "SheetContent",
    "ExcelResult",
    "ENABLE_SIGNING_DETECTION",
    "SIGNING_KEYWORDS",
    "parse_excel",
    "parse_excel_directory",
    "print_excel_summary",
    "save_excel_text",
    "save_excel_markdown",
    # Word
    "WordParser",
    "WordResult",
    "WordParagraph",
    "WordTable",
    "WordImage",
    "parse_word",
    "parse_word_directory",
    "print_word_summary",
    "save_word_text",
    "save_word_markdown",
    # Image
    "ImageParser",
    "ImageOCRResult",
    "parse_image",
    "parse_image_directory",
    "print_image_summary",
    "print_image_ocr_text",
    "save_image_text",
    "save_image_summary",
]
