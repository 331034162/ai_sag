"""
数据结构定义
============
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ImageInfo:
    """图片信息"""
    page_num: int
    image_index: int
    bbox: tuple
    width: int
    height: int
    extracted_path: Optional[str] = None
    has_watermark: bool = False
    has_stamp: bool = False
    cleaned_path: Optional[str] = None
    ocr_text: str = ""  # 图片 OCR 识别的文本内容


@dataclass
class PDFTable:
    """PDF 中检测到的表格"""
    page_num: int
    table_index: int          # 在文档中的顺序索引（1-based）
    row_count: int = 0
    col_count: int = 0
    markdown_text: str = ""   # 生成的 Markdown 表格
    form_fields: list[str] = field(default_factory=list)
    signing_info: list[str] = field(default_factory=list)
    title: str = ""           # 检测到的表格标题
    bbox: tuple = ()          # 表格在页面上的边界框 (x0, y0, x1, y1)
    source: str = ""          # 表格来源: "struct_tree" | "visual"


@dataclass
class PageContent:
    """单页内容"""
    page_num: int
    text: str = ""
    images: list[ImageInfo] = field(default_factory=list)
    tables: list[PDFTable] = field(default_factory=list)
    is_image_only: bool = False
    has_watermark: bool = False
    has_stamp: bool = False


@dataclass
class PDFResult:
    """PDF 解析结果"""
    file_path: str
    file_name: str
    total_pages: int
    pages: list[PageContent] = field(default_factory=list)
    tables: list[PDFTable] = field(default_factory=list)   # 所有表格汇总
    full_text: str = ""
    markdown_text: str = ""
    pdf_type: str = ""
    metadata: dict = field(default_factory=dict)
    ocr_backend: str = ""  # 使用的 OCR 后端
