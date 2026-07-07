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
class PageContent:
    """单页内容"""
    page_num: int
    text: str = ""
    images: list[ImageInfo] = field(default_factory=list)
    is_image_only: bool = False


@dataclass
class PDFResult:
    """PDF 解析结果"""
    file_path: str
    file_name: str
    total_pages: int
    pages: list[PageContent] = field(default_factory=list)
    full_text: str = ""
    markdown_text: str = ""
    pdf_type: str = ""
    metadata: dict = field(default_factory=dict)
    ocr_backend: str = ""  # 使用的 OCR 后端
