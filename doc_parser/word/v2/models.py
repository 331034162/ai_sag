"""
Word 文档解析器数据模型
=====================
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WordImage:
    """Word 文档中的嵌入图片"""
    image_index: int          # 在文档中的顺序索引（1-based）
    image_bytes: bytes = b""  # 图片原始字节数据
    content_type: str = ""    # MIME 类型（如 image/png）
    width: int = 0            # 原始宽度（像素）
    height: int = 0           # 原始高度（像素）
    ocr_text: str = ""        # OCR 识别文本
    extracted_path: str = ""  # 保存到磁盘的路径（如有 output_dir）
    cleaned_path: str = ""    # 清洗后图片路径（有水印时）
    has_watermark: bool = False
    has_stamp: bool = False


@dataclass
class WordTable:
    """Word 文档中的表格"""
    table_index: int          # 在文档中的顺序索引（1-based）
    row_count: int = 0
    col_count: int = 0
    markdown_text: str = ""   # 生成的 Markdown 表格
    form_fields: list[str] = field(default_factory=list)
    signing_info: list[str] = field(default_factory=list)
    title: str = ""           # 检测到的表格标题


@dataclass
class WordParagraph:
    """Word 文档中的段落"""
    text: str
    style_name: str = ""      # 段落样式名（如 Heading 1、Normal）
    is_heading: bool = False
    heading_level: int = 0    # 标题级别（1-9）
    is_list_item: bool = False
    list_level: int = 0       # 列表缩进级别
    has_bold: bool = False    # 是否含加粗文本
    images: list[WordImage] = field(default_factory=list)  # 段落内嵌入的图片


@dataclass
class WordResult:
    """Word 文档解析结果"""
    file_path: str
    file_name: str
    paragraph_count: int = 0
    table_count: int = 0
    image_count: int = 0
    paragraphs: list[WordParagraph] = field(default_factory=list)
    tables: list[WordTable] = field(default_factory=list)
    images: list[WordImage] = field(default_factory=list)   # 所有图片汇总
    full_text: str = ""        # 纯文本合并
    markdown_text: str = ""    # 完整 Markdown
    metadata: dict = field(default_factory=dict)
