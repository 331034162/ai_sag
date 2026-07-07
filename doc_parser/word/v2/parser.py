"""
Word 文档解析器
==============
将 Word (.docx) 文档解析为结构化的 WordResult，保留文档排版格式：
- 段落样式（标题、正文、列表）
- 表格结构（合并单元格、多段表格、表单、签章检测）
- 嵌入图片（OCR 识别、水印/签章检测）

用法：
    # 便捷函数
    from ai_sag.doc_parser.word import parse_word, parse_directory

    result = parse_word("/path/to/file.docx")
    results = parse_directory("/path/to/dir")

    # 类实例
    from ai_sag.doc_parser.word import WordParser

    parser = WordParser(output_dir="./output", ocr_backend="rapidocr")
    result = parser.parse("/path/to/file.docx")
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional, Union

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from lxml import etree

from ai_sag.doc_parser.word.v2.config import setup_logging, DEFAULT_OCR_BACKEND
from ai_sag.doc_parser.word.v2.models import WordResult, WordParagraph, WordTable, WordImage
from ai_sag.doc_parser.word.v2.table_handler import table_to_markdown
from ai_sag.doc_parser.word.v2.image_handler import (
    extract_images_from_document,
    extract_images_from_paragraph,
)
from ai_sag.doc_parser.image.ocr import OCRBackend

# 初始化日志
setup_logging()

logger = logging.getLogger(__name__)

# Word XML 命名空间
_NS = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
}

# 标题样式名映射
_HEADING_STYLE_MAP = {
    'Heading 1': 1, 'Heading 2': 2, 'Heading 3': 3,
    'Heading 4': 4, 'Heading 5': 5, 'Heading 6': 6,
    'Heading 7': 7, 'Heading 8': 8, 'Heading 9': 9,
    'Title': 1,
    'Subtitle': 2,
}

# 中文标题样式（部分中文 Word 模板使用）
_CN_HEADING_PATTERN = re.compile(r'^标题\s*(\d+)$')


class WordParser:
    """
    Word 文档解析器。

    完整保留文档的排版结构：标题层级、正文段落、列表、表格（含合并单元格）、
    嵌入图片（可选 OCR），输出 Markdown 格式。

    Args:
        output_dir: 输出目录，传值则保存解析结果（md/txt/图片）到磁盘，
                    传 None 则不写磁盘，仅返回内存中的解析结果
        ocr_backend: OCR 后端引擎，"rapidocr"（默认，速度快）或 "paddleocr"（精度高）
        ocr_images: 是否对嵌入图片执行 OCR 识别
        image_min_area: 图片最小面积（像素²），低于此值跳过 OCR
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        ocr_backend: OCRBackend = DEFAULT_OCR_BACKEND,
        ocr_images: bool = True,
        image_min_area: int = 5000,
    ):
        self.output_dir = output_dir
        self.ocr_backend = ocr_backend
        self.ocr_images = ocr_images
        self.image_min_area = image_min_area

    def parse(self, file_path: str) -> WordResult:
        """
        解析单个 Word 文档。

        Args:
            file_path: Word 文件路径（.docx）

        Returns:
            WordResult 解析结果
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        if path.suffix.lower() != ".docx":
            raise ValueError(f"不支持的文件格式: {path.suffix}，仅支持 .docx")

        logger.info(f"开始解析: {path.name}")

        # 准备输出目录
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        # 打开文档
        doc = Document(str(path))

        # 提取所有图片
        all_images = extract_images_from_document(
            doc,
            ocr_backend=self.ocr_backend,
            do_ocr=self.ocr_images,
            output_dir=self.output_dir,
            file_stem=path.stem,
        )
        logger.info(f"提取到 {len(all_images)} 张图片")

        # 按文档顺序遍历所有元素（段落 + 表格）
        paragraphs: list[WordParagraph] = []
        tables: list[WordTable] = []
        md_lines: list[str] = []
        table_idx = 0

        body = doc.element.body
        for element in body:
            tag = etree.QName(element.tag).localname

            if tag == 'p':
                # 段落
                from docx.text.paragraph import Paragraph
                para = Paragraph(element, doc)
                wp = self._parse_paragraph(para, all_images)

                # 检查段落中是否有图片（内联图片）
                para_images = extract_images_from_paragraph(para, all_images)
                wp.images = para_images

                paragraphs.append(wp)
                md_lines.extend(self._paragraph_to_markdown(wp))

            elif tag == 'tbl':
                # 表格
                from docx.table import Table
                tbl = Table(element, doc)
                table_idx += 1

                result = table_to_markdown(tbl, table_index=table_idx - 1)

                wt = WordTable(
                    table_index=table_idx,
                    row_count=result["row_count"],
                    col_count=result["col_count"],
                    markdown_text=result["markdown"],
                    form_fields=result["form_fields"],
                    signing_info=result["signing_info"],
                    title=result["title"],
                )
                tables.append(wt)

                if wt.title:
                    md_lines.append(f"**{wt.title}**")
                    md_lines.append("")
                md_lines.append(wt.markdown_text)

        # 合并纯文本
        full_text = _markdown_to_plain("\n".join(md_lines))

        # 构建完整 Markdown
        md_parts = []
        md_parts.append(f"# {path.stem}")
        md_parts.append("")
        md_parts.append(f"> 源文件: `{file_path}`")
        md_parts.append(f"> 段落数: {len(paragraphs)} | 表格数: {len(tables)} | 图片数: {len(all_images)}")
        md_parts.append("")
        md_parts.append("---")
        md_parts.append("")
        md_parts.extend(md_lines)

        markdown_text = "\n".join(md_parts)

        result = WordResult(
            file_path=str(path),
            file_name=path.name,
            paragraph_count=len(paragraphs),
            table_count=len(tables),
            image_count=len(all_images),
            paragraphs=paragraphs,
            tables=tables,
            images=all_images,
            full_text=full_text,
            markdown_text=markdown_text,
            metadata={
                "ocr_backend": self.ocr_backend,
                "ocr_images": self.ocr_images,
            },
        )

        # 保存到磁盘
        if self.output_dir:
            from ai_sag.doc_parser.word.v2.formatter import save_markdown, save_text
            save_markdown(result, os.path.join(self.output_dir, f"{path.stem}.md"))
            save_text(result, os.path.join(self.output_dir, f"{path.stem}.txt"))

        logger.info(
            f"解析完成: {len(paragraphs)} 段落, {len(tables)} 表格, "
            f"{len(all_images)} 图片, 文本长度 {len(full_text)} 字符"
        )
        return result

    def parse_directory(self, dir_path: str, recursive: bool = False) -> list[WordResult]:
        """
        批量解析目录下的所有 Word 文件。

        Args:
            dir_path: 目录路径
            recursive: 是否递归子目录

        Returns:
            WordResult 列表
        """
        dp = Path(dir_path)
        if not dp.is_dir():
            raise NotADirectoryError(f"不是目录: {dir_path}")

        pattern = "**/*.docx" if recursive else "*.docx"
        files = list(dp.glob(pattern))

        results = []
        for f in sorted(files):
            if f.name.startswith("~$"):
                continue
            try:
                logger.info(f"解析: {f.name}")
                result = self.parse(str(f))
                results.append(result)
            except Exception as e:
                logger.error(f"解析失败 [{f.name}]: {e}")

        return results

    # ============================================================
    # 内部方法
    # ============================================================

    @staticmethod
    def _parse_paragraph(para, all_images: list[WordImage]) -> WordParagraph:
        """
        解析单个段落，提取样式信息和文本格式。

        Args:
            para: python-docx Paragraph 对象
            all_images: 文档中所有已提取的图片（用于关联段落内图片）

        Returns:
            WordParagraph
        """
        text = para.text.strip()
        style_name = para.style.name if para.style else "Normal"

        # 检测标题
        is_heading = False
        heading_level = 0

        if style_name in _HEADING_STYLE_MAP:
            is_heading = True
            heading_level = _HEADING_STYLE_MAP[style_name]
        else:
            # 中文标题样式
            m = _CN_HEADING_PATTERN.match(style_name)
            if m:
                is_heading = True
                heading_level = int(m.group(1))
            elif style_name.startswith('toc'):
                # 目录项不算标题
                pass
            elif text:
                # 回退：基于文本内容模式检测标题
                # 匹配中文序号标题，如 "一、xxx"、"二、xxx"
                cn_heading = re.match(r'^[一二三四五六七八九十]+、', text)
                # 匹配数字序号标题，如 "1. xxx"、"1.1 xxx"、"1.1.1 xxx"
                num_heading = re.match(r'^(\d+)(\.\d+)*[\s、．.]\s*\S', text)
                if cn_heading:
                    is_heading = True
                    heading_level = 1
                elif num_heading:
                    # 根据小数点数量判断层级：1. → L1, 1.1 → L2, 1.1.1 → L3
                    depth = len(num_heading.group(2) or '') // 2  # 每个 ".N" 占2字符
                    is_heading = True
                    heading_level = min(depth + 1, 6)

        # 检测列表
        is_list_item = False
        list_level = 0
        numPr = para._element.find('w:pPr/w:numPr', _NS)
        if numPr is not None:
            is_list_item = True
            ilvl = numPr.find('w:ilvl', _NS)
            if ilvl is not None:
                val = ilvl.get(f'{{{_NS["w"]}}}val')
                if val:
                    try:
                        list_level = int(val)
                    except ValueError:
                        pass
        elif style_name.lower().startswith('list'):
            # 回退：样式名以 "List" 开头（如 List Bullet、List Number）
            # 编号信息可能定义在样式中而非段落直接属性上
            is_list_item = True

        # 检测加粗
        has_bold = False
        for run in para.runs:
            if run.bold:
                has_bold = True
                break

        return WordParagraph(
            text=text,
            style_name=style_name,
            is_heading=is_heading,
            heading_level=heading_level,
            is_list_item=is_list_item,
            list_level=list_level,
            has_bold=has_bold,
        )

    @staticmethod
    def _paragraph_to_markdown(wp: WordParagraph) -> list[str]:
        """
        将 WordParagraph 转换为 Markdown 行。

        Args:
            wp: 解析后的段落对象

        Returns:
            Markdown 行列表
        """
        lines = []

        if not wp.text and not wp.images:
            # 空段落 → 空行（保留段落间距）
            lines.append("")
            return lines

        # 标题
        if wp.is_heading and wp.heading_level > 0:
            prefix = "#" * min(wp.heading_level, 6)
            lines.append(f"{prefix} {wp.text}")
            lines.append("")
            return lines

        # 列表项
        if wp.is_list_item:
            indent = "  " * wp.list_level
            lines.append(f"{indent}- {wp.text}")
            return lines

        # 普通段落
        text = wp.text
        if wp.has_bold and len(text) < 100:
            # 短文本且全部加粗 → 用 Markdown 加粗
            # 检查是否所有 run 都是加粗的
            pass  # 保留原始文本，不加额外标记

        if text:
            lines.append(text)
            lines.append("")

        # 段落中的图片
        for img in wp.images:
            if img.ocr_text.strip():
                lines.append(f"**[图片内容]**")
                lines.append("")
                lines.append(img.ocr_text)
                lines.append("")
            elif img.extracted_path:
                img_name = os.path.basename(img.extracted_path)
                lines.append(f"![{img_name}](images/{img_name})")
                lines.append("")

        return lines


# ============================================================
# 内部辅助
# ============================================================

def _markdown_to_plain(md_text: str) -> str:
    """将 Markdown 转为纯文本"""
    lines = md_text.split("\n")
    plain_lines = []
    for line in lines:
        stripped = line.strip()
        # 跳过分隔行
        if stripped.startswith("|") and all(c in "|- " for c in stripped):
            continue
        # 去掉 Markdown 标题标记
        m = re.match(r'^(#{1,6})\s+(.*)', stripped)
        if m:
            plain_lines.append(m.group(2))
        else:
            plain_lines.append(line)
    return "\n".join(plain_lines)


# ============================================================
# 便捷函数
# ============================================================

def parse_word(
    file_path: str,
    output_dir: Optional[str] = None,
    ocr_backend: OCRBackend = DEFAULT_OCR_BACKEND,
    ocr_images: bool = True,
) -> WordResult:
    """
    一键解析 Word 文档。

    Args:
        file_path: Word 文件路径（.docx）
        output_dir: 输出目录，传值则保存解析结果到磁盘，传 None 则仅返回内存结果
        ocr_backend: OCR 后端引擎
        ocr_images: 是否对嵌入图片做 OCR

    Returns:
        WordResult 解析结果
    """
    parser = WordParser(
        output_dir=output_dir,
        ocr_backend=ocr_backend,
        ocr_images=ocr_images,
    )
    return parser.parse(file_path)


def parse_directory(
    dir_path: str,
    output_dir: Optional[str] = None,
    recursive: bool = False,
    ocr_backend: OCRBackend = DEFAULT_OCR_BACKEND,
    ocr_images: bool = True,
) -> list[WordResult]:
    """
    一键批量解析目录下的所有 Word 文件。

    Args:
        dir_path: 目录路径
        output_dir: 输出目录
        recursive: 是否递归子目录
        ocr_backend: OCR 后端引擎
        ocr_images: 是否对嵌入图片做 OCR

    Returns:
        WordResult 列表
    """
    parser = WordParser(
        output_dir=output_dir,
        ocr_backend=ocr_backend,
        ocr_images=ocr_images,
    )
    return parser.parse_directory(dir_path, recursive=recursive)
