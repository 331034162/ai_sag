"""
结果输出函数
============
"""

import os
import logging

from .models import WordResult

logger = logging.getLogger(__name__)


def print_summary(result: WordResult):
    """打印解析摘要"""
    print("=" * 60)
    print(f"文件: {result.file_name}")
    print(f"段落数: {result.paragraph_count}")
    print(f"表格数: {result.table_count}")
    print(f"图片数: {result.image_count}")
    print(f"Markdown 长度: {len(result.markdown_text)} 字符")

    for wt in result.tables:
        info = f"  - 表格 {wt.table_index}: {wt.row_count}行 × {wt.col_count}列"
        if wt.title:
            info += f" [{wt.title}]"
        if wt.form_fields:
            info += f", {len(wt.form_fields)}个表单字段"
        if wt.signing_info:
            info += f", {len(wt.signing_info)}条签章信息"
        print(info)

    for img in result.images:
        info = f"  - 图片 {img.image_index}: {img.width}×{img.height}"
        if img.ocr_text:
            info += f", OCR {len(img.ocr_text)}字符"
        tags = []
        if img.has_watermark:
            tags.append("含水印")
        if img.has_stamp:
            tags.append("含签章")
        if tags:
            info += f" ({', '.join(tags)})"
        print(info)

    print("=" * 60)


def save_markdown(result: WordResult, output_path: str):
    """保存 Markdown 到文件"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result.markdown_text)
    logger.info(f"Markdown 已保存: {output_path}")


def save_text(result: WordResult, output_path: str):
    """保存纯文本到文件"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result.full_text)
    logger.info(f"纯文本已保存: {output_path}")
