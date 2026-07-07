"""
结果输出函数
============
"""

import os
import logging

from ai_sag.doc_parser.pdf.v1.models import PDFResult

logger = logging.getLogger(__name__)


def print_summary(result: PDFResult):
    """打印解析摘要"""
    print("=" * 60)
    print(f"文件: {result.file_name}")
    print(f"类型: {result.pdf_type}")
    print(f"页数: {result.total_pages}")
    print(f"文本长度: {len(result.full_text)} 字符")
    print(f"Markdown 长度: {len(result.markdown_text)} 字符")

    image_count = sum(len(p.images) for p in result.pages)
    watermark_count = sum(
        sum(1 for img in p.images if img.has_watermark) for p in result.pages
    )
    stamp_count = sum(
        sum(1 for img in p.images if img.has_stamp) for p in result.pages
    )

    print(f"图片数量: {image_count}")
    if watermark_count > 0:
        print(f"水印检测: {watermark_count} 张")
    if stamp_count > 0:
        print(f"签章检测: {stamp_count} 张")
    print("=" * 60)


def print_page_text(result: PDFResult, max_chars: int = 500):
    """打印每页文本预览"""
    for page in result.pages:
        print(f"\n--- 第 {page.page_num} 页 ---")
        text = page.text[:max_chars] if page.text else "(无文本)"
        print(text)
        if len(page.text) > max_chars:
            print(f"... (共 {len(page.text)} 字符)")


def save_markdown(result: PDFResult, output_path: str):
    """保存 Markdown 到文件"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result.markdown_text)
    logger.info(f"Markdown 已保存: {output_path}")


def save_text(result: PDFResult, output_path: str):
    """保存纯文本到文件"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result.full_text)
    logger.info(f"纯文本已保存: {output_path}")

