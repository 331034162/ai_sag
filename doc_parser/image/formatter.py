"""
图像 OCR 结果输出函数
====================
统一的格式化输出，提供控制台打印和文件保存功能。
"""

import os
import logging

from ai_sag.doc_parser.image.models import ImageOCRResult

logger = logging.getLogger(__name__)


def print_summary(result: ImageOCRResult):
    print("=" * 60)
    print(f"文件: {result.file_name or result.file_path}")
    print(f"OCR 后端: {result.ocr_backend}")
    print(f"预处理: {'是' if result.preprocessed else '否'}")
    print(f"OCR 文本长度: {len(result.ocr_text)} 字符")
    if result.has_watermark:
        wm = result.watermark_info
        print(f"水印检测: 类型={wm.get('type', 'unknown')}, 置信度={wm.get('confidence', 0):.2f}")
    if result.has_stamp:
        st = result.stamp_info
        print(f"签章检测: 数量={len(st.get('stamps', []))}, 置信度={st.get('confidence', 0):.2f}")
    print("=" * 60)


def print_ocr_text(result: ImageOCRResult, max_chars: int = 500):
    print(f"\n--- {result.file_name or result.file_path} ---")
    text = result.ocr_text[:max_chars] if result.ocr_text else "(无 OCR 文本)"
    print(text)
    if len(result.ocr_text) > max_chars:
        print(f"... (共 {len(result.ocr_text)} 字符)")


def save_text(result: ImageOCRResult, output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result.ocr_text)
    logger.info(f"OCR 文本已保存: {output_path}")


def save_summary(result: ImageOCRResult, output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"文件: {result.file_name or result.file_path}\n")
        f.write(f"OCR 后端: {result.ocr_backend}\n")
        f.write(f"预处理: {'是' if result.preprocessed else '否'}\n")
        f.write(f"OCR 文本长度: {len(result.ocr_text)} 字符\n")
        if result.has_watermark:
            f.write(f"水印: 类型={result.watermark_info.get('type', 'unknown')}, "
                    f"置信度={result.watermark_info.get('confidence', 0):.2f}\n")
        if result.has_stamp:
            f.write(f"签章: 数量={len(result.stamp_info.get('stamps', []))}, "
                    f"置信度={result.stamp_info.get('confidence', 0):.2f}\n")
        f.write(f"\n--- OCR 文本 ---\n")
        f.write(result.ocr_text)
    logger.info(f"摘要已保存: {output_path}")


__all__ = [
    "print_summary",
    "print_ocr_text",
    "save_text",
    "save_summary",
]
