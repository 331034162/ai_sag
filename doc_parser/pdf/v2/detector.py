"""
PDF 类型检测器
==============
判断 PDF 是纯文本 / 混合 / 扫描版
"""

import fitz  # pymupdf

from .config import IMAGE_ONLY_RATIO


class PDFTypeDetector:
    """PDF 类型检测器"""

    @staticmethod
    def detect(doc: fitz.Document) -> str:
        """
        检测 PDF 类型
        Returns: "text" | "mixed" | "image_only"

        检测逻辑：
        - 同时包含文字和图片的页面 → mixed 页
        - 只有文字的页面 → text_only 页
        - 只有图片的页面 → image_only 页
        - 只要存在 mixed 页，整体即为 "mixed"（有可直接提取的文字，不应走整页 OCR）
        - 纯图片页占比超过阈值 → "image_only"（扫描件，需要整页 OCR）
        """
        text_only_count, image_only_count, mixed_count = 0, 0, 0

        for page in doc:
            has_text = bool(page.get_text().strip())
            has_images = bool(page.get_images(full=True))

            if has_text and has_images:
                mixed_count += 1
            elif has_text:
                text_only_count += 1
            elif has_images:
                image_only_count += 1
            # 空白页（无文字也无图片）不计入任一类

        total = text_only_count + image_only_count + mixed_count
        if total == 0:
            return "text"

        # 存在文字+图片共存的页面，一定是混合型
        if mixed_count > 0:
            return "mixed"

        # 纯图片页占比超过阈值 → 扫描件
        image_ratio = image_only_count / total
        if image_ratio > IMAGE_ONLY_RATIO:
            return "image_only"

        # 有图片页但占比不够高 → 混合型
        if image_only_count > 0:
            return "mixed"

        return "text"
