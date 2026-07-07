"""
Word 文档图片提取处理器
====================
从 python-docx 的 Document 对象中提取嵌入图片，
调用 doc_parser.image 模块的 OCR 引擎和水印/签章检测。

Word 中图片的存储方式：
- 图片数据通过 relationships 存储，rId 指向实际的图片文件
- 在文档 XML 中以 <w:drawing> 或 <w:pict> 元素引用
- 通过 document.part.related_parts 可获取所有图片的字节数据
"""

import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ai_sag.doc_parser.word.v1.models import WordImage
from ai_sag.doc_parser.word.v1.config import IMAGE_MIN_AREA_FOR_OCR, DEFAULT_OCR_BACKEND
from ai_sag.doc_parser.image.ocr import OCRBackend, get_ocr_engine
from ai_sag.doc_parser.image.processor import ImageProcessor

logger = logging.getLogger(__name__)

# python-docx 图片关系类型
_IMAGE_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"


def extract_images_from_document(
    document,
    ocr_backend: OCRBackend = DEFAULT_OCR_BACKEND,
    do_ocr: bool = True,
    output_dir: Optional[str] = None,
    file_stem: str = "word",
) -> list[WordImage]:
    """
    从 Word Document 中提取所有嵌入图片。

    Args:
        document: python-docx Document 对象
        ocr_backend: OCR 后端引擎
        do_ocr: 是否对图片执行 OCR 识别
        output_dir: 输出目录（为 None 时不保存到磁盘）
        file_stem: 文件名前缀

    Returns:
        WordImage 列表
    """
    images: list[WordImage] = []
    seen_rids: set[str] = set()
    img_index = 0

    # 遍历 document.part 的所有 relationships
    # document.part.related_parts 包含所有关联的子部件（图片、图表等）
    for rel_id, rel in document.part.rels.items():
        if rel.reltype != _IMAGE_REL_TYPE:
            continue
        if rel_id in seen_rids:
            continue
        seen_rids.add(rel_id)

        try:
            target_part = rel.target_part
            image_bytes = target_part.blob
            content_type = target_part.content_type

            # 解析图片扩展名
            ext = _content_type_to_ext(content_type)

            img_index += 1
            word_img = WordImage(
                image_index=img_index,
                image_bytes=image_bytes,
                content_type=content_type,
            )

            # 转为 cv2 数组进行图像处理
            img_array = ImageProcessor.image_to_cv2_array(image_bytes)
            if img_array is not None:
                h, w = img_array.shape[:2]
                word_img.width = w
                word_img.height = h

                # 水印检测
                watermark_result = ImageProcessor.detect_watermark(img_array)
                word_img.has_watermark = watermark_result.get('has_watermark', False)

                # 签章检测
                stamp_result = ImageProcessor.detect_stamp(img_array)
                word_img.has_stamp = stamp_result.get('has_stamp', False)

                # OCR
                if do_ocr:
                    img_area = h * w
                    if img_area >= IMAGE_MIN_AREA_FOR_OCR:
                        try:
                            ocr_engine = get_ocr_engine(ocr_backend)

                            # 有水印/签章时先清洗
                            ocr_source = img_array
                            already_preprocessed = False
                            if word_img.has_watermark or word_img.has_stamp:
                                cleaned = ImageProcessor.preprocess_for_ocr(
                                    img_array, grayscale=False
                                )
                                ocr_source = cleaned
                                already_preprocessed = True

                                # 保存清洗后图片
                                if output_dir:
                                    cleaned_filename = f"{file_stem}_img{img_index}_cleaned.png"
                                    cleaned_path = os.path.join(output_dir, "images", cleaned_filename)
                                    import cv2
                                    _, encoded = cv2.imencode('.png', cleaned)
                                    ImageProcessor.save_image(encoded.tobytes(), cleaned_path)
                                    word_img.cleaned_path = cleaned_path

                            ocr_texts = ocr_engine.recognize(
                                ocr_source, preprocess=not already_preprocessed
                            )
                            ocr_text = '\n'.join(ocr_texts).strip()
                            if ocr_text:
                                word_img.ocr_text = ocr_text
                                logger.info(f"图片 {img_index} OCR: {len(ocr_text)} 字符")
                        except Exception as e:
                            logger.warning(f"图片 {img_index} OCR 失败: {e}")

            # 保存原图到磁盘
            if output_dir:
                img_filename = f"{file_stem}_img{img_index}{ext}"
                img_path = os.path.join(output_dir, "images", img_filename)
                word_img.extracted_path = ImageProcessor.save_image(image_bytes, img_path)

            images.append(word_img)

        except Exception as e:
            logger.warning(f"提取图片失败 (rId={rel_id}): {e}")

    return images


def extract_images_from_paragraph(
    paragraph,
    all_images: list[WordImage],
) -> list[WordImage]:
    """
    从段落中识别关联的嵌入图片。

    通过匹配段落 XML 中的 r:embed 引用与 all_images 中的 rId 来关联。
    简化实现：按文档中出现顺序依次匹配。

    Args:
        paragraph: python-docx Paragraph 对象
        all_images: 文档中所有已提取的图片列表

    Returns:
        段落中关联的图片列表
    """
    from lxml import etree

    _NS = {
        'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
        'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
        'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    }

    paragraph_images = []

    # 查找段落中的 <a:blip r:embed="rIdX"/> 元素
    for blip in paragraph._element.findall('.//a:blip', _NS):
        r_embed = blip.get(f'{{{_NS["r"]}}}embed')
        if r_embed and hasattr(paragraph.part, 'rels'):
            rel = paragraph.part.rels.get(r_embed)
            if rel and rel.target_part:
                # 在 all_images 中找匹配
                for img in all_images:
                    if img.image_bytes == rel.target_part.blob:
                        paragraph_images.append(img)
                        break

    return paragraph_images


def _content_type_to_ext(content_type: str) -> str:
    """将 MIME content type 转为文件扩展名"""
    mapping = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
        "image/svg+xml": ".svg",
        "image/x-emf": ".emf",
        "image/x-wmf": ".wmf",
    }
    return mapping.get(content_type, ".png")
