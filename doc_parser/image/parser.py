"""
图像 OCR 解析器
================
通过参数控制是否启用表格识别：

- enable_table_recognition=False：纯 OCR 文字识别
- enable_table_recognition=True：文字识别 + 表格识别，保持原始空间布局

表格识别流程：
1. recognize_with_positions() 获取带位置的文字块
2. recognize_tables_in_image() 获取结构化表格数据
3. reconstruct_structured_text() 按阅读顺序裁切：
   - 段落文字保留
   - 表格区域替换为 Markdown 表格
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from ai_sag.doc_parser.image.config import (
    setup_logging,
    OCR_FONT_NAME,
    OCR_FONT_MIN_SIZE,
)
from ai_sag.doc_parser.image.models import ImageOCRResult
from ai_sag.doc_parser.image.processor import ImageProcessor
from ai_sag.doc_parser.image.ocr import (
    BaseOCREngine,
    OCRBackend,
    get_ocr_engine,
)
from ai_sag.doc_parser.image.table.config import (
    ENABLE_TABLE_RECOGNITION,
    TABLE_RECOGNITION_BACKEND,
)
from ai_sag.doc_parser.image.table.models import TableRecognitionResult
from ai_sag.doc_parser.image.table.recognizer import recognize_tables_in_image

setup_logging()
logger = logging.getLogger(__name__)



# ============================================================
# 统一解析器
# ============================================================

class ImageParser:
    """
    图片 OCR 解析器

    参数：
        ocr_backend:              OCR 后端 ("paddleocr" / "rapidocr")
        preprocess:               是否预处理图像
        detect_watermark:         是否检测水印
        detect_stamp:             是否检测签章
        save_images:              是否保存中间图片
        output_dir:               图片保存目录
        enable_table_recognition: 是否启用表格识别（默认 True，受全局配置控制）
        table_backend:            表格识别后端 ("paddle" / "visual")
        table_ocr_engine:         VisualTableRecognizer 所需的 OCR 引擎实例
    """

    def __init__(
        self,
        ocr_backend: OCRBackend = "rapidocr",
        preprocess: bool = True,
        detect_watermark: bool = True,
        detect_stamp: bool = True,
        save_images: bool = False,
        output_dir: Optional[str] = None,
        # 表格识别参数
        enable_table_recognition: bool = True,
        table_backend: str = "paddle",
        table_ocr_engine: Optional[BaseOCREngine] = None,
    ):
        self.ocr_backend = ocr_backend
        self.preprocess = preprocess
        self.detect_watermark = detect_watermark
        self.detect_stamp = detect_stamp
        self.save_images = save_images
        self.output_dir = output_dir
        # 表格识别
        self.enable_table_recognition = (
            enable_table_recognition and ENABLE_TABLE_RECOGNITION
        )
        self.table_backend = table_backend or TABLE_RECOGNITION_BACKEND
        self.table_ocr_engine = table_ocr_engine

    def parse(self, source: Union[str, Path, bytes, np.ndarray]) -> ImageOCRResult:
        """
        解析图片

        当前实现：纯文字 OCR 模式（ocr_text=扁平文本, tables=[]）
        表格识别代码（recognize_tables_in_image / _reconstruct_structured_text）
        已保留在模块中但不再调用，因 PaddleOCR 表格结构识别错位严重。
        如需恢复，可参照 _reconstruct_structured_text 重新接入。
        """
        image_array, file_path = self._load_source(source)

        # ---- 公共部分：水印 / 签章 检测 ----
        watermark_info = {}
        has_watermark = False
        if self.detect_watermark:
            watermark_info = ImageProcessor.detect_watermark(image_array)
            has_watermark = watermark_info.get('has_watermark', False)
            if has_watermark:
                logger.info(
                    f"检测到水印 (类型: {watermark_info['type']}, "
                    f"置信度: {watermark_info['confidence']:.2f})"
                )

        stamp_info = {}
        has_stamp = False
        if self.detect_stamp:
            stamp_info = ImageProcessor.detect_stamp(image_array)
            has_stamp = stamp_info.get('has_stamp', False)
            if has_stamp:
                logger.info(f"检测到签章 (置信度: {stamp_info['confidence']:.2f})")

        ocr_engine = get_ocr_engine(self.ocr_backend)

        # ---- 纯文字 OCR 模式 ----
        # 表格识别已关闭（PaddleOCR 结构识别错位严重），表格区域的文字
        # 会被当作普通文本行 OCR 出来，不带表格结构，但文字内容不丢失。
        ocr_text_list = ocr_engine.recognize(
            image_array, preprocess=self.preprocess,
        )
        ocr_text = "\n".join(ocr_text_list)
        tables: list[TableRecognitionResult] = []

        # ---- 保存中间图片 ----
        saved_paths = {}
        if self.save_images and self.output_dir:
            saved_paths = self._save_images(
                image_array, file_path, has_watermark, has_stamp,
            )

        return ImageOCRResult(
            file_path=file_path,
            ocr_text=ocr_text,
            has_watermark=has_watermark,
            watermark_info=watermark_info,
            has_stamp=has_stamp,
            stamp_info=stamp_info,
            preprocessed=self.preprocess,
            ocr_backend=self.ocr_backend,
            metadata=saved_paths,
            tables=tables,
        )

    def parse_directory(
        self,
        directory: str,
        extensions: tuple = (
            ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".gif",
        ),
    ) -> list[ImageOCRResult]:
        """批量解析目录下所有图片"""
        results = []
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"目录不存在: {directory}")
        for file_path in sorted(dir_path.iterdir()):
            if file_path.suffix.lower() in extensions:
                logger.info(f"正在处理: {file_path.name}")
                result = self.parse(str(file_path))
                results.append(result)
        return results

    # ------------------------------------------------------------------
    # 内部方法：加载 / 保存
    # ------------------------------------------------------------------

    def _load_source(
        self, source: Union[str, Path, bytes, np.ndarray]
    ) -> tuple[np.ndarray, str]:
        """加载图片来源（文件路径 / URL / bytes / numpy 数组）"""
        if isinstance(source, np.ndarray):
            return source.copy(), "<numpy_array>"
        if isinstance(source, bytes):
            image_array = ImageProcessor.image_to_cv2_array(source)
            if image_array is None:
                raise ValueError("无法解码图片字节数据")
            return image_array, "<bytes>"
        source_str = str(source)
        if source_str.startswith(("http://", "https://")):
            image_bytes = self._download_url(source_str)
            image_array = ImageProcessor.image_to_cv2_array(image_bytes)
            if image_array is None:
                raise ValueError(f"无法解码 URL 图片: {source_str}")
            return image_array, source_str
        path = Path(source_str)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {source_str}")
        image_array = cv2.imdecode(
            np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR,
        )
        if image_array is None:
            raise ValueError(f"无法解码图片文件: {source_str}")
        return image_array, source_str

    @staticmethod
    def _download_url(url: str) -> bytes:
        import urllib.request
        logger.info(f"正在下载图片: {url}")
        with urllib.request.urlopen(url) as resp:
            return resp.read()

    def _save_images(self, image_array, file_path, has_watermark, has_stamp) -> dict:
        """保存原始/清洗后的图片"""
        saved = {}
        os.makedirs(self.output_dir, exist_ok=True)
        if file_path and file_path not in ("<bytes>", "<numpy_array>"):
            base_name = Path(file_path).stem
        else:
            base_name = "image"
        original_path = os.path.join(self.output_dir, f"{base_name}_original.png")
        cv2.imwrite(original_path, image_array)
        saved['original_path'] = original_path
        if has_watermark or has_stamp:
            ocr_engine = get_ocr_engine(self.ocr_backend)
            cleaned = ImageProcessor.preprocess_for_ocr(
                image_array, remove_watermark_flag=has_watermark,
                grayscale=ocr_engine._preprocess_grayscale,
                binary=ocr_engine._preprocess_binary,
            )
            cleaned_path = os.path.join(self.output_dir, f"{base_name}_cleaned.png")
            cv2.imwrite(cleaned_path, cleaned)
            saved['cleaned_path'] = cleaned_path
        return saved


# ============================================================
# 便捷函数
# ============================================================

def parse_image(
    source: Union[str, Path, bytes, np.ndarray],
    ocr_backend: OCRBackend = "rapidocr",
    preprocess: bool = True,
    detect_watermark: bool = True,
    detect_stamp: bool = True,
    enable_table_recognition: bool = True,
    table_backend: str = "paddle",
) -> ImageOCRResult:
    """便捷函数：解析单张图片"""
    parser = ImageParser(
        ocr_backend=ocr_backend,
        preprocess=preprocess,
        detect_watermark=detect_watermark,
        detect_stamp=detect_stamp,
        enable_table_recognition=enable_table_recognition,
        table_backend=table_backend,
    )
    return parser.parse(source)


def parse_directory(
    directory: str,
    ocr_backend: OCRBackend = "rapidocr",
    preprocess: bool = True,
    detect_watermark: bool = True,
    detect_stamp: bool = True,
    enable_table_recognition: bool = True,
    table_backend: str = "paddle",
) -> list[ImageOCRResult]:
    """便捷函数：批量解析目录下的所有图片"""
    parser = ImageParser(
        ocr_backend=ocr_backend,
        preprocess=preprocess,
        detect_watermark=detect_watermark,
        detect_stamp=detect_stamp,
        enable_table_recognition=enable_table_recognition,
        table_backend=table_backend,
    )
    return parser.parse_directory(directory)


# ============================================================
# PyMuPDF 集成适配器
# ============================================================

def ocr_for_pymupdf(
    page,
    ocr_engine: BaseOCREngine,
    dpi: int = 300,
    pixmap=None,
    **_kwargs,
):
    """PyMuPDF OCR 回调：渲染页面 → OCR 识别 → 写入文字层

    用法（传给 pymupdf4llm.to_markdown）：
        kwargs["ocr_function"] = lambda page, **kw: ocr_for_pymupdf(
            page, ocr_engine=my_engine, **kw
        )
    """
    import fitz

    if pixmap is None:
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pixmap = page.get_pixmap(matrix=mat)

    img_array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )

    texts = ocr_engine.recognize(img_array, preprocess=True)

    if texts:
        font = fitz.Font("cjk")
        page.insert_font(fontname=OCR_FONT_NAME, fontbuffer=font.buffer)

        rect = page.rect
        available_height = rect.height - 10
        fontsize = min(
            max(OCR_FONT_MIN_SIZE, rect.height / 60),
            available_height / (len(texts) + 1),
        )
        for i, text in enumerate(texts):
            page.insert_text(
                rect.tl + (5, fontsize * (i + 1) + 5),
                text,
                fontsize=fontsize,
                fontname=OCR_FONT_NAME,
            )
