"""
OCR 引擎模块 — 文字识别（含位置）
================================

包含：
- OCRTextBlock：带位置信息的文字块数据类
- BaseOCREngine / PaddleOCREngine / RapidOCREngine：统一的引擎基类和实现
- get_ocr_engine()：工厂函数，单例缓存

职责：
- recognize()       → 纯文字列表 list[str]  （不需要位置时）
- recognize_with_positions() → 带位置文字块 list[OCRTextBlock]  （结构化/表格场景）

用法:
    from ai_sag.doc_parser.image.ocr import get_ocr_engine

    engine = get_ocr_engine("paddleocr")
    engine.recognize(img_array)                    # → ["行1", "行2"]
    engine.recognize_with_positions(img_array)     # → [OCRTextBlock(...), ...]
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ai_sag.doc_parser.image.processor import ImageProcessor
from ai_sag.doc_parser.image.config import (
    OCR_FONT_NAME,
    OCR_FONT_MIN_SIZE,
    OCR_CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)

OCRBackend = Literal["paddleocr", "rapidocr"]


@dataclass(frozen=True)
class OCRTextBlock:
    """单个 OCR 文字块（含位置信息）"""
    text: str
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def cx(self) -> float:
        """中心点 X"""
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        """中心点 Y"""
        return (self.y0 + self.y1) / 2

    def inside_bbox(self, bbox: tuple[int, int, int, int]) -> bool:
        """判断中心点是否在给定 bbox 内（含边界）"""
        if not bbox or len(bbox) < 4:
            return False
        x0_t, y0_t, x1_t, y1_t = bbox
        return (x0_t <= self.cx <= x1_t) and (y0_t <= self.cy <= y1_t)


# ============================================================
# 统一的引擎类定义
# ============================================================

class BaseOCREngine(ABC):
    """OCR 引擎抽象基类"""

    _preprocess_grayscale: bool = True
    _preprocess_binary: bool = True

    @property
    @abstractmethod
    def engine(self):
        """延迟初始化的 OCR 引擎实例"""
        ...

    @abstractmethod
    def _raw_recognize(self, img_array: np.ndarray):
        """调用引擎原始推理"""
        ...

    @abstractmethod
    def _parse_result(self, result) -> list[str]:
        """解析引擎原生结果为文本列表"""
        ...

    @abstractmethod
    def _parse_result_with_positions(self, result) -> list[OCRTextBlock]:
        """解析引擎原生结果为带位置的文本块列表"""
        ...

    def recognize(self, img_array: np.ndarray, preprocess: bool = True) -> list[str]:
        """执行纯 OCR 文字识别，返回文本列表（不含位置）"""
        if preprocess:
            img_array = ImageProcessor.preprocess_for_ocr(
                img_array,
                grayscale=self._preprocess_grayscale,
                binary=self._preprocess_binary,
            )
        raw = self._raw_recognize(img_array)
        texts = self._parse_result(raw)
        return texts

    def recognize_with_positions(
        self, img_array: np.ndarray, preprocess: bool = True
    ) -> list[OCRTextBlock]:
        """执行 OCR 文字识别，返回带位置信息的文本块列表"""
        if preprocess:
            img_array = ImageProcessor.preprocess_for_ocr(
                img_array,
                grayscale=self._preprocess_grayscale,
                binary=self._preprocess_binary,
            )
        raw = self._raw_recognize(img_array)
        return self._parse_result_with_positions(raw)

class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR 引擎：精度高，内部预处理完善，CPU 较慢"""

    _preprocess_grayscale = True
    _preprocess_binary = False

    def __init__(self):
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            from paddleocr import PaddleOCR
            logger.info("初始化 PaddleOCR...")
            self._engine = PaddleOCR(use_textline_orientation=True, lang='ch')
            logger.info("PaddleOCR 初始化完成")
        return self._engine

    def _raw_recognize(self, img_array: np.ndarray):
        import cv2
        if img_array.ndim == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        return self.engine.predict(img_array)

    def _parse_result(self, result) -> list[str]:
        if not result or not isinstance(result, list) or len(result) == 0:
            return []
        res = result[0]
        rec_texts = list(res.get('rec_texts', []))
        rec_scores = list(res.get('rec_scores', []))
        texts = []
        for i in range(min(len(rec_texts), len(rec_scores))):
            if rec_texts[i] and float(rec_scores[i]) > OCR_CONFIDENCE_THRESHOLD:
                texts.append(str(rec_texts[i]))
        return texts

    def _parse_result_with_positions(self, result) -> list[OCRTextBlock]:
        """PaddleOCR 返回文字框坐标（det_boxes / dt_polys）"""
        if not result or not isinstance(result, list) or len(result) == 0:
            return []
        res = result[0]
        rec_texts = list(res.get('rec_texts', []))
        rec_scores = list(res.get('rec_scores', []))
        # PaddleOCR 3.x 用 dt_polys，2.x 用 det_boxes
        det_boxes = res.get('dt_polys') or res.get('det_boxes') or []
        blocks: list[OCRTextBlock] = []
        n = min(len(rec_texts), len(rec_scores))
        for i in range(n):
            if not rec_texts[i]:
                continue
            if float(rec_scores[i]) <= OCR_CONFIDENCE_THRESHOLD:
                continue
            text = str(rec_texts[i])
            # 从 4 角点取 axis-aligned bbox
            if i < len(det_boxes) and det_boxes[i] is not None:
                box = np.array(det_boxes[i]).reshape(-1, 2)
                x0, y0 = int(box[:, 0].min()), int(box[:, 1].min())
                x1, y1 = int(box[:, 0].max()), int(box[:, 1].max())
            else:
                x0 = y0 = x1 = y1 = 0
            blocks.append(OCRTextBlock(text=text, x0=x0, y0=y0, x1=x1, y1=y1))
        return blocks


class RapidOCREngine(BaseOCREngine):
    """RapidOCR 引擎：基于 ONNX Runtime，CPU 速度快"""

    _preprocess_grayscale = True
    _preprocess_binary = False

    def __init__(self):
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR
            logger.info("初始化 RapidOCR...")
            self._engine = RapidOCR()
            logger.info("RapidOCR 初始化完成")
        return self._engine

    def _raw_recognize(self, img_array: np.ndarray):
        result, _ = self.engine(img_array)
        return result

    def _parse_result(self, result) -> list[str]:
        if not result:
            return []
        texts = []
        for item in result:
            text = str(item[1])
            confidence = float(item[2])
            if text and confidence > OCR_CONFIDENCE_THRESHOLD:
                texts.append(text)
        return texts

    def _parse_result_with_positions(self, result) -> list[OCRTextBlock]:
        """RapidOCR 返回 [bbox, text, confidence]，bbox 为 4 角点"""
        if not result:
            return []
        blocks: list[OCRTextBlock] = []
        for item in result:
            text = str(item[1])
            confidence = float(item[2])
            if not text or confidence <= OCR_CONFIDENCE_THRESHOLD:
                continue
            # item[0] = [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]
            bbox_raw = item[0]
            try:
                box = np.array(bbox_raw).reshape(-1, 2)
                x0, y0 = int(box[:, 0].min()), int(box[:, 1].min())
                x1, y1 = int(box[:, 0].max()), int(box[:, 1].max())
            except Exception:
                x0 = y0 = x1 = y1 = 0
            blocks.append(OCRTextBlock(text=text, x0=x0, y0=y0, x1=x1, y1=y1))
        return blocks


# ============================================================
# 工厂函数：单例缓存
# ============================================================

_engine_instances: dict[str, BaseOCREngine] = {}


def get_ocr_engine(backend: OCRBackend = "rapidocr") -> BaseOCREngine:
    """获取 OCR 引擎实例（单例缓存）

    Args:
        backend: OCR 后端类型 ("paddleocr" / "rapidocr")

    Returns:
        缓存的引擎实例
    """
    _registry = {
        "paddleocr": PaddleOCREngine,
        "rapidocr": RapidOCREngine,
    }
    if backend not in _registry:
        raise ValueError(
            f"不支持的 OCR 后端: {backend}，可选: {list(_registry.keys())}"
        )

    if backend not in _engine_instances:
        _engine_instances[backend] = _registry[backend]()
        logger.debug(f"OCR 引擎 [{backend}] 已创建")

    return _engine_instances[backend]


__all__ = [
    "OCRTextBlock",
    "BaseOCREngine",
    "PaddleOCREngine",
    "RapidOCREngine",
    "get_ocr_engine",
    "OCRBackend",
]
