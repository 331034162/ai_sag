"""
图像预处理组件
==============
水印检测与去除、签章检测、OCR 预处理、图片保存
"""

import os
import logging
from typing import Optional

import cv2
import numpy as np

from ai_sag.doc_parser.image.config import (
    WATERMARK_CONTOUR_THRESHOLD,
    WATERMARK_BG_WHITE_RATIO,
    WATERMARK_BG_DARK_RATIO,
    WATERMARK_BG_CONTENT_RATIO,
    STAMP_MIN_AREA,
    STAMP_MAX_SIZE,
    STAMP_ELLIPSE_RATIO,
    STAMP_MIN_FIT_RATIO,
    STAMP_MIN_SOLIDITY,
)

logger = logging.getLogger(__name__)


class WatermarkHandler:
    """水印检测与去除"""

    @staticmethod
    def detect(image_array: np.ndarray) -> dict:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if image_array.ndim == 3 else image_array
        h, w = gray.shape
        result = {'has_watermark': False, 'confidence': 0.0, 'type': 'none', 'regions': []}

        # 方法 1：浅色半透明文本水印
        _, binary_light = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        light_ratio = np.sum(binary_light == 255) / (h * w)

        if light_ratio > 0.3:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(binary_light, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            small_contours = [c for c in contours if cv2.contourArea(c) < 1000]
            if len(small_contours) > WATERMARK_CONTOUR_THRESHOLD:
                result.update({
                    'has_watermark': True,
                    'confidence': min(len(small_contours) / 200.0, 1.0),
                    'type': 'text',
                    'regions': [cv2.boundingRect(c) for c in small_contours[:20]],
                })

        # 方法 2：对角线水印
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)
        if lines is not None and len(lines) > 10:
            diagonal_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if 30 < angle < 60 or 120 < angle < 150:
                    diagonal_lines.append(line)
            if len(diagonal_lines) > 5:
                result.update({
                    'has_watermark': True,
                    'confidence': max(result['confidence'], min(len(diagonal_lines) / 20.0, 1.0)),
                    'type': 'pattern',
                })

        # 方法 3：背景水印
        white_ratio = np.sum(gray >= 250) / (h * w)
        dark_ratio = np.sum(gray < 100) / (h * w)
        if white_ratio > WATERMARK_BG_WHITE_RATIO and dark_ratio < WATERMARK_BG_DARK_RATIO:
            light_gray_ratio = np.sum((gray >= 200) & (gray < 250)) / (h * w)
            has_color_content = False
            if image_array.ndim == 3:
                b_ch, g_ch, r_ch = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
                color_diff = np.max(np.stack([
                    np.abs(r_ch.astype(int) - g_ch.astype(int)),
                    np.abs(g_ch.astype(int) - b_ch.astype(int)),
                    np.abs(r_ch.astype(int) - b_ch.astype(int)),
                ]), axis=0)
                has_color_content = np.sum(color_diff > 25) / (h * w) > WATERMARK_BG_CONTENT_RATIO

            if light_gray_ratio > WATERMARK_BG_CONTENT_RATIO or has_color_content:
                bg_confidence = min((light_gray_ratio * 10 + (0.3 if has_color_content else 0)), 1.0)
                if bg_confidence > result['confidence']:
                    result.update({
                        'has_watermark': True,
                        'confidence': bg_confidence,
                        'type': 'background',
                    })
        return result

    @staticmethod
    def remove(image_array: np.ndarray, watermark_info: dict) -> np.ndarray:
        if not watermark_info['has_watermark']:
            return image_array

        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if image_array.ndim == 3 else image_array.copy()
        h, w = gray.shape

        if watermark_info['type'] == 'text':
            mask = np.zeros((h, w), dtype=np.uint8)
            for x, y, bw, bh in watermark_info['regions']:
                cv2.rectangle(mask, (x, y), (x + bw, y + bh), 255, -1)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.dilate(mask, kernel, iterations=2)
            return cv2.inpaint(image_array, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        if watermark_info['type'] == 'pattern':
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            if image_array.ndim == 3:
                return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            return enhanced

        if watermark_info['type'] == 'background':
            if image_array.ndim == 3:
                lab = cv2.cvtColor(image_array, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                enhanced = cv2.merge([l, a, b])
                return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                return clahe.apply(gray)

        return image_array


class StampDetector:
    """签章/印章检测"""

    @staticmethod
    def detect(image_array: np.ndarray) -> dict:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if image_array.ndim == 3 else image_array
        result = {'has_stamp': False, 'confidence': 0.0, 'stamps': []}

        red_mask = None
        if image_array.ndim == 3:
            lower_red1, upper_red1 = np.array([0, 0, 100]), np.array([80, 80, 255])
            lower_red2, upper_red2 = np.array([150, 0, 0]), np.array([255, 120, 120])
            mask1 = cv2.inRange(image_array, lower_red1, upper_red1)
            mask2 = cv2.inRange(image_array, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)

        edges = cv2.Canny(gray, 30, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        stamps = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < STAMP_MIN_AREA or len(c) < 8:
                continue

            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < STAMP_MIN_SOLIDITY:
                continue

            ellipse = cv2.fitEllipse(c)
            (cx, cy), (ma, MA), _ = ellipse
            ellipse_area = np.pi * (ma / 2) * (MA / 2)
            fit_ratio = area / ellipse_area if ellipse_area > 0 else 0
            if fit_ratio < STAMP_MIN_FIT_RATIO:
                continue

            if STAMP_ELLIPSE_RATIO[0] < ma / MA < STAMP_ELLIPSE_RATIO[1] and 30 < MA < STAMP_MAX_SIZE:
                x, y, w, h = cv2.boundingRect(c)
                stamps.append({'bbox': (x, y, w, h), 'center': (int(cx), int(cy)), 'area': area})

        if stamps:
            confidence = min(len(stamps) / 5.0, 1.0)
            if red_mask is not None and np.sum(red_mask > 0) > 500:
                confidence = max(confidence, 0.8)
            result.update({'has_stamp': True, 'confidence': confidence, 'stamps': stamps})

        return result


class ImagePreprocessor:
    """OCR 预处理（锐化、去噪、对比度增强）"""

    @staticmethod
    def _sharpen(image_array: np.ndarray, strength: str = 'moderate') -> np.ndarray:
        kernels = {
            'mild': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
            'moderate': np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),
            'strong': np.array([[1, 1, 1], [1, -7, 1], [1, 1, 1]]),
        }
        return cv2.filter2D(image_array, -1, kernels.get(strength, kernels['moderate']))

    @staticmethod
    def _unsharp_mask(image_array: np.ndarray, sigma: float = 1.0, amount: float = 1.5) -> np.ndarray:
        blurred = cv2.GaussianBlur(image_array, (0, 0), sigma)
        return cv2.addWeighted(image_array, 1.0 + amount, blurred, -amount, 0)

    @staticmethod
    def _assess_quality(image_array: np.ndarray) -> float:
        if image_array.ndim == 2:
            return cv2.Laplacian(image_array, cv2.CV_64F).var() / 500.0
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0

    @staticmethod
    def preprocess(image_array: np.ndarray, grayscale: bool = True, binary: bool = True) -> np.ndarray:
        img = image_array.copy()
        quality = ImagePreprocessor._assess_quality(img)

        if quality < 0.6:
            if quality < 0.3:
                img = ImagePreprocessor._unsharp_mask(img, sigma=1.5, amount=2.0)
                img = ImagePreprocessor._sharpen(img, 'strong')
            else:
                img = ImagePreprocessor._unsharp_mask(img, sigma=1.0, amount=1.5)

        if grayscale:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

            if quality < 0.3:
                gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
            elif quality < 0.6:
                gray = cv2.medianBlur(gray, 3)
            else:
                gray = cv2.GaussianBlur(gray, (3, 3), 0)

            if quality < 0.3:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
                gamma = 0.8
                table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
                gray = cv2.LUT(gray, table)
            elif quality < 0.6:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray = clahe.apply(gray)

            if binary:
                if quality < 0.3:
                    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 13, 3)
                elif quality < 0.6:
                    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 13, 3)
                else:
                    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 11, 2)
            return gray
        else:
            if img.ndim == 3:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clip_limit = 3.0 if quality < 0.3 else 2.0
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                l = clahe.apply(l)
                if quality < 0.3:
                    l = cv2.bilateralFilter(l, 5, 50, 50)
                enhanced = cv2.merge([l, a, b])
                return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            return img


class ImageProcessor:
    """图片处理器（水印、签章、预处理、保存的统一入口）"""

    @staticmethod
    def detect_watermark(image_array: np.ndarray) -> dict:
        return WatermarkHandler.detect(image_array)

    @staticmethod
    def remove_watermark(image_array: np.ndarray, watermark_info: dict) -> np.ndarray:
        return WatermarkHandler.remove(image_array, watermark_info)

    @staticmethod
    def detect_stamp(image_array: np.ndarray) -> dict:
        return StampDetector.detect(image_array)

    @staticmethod
    def preprocess_for_ocr(image_array: np.ndarray, remove_watermark_flag: bool = True,
                           grayscale: bool = True, binary: bool = True) -> np.ndarray:
        img = image_array.copy()
        if remove_watermark_flag:
            watermark_info = WatermarkHandler.detect(img)
            if watermark_info['has_watermark']:
                logger.info(f"检测到水印 (类型: {watermark_info['type']}, "
                            f"置信度: {watermark_info['confidence']:.2f})")
                img = WatermarkHandler.remove(img, watermark_info)
        return ImagePreprocessor.preprocess(img, grayscale=grayscale, binary=binary)

    @staticmethod
    def detect_text_regions(image_array: np.ndarray) -> list:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if image_array.ndim == 3 else image_array
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w > 20 and h > 10:
                regions.append((x, y, w, h))
        return regions

    @staticmethod
    def save_image(image_bytes: bytes, output_path: str) -> str:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(image_bytes)
        return output_path

    @staticmethod
    def image_to_cv2_array(image_bytes: bytes) -> Optional[np.ndarray]:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img if img is not None else None


__all__ = [
    "WatermarkHandler",
    "StampDetector",
    "ImagePreprocessor",
    "ImageProcessor",
]
