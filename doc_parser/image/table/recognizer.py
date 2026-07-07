"""
表格识别器
==========
从图像中检测并提取表格结构化数据。

支持两种后端：
1. PaddleTableRecognizer — 基于 PaddleOCR PP-Structure，精度最高，适合扫描件/图片
2. VisualTableRecognizer — 基于几何线条分析，适合清晰表格图片

输出统一的 TableRecognitionResult，可被 PDF/Word 解析器消费。
"""

import logging
import re
from abc import ABC, abstractmethod

import numpy as np

from ai_sag.doc_parser.image.table.models import TableCell, TableRecognitionResult

logger = logging.getLogger(__name__)


class TableRecognizer(ABC):
    """表格识别器抽象基类"""

    @abstractmethod
    def recognize(self, image_array: np.ndarray) -> list[TableRecognitionResult]:
        """
        从图像中识别所有表格

        Args:
            image_array: BGR 格式的 numpy 数组（cv2 默认格式）

        Returns:
            TableRecognitionResult 列表（一张图可能包含多个表格）
        """
        ...

    @staticmethod
    def _clean_cell_text(text: str) -> str:
        """清理单元格文本"""
        if not text:
            return ""
        t = str(text).strip()
        t = t.replace("\n", " ").replace("\t", " ")
        t = re.sub(r'[ ]{2,}', ' ', t)
        t = re.sub(r'([\u4e00-\u9fff]) ([\u4e00-\u9fff])', r'\1\2', t)
        return t.strip()


class PaddleTableRecognizer(TableRecognizer):
    """
    基于 PaddleOCR 的表格识别器。

    自动适配 PaddleOCR 版本:
    - 3.x+: 使用 PPStructureV3 / TableRecognitionPipelineV2
    - 2.x:  使用 PPStructure (旧版)

    管线：表格区域检测 → 结构识别(HTML) → 单元格 OCR
    适用场景：扫描件、照片拍摄的表格、低质量图片中的表格
    精度：高（中文场景最优）
    """

    def __init__(self):
        self._engine = None
        self._api_version = None

    @property
    def engine(self):
        """延迟初始化 PaddleOCR 表格识别引擎（自动适配版本）"""
        if self._engine is None:
            try:
                import paddleocr as pocr
                version = getattr(pocr, '__version__', '0.0')
                major = int(version.split('.')[0]) if version else 0

                if major >= 3:
                    logger.info(f"初始化 PaddleOCR {version} (TableRecognitionPipelineV2)...")
                    try:
                        from paddleocr import TableRecognitionPipelineV2
                        self._engine = TableRecognitionPipelineV2()
                        self._api_version = "v3"
                    except Exception:
                        from paddleocr import PPStructureV3
                        self._engine = PPStructureV3(use_table_recognition=True)
                        self._api_version = "v3_ppstructure"
                else:
                    logger.info(f"初始化 PaddleOCR {version} (PPStructure)...")
                    from paddleocr import PPStructure
                    self._engine = PPStructure(
                        show_log=False,
                        image_orientation=True,
                        structure_version='PP-StructureV2',
                    )
                    self._api_version = "legacy"

                logger.info(f"PaddleOCR 表格引擎初始化完成 (API: {self._api_version})")
            except ImportError:
                logger.error("PaddleOCR 未安装。请安装: pip install paddleocr")
                raise
        return self._engine

    def recognize(self, image_array: np.ndarray) -> list[TableRecognitionResult]:
        results = []

        try:
            _ = self.engine
            logger.info(f"Paddle 调用: api_version={self._api_version}, engine_type={type(self.engine).__name__}")
            raw_results = self._call_engine(image_array)
            logger.info(f"Paddle 返回: type={type(raw_results)}, len={len(raw_results) if isinstance(raw_results, (list, tuple)) else 'N/A'}")
        except Exception as e:
            logger.warning(f"PaddleOCR 表格识别失败: {e}")
            return results

        if self._api_version == "legacy":
            results = self._parse_legacy_results(raw_results)
        elif self._api_version in ("v3", "v3_ppstructure"):
            results = self._parse_v3_results(raw_results)

        return results

    def _call_engine(self, image_array: np.ndarray):
        if hasattr(self.engine, 'predict') and callable(getattr(self.engine, 'predict', None)):
            try:
                return self.engine.predict(image_array)
            except TypeError as e:
                if 'not callable' in str(e) or 'callable' in str(e):
                    logger.debug(f".predict() 不可用，回退到直接调用: {e}")
                else:
                    raise
        if callable(self.engine):
            return self.engine(image_array)
        raise TypeError(
            f"{type(self.engine).__name__} 不支持任何已知的调用方式。"
            f"可用的方法: {[m for m in dir(self.engine) if not m.startswith('_') and callable(getattr(self.engine, m, None))]}"
        )

    def _parse_legacy_results(self, raw_results: list) -> list[TableRecognitionResult]:
        """解析旧版 PP-Structure 输出格式"""
        results = []
        for item in raw_results:
            if item.get('type') != 'table':
                continue
            html_table = ""
            res_list = item.get('res', [])
            if isinstance(res_list, str):
                html_table = res_list
            elif isinstance(res_list, list) and res_list:
                for r in res_list:
                    if isinstance(r, dict) and 'html' in r:
                        html_table = r['html']; break
                    elif isinstance(r, str):
                        html_table = r; break

            cells, n_rows, n_cols = self._parse_html_to_cells(html_table)
            result = TableRecognitionResult(
                source="paddle_ppstructure",
                cells=cells, total_rows=n_rows, total_cols=n_cols,
                html_table=html_table,
                confidence=self._estimate_confidence(cells),
                bbox=tuple(item.get('bbox', ())),
            )
            result.raw_grid = result.grid
            results.append(result)
            logger.info(f"Paddle 表格识别: {n_rows}行 × {n_cols}列, "
                        f"{len(cells)} 个单元格, 置信度 {result.confidence:.2f}")
        return results

    def _parse_v3_results(self, raw_results: list) -> list[TableRecognitionResult]:
        """
        解析 PaddleOCR 3.x 输出格式 (TableRecognitionPipelineV2)。
        """
        results = []
        for item in raw_results:
            if not isinstance(item, dict):
                logger.debug(f"跳过非 dict 项: {type(item).__name__}")
                continue

            table_list = item.get('table_res_list', [])
            if not table_list:
                logger.info("  [调试] table_res_list 为空或不存在，打印所有 key:")
                for k, v in item.items():
                    v_str = str(v)[:150] if isinstance(v, (str, int, float)) else f"<{type(v).__name__}>"
                    if isinstance(v, (list, dict)):
                        v_str = f"<{type(v).__name__} len={len(v)}>"
                    if isinstance(v, dict):
                        v_str += f" keys={list(v.keys())[:10]}"
                    logger.info(f"    {k}: {v_str}")
                continue

            for tidx, table_item in enumerate(table_list):
                if not isinstance(table_item, dict):
                    logger.debug(f"table_res_list[{tidx}] 不是 dict: {type(table_item).__name__}")
                    continue

                logger.info(f"  [调试] table_res_list[{tidx}] keys={list(table_item.keys())}")
                for tk, tv in table_item.items():
                    if isinstance(tv, str) and len(tv) > 100:
                        logger.info(f"    {tk}: <str len={len(tv)}> {tv[:120]}...")
                    elif isinstance(tv, (list, dict)):
                        logger.info(f"    {tk}: <{type(tv).__name__} len={len(tv)}>")
                        if isinstance(tv, dict):
                            logger.info(f"         keys={list(tv.keys())[:15]}")
                        elif isinstance(tv, list) and len(tv) > 0 and isinstance(tv[0], dict):
                            logger.info(f"         [0] keys={list(tv[0].keys())[:10]}")
                            logger.info(f"         [0] str={str(tv[0])[:200]}")
                    else:
                        logger.info(f"    {tk}: {tv}")

                html_table = ""

                for html_key in ('pred_html', 'html', 'table_html', 'res_html'):
                    val = table_item.get(html_key)
                    if isinstance(val, str) and '<table' in val.lower():
                        html_table = val
                        break

                if not html_table:
                    res_val = table_item.get('res')
                    if isinstance(res_val, str) and '<table' in res_val.lower():
                        html_table = res_val
                    elif isinstance(res_val, dict):
                        html_table = res_val.get('html', res_val.get('table_html', ''))

                if not html_table:
                    cells_data = table_item.get('cells') or table_item.get('cell_boxes') or []
                    if cells_data:
                        logger.info(f"  使用 cells 数据构建 HTML，共 {len(cells_data)} 个单元格")
                        html_table = self._cells_to_html(cells_data)

                cells, n_rows, n_cols = self._parse_html_to_cells(html_table)
                bbox = table_item.get('bbox', ())

                cell_box_list = table_item.get('cell_box_list') or []
                if not bbox and cell_box_list:
                    bbox = self._calc_bbox_from_cells(cell_box_list)

                table_ocr_pred = table_item.get('table_ocr_pred') or {}
                if cell_box_list and table_ocr_pred and cells:
                    cells = self._refill_cells_with_ocr(
                        cells, n_rows, n_cols, cell_box_list, table_ocr_pred
                    )

                result = TableRecognitionResult(
                    source=f"paddle_v3_{self._api_version}",
                    cells=cells, total_rows=n_rows, total_cols=n_cols,
                    html_table=html_table,
                    confidence=self._estimate_confidence(cells),
                    bbox=tuple(bbox) if bbox else (),
                )
                result.raw_grid = result.grid
                results.append(result)
                logger.info(
                    f"Paddle V3 表格识别[{tidx}]: {n_rows}行 × {n_cols}列, "
                    f"{len(cells)} 个单元格, 置信度 {result.confidence:.2f}"
                )

        return results

    @staticmethod
    def _cells_to_html(cells_data) -> str:
        """将单元格数据转为 HTML（备用方案）"""
        if not cells_data:
            return ""
        lines = ["<table><tbody>"]
        for cell in cells_data:
            if isinstance(cell, dict):
                text = cell.get('text', cell.get('content', ''))
                lines.append(f"<tr><td>{text}</td></tr>")
            else:
                lines.append(f"<tr><td>{str(cell)}</td></tr>")
        lines.append("</tbody></table>")
        return "\n".join(lines)

    @staticmethod
    def _parse_html_to_cells(html_table: str) -> tuple[list[TableCell], int, int]:
        """
        将 HTML 表格解析为 TableCell 列表。

        处理 colspan/rowspan 合并属性。
        """
        cells = []
        if not html_table or '<table' not in html_table.lower():
            return cells, 0, 0

        tr_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
        td_pattern = re.compile(
            r'<t[dh]([^>]*)>(.*?)</t[dh]>',
            re.DOTALL | re.IGNORECASE
        )

        rows_html = tr_pattern.findall(html_table)
        if not rows_html:
            return cells, 0, 0

        max_cols = 0
        occupied = set()
        all_cells_info = []

        for ri, row_html in enumerate(rows_html):
            td_matches = td_pattern.findall(row_html)
            ci = 0
            for attrs_str, text_content in td_matches:
                while (ri, ci) in occupied:
                    ci += 1

                colspan_match = re.search(r'colspan\s*=\s*["\']?(\d+)', attrs_str, re.I)
                rowspan_match = re.search(r'rowspan\s*=\s*["\']?(\d+)', attrs_str, re.I)
                colspan = int(colspan_match.group(1)) if colspan_match else 1
                rowspan = int(rowspan_match.group(1)) if rowspan_match else 1

                text_clean = re.sub(r'<[^>]+>', '', text_content).strip()

                all_cells_info.append((ri, ci, text_clean, colspan, rowspan))

                for dr in range(rowspan):
                    for dc in range(colspan):
                        occupied.add((ri + dr, ci + dc))

                ci += colspan
                max_cols = max(max_cols, ci)

        n_rows = len(rows_html)
        n_cols = max_cols if max_cols > 0 else 1

        for ri, ci, text, colspan, rowspan in all_cells_info:
            cells.append(TableCell(
                row=ri,
                col=ci,
                text=TableRecognizer._clean_cell_text(text),
                row_span=rowspan,
                col_span=colspan,
            ))

        return cells, n_rows, n_cols

    @staticmethod
    def _estimate_confidence(cells: list[TableCell]) -> float:
        """根据非空单元格比例估算整体置信度"""
        if not cells:
            return 0.0
        non_empty = sum(1 for c in cells if c.text.strip())
        return non_empty / len(cells)

    @staticmethod
    def _calc_bbox_from_cells(cell_box_list: list) -> tuple:
        """从单元格坐标列表推算表格整体 bbox"""
        try:
            xs, ys = [], []
            for box in cell_box_list:
                if box is None:
                    continue
                arr = np.array(box, dtype=float).reshape(-1, 2)
                xs.extend(arr[:, 0].tolist())
                ys.extend(arr[:, 1].tolist())
            if not xs:
                return ()
            return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
        except Exception:
            return ()

    @staticmethod
    def _refill_cells_with_ocr(
        cells: list[TableCell],
        n_rows: int,
        n_cols: int,
        cell_box_list: list,
        table_ocr_pred: dict,
    ) -> list[TableCell]:
        """用 cell_box_list 坐标 + table_ocr_pred OCR 文字重新填充表格"""
        try:
            from ai_sag.doc_parser.image.config import OCR_CONFIDENCE_THRESHOLD

            cell_bboxes = []
            for box in cell_box_list:
                if box is None:
                    cell_bboxes.append(None)
                    continue
                arr = np.array(box, dtype=float).reshape(-1, 2)
                cell_bboxes.append((
                    arr[:, 0].min(), arr[:, 1].min(),
                    arr[:, 0].max(), arr[:, 1].max(),
                ))

            if len(cell_bboxes) != len(cells):
                logger.debug(
                    f"cell_box_list({len(cell_bboxes)}) 与 cells({len(cells)}) 数量不一致，跳过重填"
                )
                return cells

            coord_to_cell: list[tuple[tuple, int, int]] = []
            for cell, bbox in zip(cells, cell_bboxes):
                if bbox is None:
                    continue
                coord_to_cell.append((bbox, cell.row, cell.col))

            rec_polys = table_ocr_pred.get('rec_polys') or table_ocr_pred.get('dt_polys') or []
            rec_texts = table_ocr_pred.get('rec_texts', [])
            rec_scores = table_ocr_pred.get('rec_scores', [])

            new_grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
            for cell in cells:
                if 0 <= cell.row < n_rows and 0 <= cell.col < n_cols:
                    new_grid[cell.row][cell.col] = ""

            n = min(len(rec_polys), len(rec_texts), len(rec_scores))
            matched = 0
            for i in range(n):
                if not rec_texts[i]:
                    continue
                try:
                    score = float(rec_scores[i])
                except (TypeError, ValueError):
                    score = 1.0
                if score <= OCR_CONFIDENCE_THRESHOLD:
                    continue

                poly = np.array(rec_polys[i], dtype=float).reshape(-1, 2)
                cx = float(poly[:, 0].mean())
                cy = float(poly[:, 1].mean())

                best_row, best_col = -1, -1
                for bbox, row, col in coord_to_cell:
                    if bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]:
                        best_row, best_col = row, col
                        break

                if best_row >= 0 and best_col >= 0:
                    text = TableRecognizer._clean_cell_text(str(rec_texts[i]))
                    if text:
                        if new_grid[best_row][best_col]:
                            new_grid[best_row][best_col] += " " + text
                        else:
                            new_grid[best_row][best_col] = text
                        matched += 1

            logger.info(
                f"表格内容重填: OCR文字 {n} 个, 匹配到单元格 {matched} 个"
            )

            new_cells = []
            for ri in range(n_rows):
                for ci in range(n_cols):
                    text = new_grid[ri][ci]
                    if text:
                        new_cells.append(TableCell(
                            row=ri, col=ci, text=text,
                            row_span=1, col_span=1,
                        ))

            return new_cells if new_cells else cells

        except Exception as e:
            logger.warning(f"表格内容重填失败，保留原始结果: {e}")
            return cells


class VisualTableRecognizer(TableRecognizer):
    """
    基于视觉几何分析的表格识别器。

    管线：
    1. 检测水平和垂直线条（Hough 变换 / 形态学）
    2. 构建网格交叉点
    3. 分割单元格区域
    4. 对每个单元格调用 OCR 引擎识别文字

    适用场景：清晰、线条规整的电子表格截图/打印件
    精度：中等（依赖线条质量）
    """

    def __init__(self, ocr_engine=None):
        self.ocr_engine = ocr_engine

    def recognize(self, image_array: np.ndarray) -> list[TableRecognitionResult]:
        results = []

        import cv2

        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if image_array.ndim == 3 else image_array.copy()
        h, w = gray.shape

        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, -5
        )

        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 20, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 20))

        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        table_mask = cv2.bitwise_or(h_lines, v_lines)

        line_density = np.sum(table_mask > 0) / (h * w)
        if line_density < 0.02:
            logger.debug("视觉表格检测：线条密度过低，未发现表格")
            return results

        h_proj = np.sum(h_lines, axis=1)
        v_proj = np.sum(v_lines, axis=0)

        h_threshold = np.max(h_proj) * 0.3 if np.max(h_proj) > 0 else 0
        v_threshold = np.max(v_proj) * 0.3 if np.max(v_proj) > 0 else 0

        h_positions = np.where(h_proj > h_threshold)[0]
        v_positions = np.where(v_proj > v_threshold)[0]

        h_lines_pos = self._filter_line_positions(h_positions, min_gap=h // 30)
        v_lines_pos = self._filter_line_positions(v_positions, min_gap=w // 30)

        if len(h_lines_pos) < 3 or len(v_lines_pos) < 3:
            logger.debug(f"视觉表格检测：线条不足 ({len(h_lines_pos)} 水平, {len(v_lines_pos)} 垂直)")
            return results

        n_rows = len(h_lines_pos) - 1
        n_cols = len(v_lines_pos) - 1
        cells = []

        for ri in range(n_rows):
            y1 = h_lines_pos[ri]
            y2 = h_lines_pos[ri + 1]
            for ci in range(n_cols):
                x1 = v_lines_pos[ci]
                x2 = v_lines_pos[ci + 1]

                cell_img = image_array[y1:y2, x1:x2]
                text = ""
                conf = 0.0
                has_content = False

                if self.ocr_engine is not None and cell_img.size > 0:
                    try:
                        ocr_texts = self.ocr_engine.recognize(cell_img, preprocess=True)
                        text = " ".join(ocr_texts).strip()
                        conf = min(len(text) / 5.0, 1.0) if text else 0.0
                        has_content = bool(text)
                    except Exception as e:
                        logger.warning(f"单元格 ({ri},{ci}) OCR 失败: {e}")
                elif cell_img.size > 0:
                    cell_gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY) if cell_img.ndim == 3 else cell_img
                    has_content = np.mean(cell_gray) < 240

                if text or has_content:
                    cells.append(TableCell(
                        row=ri, col=ci,
                        text=TableRecognizer._clean_cell_text(text),
                        row_span=1, col_span=1,
                        confidence=conf,
                    ))

        result = TableRecognitionResult(
            source="visual_lines",
            cells=cells,
            total_rows=n_rows,
            total_cols=n_cols,
            confidence=self._estimate_confidence(cells),
            bbox=(0, 0, w, h),
        )
        result.raw_grid = result.grid
        results.append(result)

        logger.info(
            f"视觉表格识别: {n_rows}行 × {n_cols}列, "
            f"{len(cells)} 个非空单元格"
        )

        return results

    @staticmethod
    def _estimate_confidence(cells: list) -> float:
        """根据非空单元格比例估算置信度"""
        if not cells:
            return 0.0
        non_empty = sum(1 for c in cells if c.text.strip())
        return non_empty / len(cells)

    @staticmethod
    def _filter_line_positions(positions: np.ndarray, min_gap: int = 10) -> list[int]:
        """过滤线条位置：去除过于密集的候选，保留局部峰值"""
        if len(positions) == 0:
            return []

        filtered = [int(positions[0])]
        for p in positions[1:]:
            if p - filtered[-1] >= min_gap:
                filtered.append(int(p))
        return filtered


class PositionTableRecognizer(TableRecognizer):
    """
    基于 OCR 文字块位置聚类的表格识别器。

    不依赖 PaddleOCR 的表格结构识别（PP-Structure），而是：
    1. 对图片运行 OCR 获取带位置的文字块（或接收外部传入的块）
    2. 按 y 坐标聚类为行
    3. 分析行间水平间隙模式，识别表格区域
    4. 在表格区域内按 x 坐标检测列边界
    5. 将文字块分配到 (行, 列) 单元格

    优势：
    - 避免 PaddleOCR 结构识别的单元格错位问题
    - 单元格文字直接来自 OCR，无需二次匹配
    - 速度快（无额外的结构识别模型推理）
    """

    def __init__(self, ocr_engine=None):
        self.ocr_engine = ocr_engine

    def recognize(
        self,
        image_array: np.ndarray,
        ocr_blocks: list = None,
    ) -> list[TableRecognitionResult]:
        """从图像或预计算的 OCR 块中识别表格

        Args:
            image_array: BGR 格式 numpy 数组
            ocr_blocks: 可选，预计算的 OCRTextBlock 列表。
                        传入时跳过 OCR，直接使用这些块做表格检测。

        Returns:
            TableRecognitionResult 列表
        """
        from ai_sag.doc_parser.image.table.layout import detect_tables_from_blocks

        blocks = ocr_blocks

        # 如果没有预计算的块，运行 OCR
        if blocks is None:
            blocks = self._run_ocr(image_array)

        if not blocks:
            logger.info("[PositionTableRecognizer] 无 OCR 文字块")
            return []

        tables, regions = detect_tables_from_blocks(blocks)

        if tables:
            logger.info(
                f"[PositionTableRecognizer] 检测到 {len(tables)} 个表格, "
                f"区域: {regions}"
            )
        else:
            logger.info("[PositionTableRecognizer] 未检测到表格")

        return tables

    def _run_ocr(self, image_array: np.ndarray) -> list:
        """运行 OCR 获取带位置的文字块"""
        if self.ocr_engine is not None:
            return self.ocr_engine.recognize_with_positions(
                image_array, preprocess=True,
            )

        # 没有 OCR 引擎，尝试默认获取
        try:
            from ai_sag.doc_parser.image.ocr import get_ocr_engine
            engine = get_ocr_engine("rapidocr")
            return engine.recognize_with_positions(
                image_array, preprocess=True,
            )
        except Exception as e:
            logger.warning(f"OCR 失败: {e}")
            return []


_TABLE_RECOGNIZER_REGISTRY: dict = {
    "paddle": PaddleTableRecognizer,
    "visual": VisualTableRecognizer,
    "position": PositionTableRecognizer,
}
_recognizer_cache: dict = {}


def get_table_recognizer(backend: str = "paddle", **kwargs) -> TableRecognizer:
    """
    获取表格识别器单例

    Args:
        backend: "paddle" (PP-Structure, 高精度) 或 "visual" (线条分析, 轻量)
                 或 "position" (位置聚类, 无需结构识别模型)
        **kwargs: 传递给识别器的额外参数（如 ocr_engine）

    Returns:
        TableRecognizer 实例
    """
    backend = backend.lower()
    if backend not in _TABLE_RECOGNIZER_REGISTRY:
        raise ValueError(
            f"不支持的表格识别后端: {backend}，可选: {list(_TABLE_RECOGNIZER_REGISTRY.keys())}"
        )
    cache_key = backend + str(sorted(kwargs.items())) if kwargs else backend
    if cache_key not in _recognizer_cache:
        _recognizer_cache[cache_key] = _TABLE_RECOGNIZER_REGISTRY[backend](**kwargs)
    return _recognizer_cache[cache_key]


def recognize_tables_in_image(image_array: np.ndarray,
                               backend: str = "paddle",
                               ocr_blocks: list = None,
                               **kwargs) -> list[TableRecognitionResult]:
    """
    便捷函数：从图像中识别所有表格

    Args:
        image_array: BGR 格式 numpy 数组
        backend: "paddle"、"visual" 或 "position"
        ocr_blocks: 可选，预计算的 OCRTextBlock 列表（仅 "position" 后端使用）

    Returns:
        TableRecognitionResult 列表
    """
    recognizer = get_table_recognizer(backend, **kwargs)
    if backend == "position" and ocr_blocks is not None:
        return recognizer.recognize(image_array, ocr_blocks=ocr_blocks)
    return recognizer.recognize(image_array)
