"""
PDF 解析器主类（V2）
====================
支持文本/混合/扫描版 PDF 解析，支持 PaddleOCR 或 RapidOCR 后端。
V2 新增：PyMuPDF 表格检测 + 合并单元格处理。
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional, Union

import fitz  # pymupdf
import numpy as np
import cv2
import pymupdf4llm

from .models import ImageInfo, PageContent, PDFResult, PDFTable
from .detector import PDFTypeDetector
from .image_processor import ImageProcessor
from .formatter import save_markdown, save_text
from ...image.ocr import OCRBackend, BaseOCREngine, get_ocr_engine
from .config import (
    OCR_PAGE_DPI, setup_logging,
    ENABLE_TABLE_DETECTION, TABLE_MIN_ROWS, TABLE_MIN_COLS,
)
from .table_handler import (
    table_to_markdown,
    extract_tables_from_struct_tree,
    _looks_like_toc,
    is_tagged_pdf,
)

# 初始化日志（模块级别，确保整个包可用）
setup_logging()

logger = logging.getLogger(__name__)


# ============================================================
# PDF 解析器
# ============================================================
class PDFParser:
    """PDF 解析器主类（V2：含表格检测）"""

    def __init__(self, output_dir: Optional[str] = None, extract_images: bool = True,
                 ocr_backend: OCRBackend = "rapidocr",
                 ocr_images: bool = True, ocr_image_min_area: int = 10000,
                 detect_tables: bool = True):
        """
        Args:
            output_dir: 输出目录，传值则保存解析结果（md/txt/图片）到磁盘，
                        传 None 则不写磁盘，仅返回内存中的解析结果
            extract_images: 是否提取并处理图片（水印检测、签章检测、OCR）
            ocr_backend: OCR 后端引擎
            ocr_images: 是否对 text/mixed 类型 PDF 中提取的图片执行 OCR
            ocr_image_min_area: 图片最小面积（像素²），低于此值的图片不做 OCR
            detect_tables: 是否启用表格检测
        """
        self.output_dir = output_dir
        self.extract_images = extract_images
        self.ocr_backend = ocr_backend
        self.ocr_images = ocr_images
        self.ocr_image_min_area = ocr_image_min_area
        self.detect_tables = detect_tables and ENABLE_TABLE_DETECTION
        self._ocr: "BaseOCREngine | None" = None

    @property
    def ocr(self) -> BaseOCREngine:
        """延迟获取 OCR 引擎"""
        if self._ocr is None:
            self._ocr = get_ocr_engine(self.ocr_backend)
        return self._ocr

    # ---------- 公开方法 ----------
    def parse(self, source: Union[str, bytes], file_name: Optional[str] = None) -> PDFResult:
        """
        解析单个 PDF 文档

        Args:
            source: PDF 数据源，支持本地路径 / 远程 URL / S3 URI / 字节流
            file_name: 文件名（用于输出文件命名），仅 bytes 输入时需要

        Returns:
            PDFResult 解析结果
        """
        if isinstance(source, bytes):
            pdf_bytes = source
            display_name = file_name or "input.pdf"
        elif isinstance(source, str):
            if source.startswith(("http://", "https://")):
                logger.info(f"下载远程文件: {source}")
                pdf_bytes = self._fetch_url(source)
                display_name = file_name or self._extract_filename_from_url(source)
            elif source.startswith("s3://"):
                logger.info(f"下载 S3 文件: {source}")
                pdf_bytes = self._fetch_s3(source)
                display_name = file_name or self._extract_filename_from_url(source)
            else:
                if not os.path.exists(source):
                    raise FileNotFoundError(f"PDF 文件不存在: {source}")
                with open(source, "rb") as f:
                    pdf_bytes = f.read()
                display_name = file_name or os.path.basename(source)
        else:
            raise TypeError(f"不支持的 source 类型: {type(source)}，期望 str 或 bytes")

        logger.info(f"开始解析: {display_name} ({len(pdf_bytes)} 字节)")

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            pdf_type = PDFTypeDetector.detect(doc)
            logger.info(f"检测到 PDF 类型: {pdf_type}")

            result = PDFResult(
                file_path=display_name if isinstance(source, str) else "",
                file_name=display_name,
                total_pages=len(doc),
                pdf_type=pdf_type,
                metadata=dict(doc.metadata) if doc.metadata else {},
                ocr_backend=self.ocr_backend,
            )

            # 表格提取策略：find_tables() 视觉检测优先，结构树回退
            # 第一步：全局扫描所有页的 find_tables() 结果，确定哪些页已通过视觉方式找到表格
            visual_table_pages: set = set()  # 已通过 find_tables 找到表格的页码集合
            if self.detect_tables:
                try:
                    for pg_idx in range(len(doc)):
                        finder = doc[pg_idx].find_tables()
                        if finder.tables:
                            visual_table_pages.add(pg_idx + 1)  # 1-based 页码
                except Exception as e:
                    logger.warning(f"find_tables() 全局预扫描失败: {e}")

            # 第二步：仅对 find_tables() 未覆盖的页尝试结构树提取（回退）
            struct_tree_tables: dict = {}
            if self.detect_tables and visual_table_pages:
                pages_need_fallback = set(range(1, len(doc) + 1)) - visual_table_pages
                if pages_need_fallback and is_tagged_pdf(doc):
                    try:
                        all_struct = extract_tables_from_struct_tree(doc, fuse_with_visual_flag=False)
                        # 只保留需要回退的页的结构树结果
                        struct_tree_tables = {
                            pn: tbls for pn, tbls in all_struct.items()
                            if pn in pages_need_fallback
                        }
                        if struct_tree_tables:
                            logger.info(
                                f"结构树回退: 对 {len(struct_tree_tables)} 个页面补充了 "
                                f"{sum(len(t) for t in struct_tree_tables.values())} 个表格"
                            )
                    except Exception as e:
                        logger.warning(f"结构树表格提取失败: {e}")

            if pdf_type == "image_only":
                self._parse_image_only(doc, result)
            else:
                self._parse_text_or_mixed(doc, result, struct_tree_tables=struct_tree_tables)

            # 汇总所有表格
            for page in result.pages:
                result.tables.extend(page.tables)

            # 生成 Markdown
            result.markdown_text = self._to_markdown(doc, pdf_type, result.pages)
        finally:
            doc.close()

        result.full_text = "\n\n".join(
            p.text for p in result.pages if p.text.strip()
        )

        logger.info(
            f"解析完成: {len(result.pages)} 页, "
            f"{len(result.tables)} 个表格, "
            f"文本长度 {len(result.full_text)} 字符"
        )
        return result

    # ---------- 远程文件获取 ----------
    @staticmethod
    def _fetch_url(url: str) -> bytes:
        import urllib.request
        response = urllib.request.urlopen(url)
        return response.read()

    @staticmethod
    def _fetch_s3(s3_uri: str) -> bytes:
        try:
            import boto3
        except ImportError:
            raise ImportError("S3 支持需要 boto3，请安装: pip install boto3")
        parts = s3_uri.replace("s3://", "").split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"S3 URI 格式错误: {s3_uri}")
        bucket, key = parts
        s3_client = boto3.client("s3")
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()

    @staticmethod
    def _extract_filename_from_url(url: str) -> str:
        path = url.split("?")[0].split("#")[0]
        name = os.path.basename(path)
        if not name or "." not in name:
            name = "remote_file.pdf"
        return name

    def parse_directory(self, dir_path: str, recursive: bool = False) -> "list[PDFResult]":
        """批量解析目录下的所有 PDF"""
        results: list[PDFResult] = []
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(Path(dir_path).glob(pattern))
        logger.info(f"找到 {len(pdf_files)} 个 PDF 文件")

        for pdf_file in pdf_files:
            try:
                result = self.parse(str(pdf_file))
                results.append(result)
                if self.output_dir:
                    stem = Path(pdf_file).stem
                    save_markdown(result, os.path.join(self.output_dir, f"{stem}.md"))
                    save_text(result, os.path.join(self.output_dir, f"{stem}.txt"))
            except Exception as e:
                logger.error(f"解析失败 {pdf_file}: {e}")

        return results

    # ---------- 表格检测 ----------
    def _detect_tables_on_page(self, page: fitz.Page, page_num: int,
                                global_table_idx: int,
                                pre_extracted: list = None) -> tuple[list[PDFTable], bool]:
        """检测页面上的表格并转换为结构化结果

        新策略（V2 调整）：
        1. 优先使用 PyMuPDF 视觉检测 (page.find_tables()) — 兼容所有 PDF
        2. 如果视觉检测无结果，回退到 Tagged PDF StructTreeRoot 预提取

        Args:
            page: PyMuPDF 页面对象
            page_num: 页码（1-based）
            global_table_idx: 文档级表格计数器（会被更新）
            pre_extracted: 从结构树预提取的 [StructTableResult, ...] 列表（回退用）

        Returns:
            (PDFTable列表, 是否使用了视觉检测结果)
        """
        tables: list[PDFTable] = []

        # ====== 路径1：PyMuPDF 视觉检测（优先） ======
        try:
            finder = page.find_tables()
        except Exception as e:
            logger.warning(f"页 {page_num} find_tables() 失败: {e}")
            finder = None

        if finder and finder.tables:
            for tbl in finder.tables:
                extracted = tbl.extract()
                if not extracted:
                    continue
                n_rows = len(extracted)
                n_cols = max(len(row) for row in extracted) if extracted else 0
                if n_rows < TABLE_MIN_ROWS or n_cols < TABLE_MIN_COLS:
                    continue

                # 过滤目录型误检
                if _looks_like_toc(extracted):
                    logger.info(f"页 {page_num}: 过滤掉疑似目录的视觉检测表格 ({n_rows}×{n_cols})")
                    continue

                global_table_idx += 1
                result = table_to_markdown(tbl, table_index=global_table_idx - 1)

                pdf_table = PDFTable(
                    page_num=page_num,
                    table_index=global_table_idx,
                    row_count=result["row_count"],
                    col_count=result["col_count"],
                    markdown_text=result["markdown"],
                    form_fields=result["form_fields"],
                    signing_info=result["signing_info"],
                    title=result["title"],
                    bbox=result.get("bbox", ()),
                    source="visual",
                )
                tables.append(pdf_table)
                logger.info(
                    f"页 {page_num} 视觉检测表格 {global_table_idx}: "
                    f"{pdf_table.row_count}行 × {pdf_table.col_count}列"
                )

            if tables:
                return tables, True  # 视觉检测成功，直接返回

        # ====== 路径2：结构树回退 ======
        if pre_extracted:
            for tbl_result in pre_extracted:
                extracted = tbl_result.extract()
                if not extracted:
                    continue
                n_rows = len(extracted)
                n_cols = max(len(row) for row in extracted) if extracted else 0
                if n_rows < TABLE_MIN_ROWS or n_cols < TABLE_MIN_COLS:
                    continue

                # 过滤目录型误检
                if _looks_like_toc(extracted):
                    logger.info(f"页 {page_num}: 过滤掉疑似目录的结构树表格 ({n_rows}×{n_cols})")
                    continue

                global_table_idx += 1
                result = table_to_markdown(tbl_result, table_index=global_table_idx - 1)

                pdf_table = PDFTable(
                    page_num=page_num,
                    table_index=global_table_idx,
                    row_count=result["row_count"],
                    col_count=result["col_count"],
                    markdown_text=result["markdown"],
                    form_fields=result["form_fields"],
                    signing_info=result["signing_info"],
                    title=result["title"],
                    bbox=result.get("bbox", ()),
                    source="struct_tree",
                )
                tables.append(pdf_table)
                logger.info(
                    f"页 {page_num} 结构树表格 {global_table_idx}: "
                    f"{pdf_table.row_count}行 × {pdf_table.col_count}列"
                )

        return tables, False

    # ---------- 内部解析方法 ----------
    def _parse_text_or_mixed(self, doc: fitz.Document, result: PDFResult,
                              struct_tree_tables: dict = None) -> None:
        global_table_idx = 0
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            has_images = bool(page.get_images(full=True))

            # 表格检测：find_tables() 视觉优先，结构树回退
            page_tables = []
            if self.detect_tables:
                pre_extracted = (struct_tree_tables or {}).get(page_num)
                page_tables, _ = self._detect_tables_on_page(
                    page, page_num, global_table_idx,
                    pre_extracted=pre_extracted,
                )
                global_table_idx += len(page_tables)

            # 纯文本页：跳过图片提取和 OCR
            if not has_images or not self.extract_images:
                result.pages.append(PageContent(
                    page_num=page_num,
                    text=text.strip(),
                    images=[],
                    tables=page_tables,
                    is_image_only=False,
                ))
                continue

            images = self._extract_images_from_page(page, page_num, result.file_name,
                                                     page_text=text)

            image_ocr_parts = []
            for img in images:
                content = img.ocr_text.strip()
                if content:
                    image_ocr_parts.append(content)
            if image_ocr_parts:
                combined_text = text.strip() + "\n\n" + "\n\n".join(image_ocr_parts) \
                    if text.strip() else "\n\n".join(image_ocr_parts)
            else:
                combined_text = text.strip()

            result.pages.append(PageContent(
                page_num=page_num,
                text=combined_text,
                images=images,
                tables=page_tables,
                is_image_only=(len(text.strip()) == 0 and len(images) > 0),
            ))

    def _parse_image_only(self, doc: fitz.Document, result: PDFResult) -> None:
        """扫描版 PDF 解析：整页渲染 → OCR 文字识别

        图片表格识别已关闭（PaddleOCR 结构识别错位严重），表格区域的文字
        会被当作普通文本行 OCR 出来，不带表格结构，但文字内容不丢失。
        """
        logger.info("使用 OCR 策略解析扫描版 PDF")

        for page_num, page in enumerate(doc, start=1):
            # 整页 OCR（扫描版 PDF 无文字层，必须 OCR）
            text = self._ocr_page(page, page_num)
            logger.info(f"[image_only] 页 {page_num} OCR 完成: {len(text)} 字符")

            # 提取嵌入图片（水印/签章检测、图片保存）
            # image_only 整页 OCR 已覆盖图片内容，do_ocr=False 避免重复 OCR
            images = self._extract_images_from_page(page, page_num, result.file_name,
                                                     page_text="", do_ocr=False) \
                if self.extract_images else []

            # 聚合嵌入图片的水印/签章检测结果
            page_has_watermark = any(img.has_watermark for img in images)
            page_has_stamp = any(img.has_stamp for img in images)

            # 打印回填到 PDF 的最终文本
            logger.info(
                f"[image_only] 页 {page_num} 回填文本预览 "
                f"({len(text)} 字符):\n"
                f"{text[:1000]}"
                f"{'...' if len(text) > 1000 else ''}"
            )

            result.pages.append(PageContent(
                page_num=page_num,
                text=text,
                images=images,
                tables=[],
                is_image_only=True,
                has_watermark=page_has_watermark,
                has_stamp=page_has_stamp,
            ))

    # ---------- OCR ----------
    def _ocr_page(self, page: fitz.Page, page_num: int) -> str:
        """对单页进行 OCR 识别（整页渲染 → OCR）

        使用带位置信息的 OCR 结果，按行合并文本框，避免表格单元格被
        错误地拆分成多行（每个单元格独占一行）。
        """
        mat = fitz.Matrix(OCR_PAGE_DPI / 72, OCR_PAGE_DPI / 72)
        pix = page.get_pixmap(matrix=mat)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )

        try:
            blocks = self.ocr.recognize_with_positions(img_array, preprocess=True)
            ocr_text = self._merge_blocks_to_lines(blocks)
            if ocr_text:
                logger.info(f"页面 {page_num} {self.ocr_backend} 识别成功: {len(ocr_text)} 字符")
                return ocr_text
        except Exception as e:
            logger.warning(f"{self.ocr_backend} 识别失败: {e}")

    @staticmethod
    def _merge_blocks_to_lines(blocks: list, y_ratio: float = 0.5) -> str:
        """将带位置的 OCR 文本块按行合并

        按 y 中心坐标聚类同一行的文本框，同行内按 x 坐标排序后用空格连接。
        这样表格的表头和每行数据会被合并到同一行，而非每个单元格独占一行。

        Args:
            blocks: OCRTextBlock 列表
            y_ratio: 行合并阈值系数（相对于平均行高）

        Returns:
            合并后的文本，每行一个字符串，用换行符连接
        """
        if not blocks:
            return ""

        heights = [b.y1 - b.y0 for b in blocks if b.y1 > b.y0]
        avg_height = sum(heights) / len(heights) if heights else 20
        y_threshold = avg_height * y_ratio

        sorted_blocks = sorted(blocks, key=lambda b: (b.cy, b.cx))

        lines: list[list] = []
        current_line = [sorted_blocks[0]]

        for b in sorted_blocks[1:]:
            if abs(b.cy - current_line[-1].cy) <= y_threshold:
                current_line.append(b)
            else:
                lines.append(current_line)
                current_line = [b]
        if current_line:
            lines.append(current_line)

        result_lines = []
        for line in lines:
            line.sort(key=lambda x: x.cx)
            result_lines.append(" ".join(item.text for item in line))

        return "\n".join(result_lines)

    # ---------- 图片提取 ----------
    def _extract_images_from_page(self, page: fitz.Page, page_num: int,
                                  file_name: str, page_text: str = "",
                                  do_ocr: Optional[bool] = None) -> list[ImageInfo]:
        """提取页面中的所有图片并做水印/签章检测和 OCR

        Args:
            do_ocr: 是否对图片做 OCR。None 时用 self.ocr_images；
                    image_only 分支应传 False（整页 OCR 已覆盖图片内容）。
        """
        images: list[ImageInfo] = []
        stem = Path(file_name).stem
        for img_index, img_info in enumerate(page.get_images(full=True), start=1):
            xref = img_info[0]
            try:
                base_image = page.parent.extract_image(xref)
                if not base_image:
                    continue

                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                width, height = base_image["width"], base_image["height"]
                bbox = page.get_image_bbox(img_info)
                bbox_tuple = (bbox.x0, bbox.y0, bbox.x1, bbox.y1) if bbox else (0, 0, width, height)

                img_info_obj = ImageInfo(
                    page_num=page_num, image_index=img_index,
                    bbox=bbox_tuple, width=width, height=height,
                )

                img_array = ImageProcessor.image_to_cv2_array(image_bytes)
                if img_array is not None:
                    watermark_result = ImageProcessor.detect_watermark(img_array)
                    img_info_obj.has_watermark = watermark_result['has_watermark']
                    img_info_obj.has_stamp = ImageProcessor.detect_stamp(img_array)['has_stamp']

                    if (watermark_result['has_watermark']
                            and watermark_result['type'] == 'background'
                            and page_text.strip()):
                        logger.info(f"页 {page_num} 图片 {img_index}: 背景水印图，跳过 OCR")
                        if self.output_dir:
                            img_filename = f"{stem}_page{page_num}_img{img_index}.{image_ext}"
                            img_path = os.path.join(self.output_dir, "images", img_filename)
                            img_info_obj.extracted_path = ImageProcessor.save_image(image_bytes, img_path)
                        images.append(img_info_obj)
                        continue

                    ocr_source = img_array
                    if img_info_obj.has_watermark or img_info_obj.has_stamp:
                        cleaned = ImageProcessor.preprocess_for_ocr(img_array, grayscale=False)
                        ocr_source = cleaned

                        if self.output_dir:
                            cleaned_filename = f"{stem}_page{page_num}_img{img_index}_cleaned.png"
                            cleaned_path = os.path.join(self.output_dir, "images", cleaned_filename)
                            _, encoded = cv2.imencode('.png', cleaned)
                            ImageProcessor.save_image(encoded.tobytes(), cleaned_path)
                            img_info_obj.cleaned_path = cleaned_path

                    if do_ocr if do_ocr is not None else self.ocr_images:
                        already_preprocessed = img_info_obj.has_watermark or img_info_obj.has_stamp
                        img_area = img_array.shape[0] * img_array.shape[1]
                        if img_area >= self.ocr_image_min_area:
                            try:
                                ocr_blocks = self.ocr.recognize_with_positions(
                                    ocr_source, preprocess=not already_preprocessed
                                )
                                ocr_text = self._merge_blocks_to_lines(ocr_blocks)
                                if ocr_text:
                                    img_info_obj.ocr_text = ocr_text
                                logger.info(f"页 {page_num} 图片 {img_index} OCR: {len(ocr_text)} 字符")
                            except Exception as e:
                                logger.warning(f"页 {page_num} 图片 {img_index} OCR 失败: {e}")

                if self.output_dir:
                    img_filename = f"{stem}_page{page_num}_img{img_index}.{image_ext}"
                    img_path = os.path.join(self.output_dir, "images", img_filename)
                    img_info_obj.extracted_path = ImageProcessor.save_image(image_bytes, img_path)

                images.append(img_info_obj)
            except Exception as e:
                logger.warning(f"提取图片失败 (page {page_num}, img {img_index}): {e}")
        return images

    # ---------- Markdown 生成 ----------
    def _to_markdown(self, doc: fitz.Document, pdf_type: str, pages: list[PageContent]) -> str:
        """生成 Markdown（V2：表格内联到文档流中）

        与 Word v2 一致，将检测到的表格 Markdown 插入到对应页面内容之后，
        而非简单地追加到文档末尾。
        """
        # 构建页面到表格的映射
        page_table_map: dict[int, list] = {}
        for page in pages:
            if page.tables:
                page_table_map[page.page_num] = page.tables

        if pdf_type == "image_only":
            # ImageParser 已在 _parse_image_only 中完成 OCR+表格+布局重组
            # PageContent.text 直接就是段落+表格MD的完整输出
            md_text = self._fallback_markdown(pages)
        else:
            try:
                kwargs = dict(doc=doc, write_images=False, page_chunks=True)
                if self.output_dir:
                    kwargs["image_path"] = os.path.join(self.output_dir, "images")
                kwargs["use_ocr"] = False
                page_chunks = pymupdf4llm.to_markdown(**kwargs)
                md_text = self._build_markdown_with_tables(page_chunks, page_table_map, pages)
                md_text = self._recover_missing_text(md_text, doc)
            except Exception as e:
                logger.warning(f"pymupdf4llm 转换失败: {e}，降级为 OCR 结果")
                md_text = self._fallback_markdown(pages)

        return md_text

    def _build_markdown_with_tables(self, page_chunks: list, page_table_map: dict,
                                      pages: list = None) -> str:
        """将 pymupdf4llm 的逐页 Markdown 与增强表格组合

        逐页构建 Markdown，对检测到表格的页面：
        1. 移除 pymupdf4llm 的基础表格输出
        2. 在页面内容后插入 table_handler 生成的增强版表格

        Args:
            page_chunks: pymupdf4llm page_chunks=True 返回的页面列表
            page_table_map: {页码: [PDFTable, ...]} 映射
            pages: PageContent 列表（用于图片 OCR 注入）

        Returns:
            完整的 Markdown 文本，表格内联到对应页面之后
        """
        parts = []

        for i, chunk in enumerate(page_chunks):
            page_num = i + 1
            text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            text = text.strip()

            # 注入图片 OCR 文本
            if pages and page_num <= len(pages):
                text = self._inject_image_ocr_to_page_markdown(text, pages[page_num - 1])

            # 如果此页有增强表格，移除 pymupdf4llm 的基础表格以避免重复
            has_enhanced_tables = page_num in page_table_map
            if has_enhanced_tables:
                text = self._remove_markdown_tables(text)

            if text.strip():
                parts.append(text)

            # 在页面内容后插入增强表格
            if has_enhanced_tables:
                for tbl in page_table_map[page_num]:
                    table_parts = []
                    if tbl.title:
                        table_parts.append(f"**{tbl.title}**")
                        table_parts.append("")
                    table_parts.append(tbl.markdown_text.rstrip())
                    parts.append("\n".join(table_parts))

        return "\n\n".join(parts)

    @staticmethod
    def _remove_markdown_tables(text: str) -> str:
        """从 Markdown 文本中移除已有的基础表格（避免与增强表格重复）

        检测并移除 pymupdf4llm 生成的标准 Markdown 表格（| col1 | col2 | 格式），
        以便用 table_handler 的增强版表格替代。
        """
        lines = text.split('\n')
        result_lines = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            # 检测表头行: | col1 | col2 | ... |
            if line.startswith('|') and line.endswith('|') and line.count('|') >= 3:
                # 向前查看是否有分隔行 | --- | --- |
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line.startswith('|') and '---' in next_line and '|' in next_line[1:]:
                        # 这是一个 Markdown 表格 - 跳过整个表格块
                        i += 2  # 跳过表头和分隔行
                        while i < len(lines):
                            table_line = lines[i].strip()
                            if (table_line.startswith('|') and table_line.endswith('|')
                                    and table_line.count('|') >= 2):
                                i += 1
                            else:
                                break
                        continue
            result_lines.append(lines[i])
            i += 1

        # 清理多余的空行
        result = '\n'.join(result_lines)
        while '\n\n\n' in result:
            result = result.replace('\n\n\n', '\n\n')
        return result.strip()

    @staticmethod
    def _inject_image_ocr_to_page_markdown(page_md: str, page_content) -> str:
        """将单页中图片的 OCR 文本替换到 Markdown 的图片占位符中

        与 _inject_image_ocr_to_markdown 功能相同，但按单页操作，
        避免跨页图片顺序错位问题。
        """
        if not page_content.images:
            return page_md

        # V2: OCR recognize() 已自动包含表格 Markdown
        ocr_texts = []
        for img in page_content.images:
            if img.ocr_text:
                ocr_texts.append(img.ocr_text.strip())
        if not ocr_texts:
            return page_md

        pattern = re.compile(r'\*\*==> picture \[.*?\] intentionally omitted <==\*\*')

        ocr_idx = 0
        def replacer(match):
            nonlocal ocr_idx
            if ocr_idx < len(ocr_texts) and ocr_texts[ocr_idx]:
                text = ocr_texts[ocr_idx]
                ocr_idx += 1
                return f"**[图片内容]**\n\n{text}"
            ocr_idx += 1
            return match.group(0)

        return pattern.sub(replacer, page_md)

    def _recover_missing_text(self, md_text: str, doc: fitz.Document) -> str:
        """恢复被 pymupdf4llm 误识别为图片占位符的文本内容"""
        placeholder_pat = re.compile(
            r'\*\*==> picture \[(\d+) x (\d+)\] intentionally omitted <==\*\*'
        )

        remaining = list(placeholder_pat.finditer(md_text))
        if not remaining:
            return md_text

        real_img_sizes: set[tuple[int, int]] = set()
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            for img_info in page.get_images(full=True):
                try:
                    bbox = page.get_image_bbox(img_info)
                    if bbox and bbox.is_valid:
                        w_pt = round(bbox.width)
                        h_pt = round(bbox.height)
                        real_img_sizes.add((w_pt, h_pt))
                except Exception:
                    pass

        page_lines_map: dict[int, list[str]] = {}
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            lines = [l.strip() for l in page.get_text("text").split('\n')]
            page_lines_map[page_idx] = lines

        _TOLERANCE = 0.20

        def _is_real_image(w: int, h: int) -> bool:
            for rw, rh in real_img_sizes:
                if (abs(w - rw) <= max(rw * _TOLERANCE, 5)
                        and abs(h - rh) <= max(rh * _TOLERANCE, 5)):
                    return True
            return False

        phantom_matches: list[re.Match] = []
        for m in remaining:
            w, h = int(m.group(1)), int(m.group(2))
            if not _is_real_image(w, h):
                phantom_matches.append(m)

        if not phantom_matches:
            return md_text

        logger.info(f"检测到 {len(phantom_matches)} 个伪图片占位符")

        md_stripped = placeholder_pat.sub('', md_text)
        md_norm = re.sub(r'\s+', '', md_stripped)

        all_missing: list[str] = []
        for page_idx in sorted(page_lines_map):
            for line in page_lines_map[page_idx]:
                norm = re.sub(r'\s+', '', line)
                if norm and norm not in md_norm:
                    all_missing.append(line)

        if not all_missing:
            result = md_text
            for m in phantom_matches:
                result = result.replace(m.group(0), '', 1)
            return result

        result = md_text
        used: set[str] = set()
        recovered_count = 0

        for m in phantom_matches:
            original = m.group(0)
            pos = result.find(original)
            if pos == -1:
                continue

            before = result[:pos].rstrip()
            ctx_lines = [l.strip() for l in before.split('\n') if l.strip()]
            ctx = ctx_lines[-1] if ctx_lines else ''
            ctx_norm = re.sub(r'\s+', '', ctx)

            recovered: list[str] = []
            for page_idx in sorted(page_lines_map):
                plines = page_lines_map[page_idx]
                for i, pl in enumerate(plines):
                    if ctx_norm and ctx_norm in re.sub(r'\s+', '', pl):
                        for j in range(i + 1, min(i + 10, len(plines))):
                            ln = plines[j].strip()
                            if not ln:
                                continue
                            ln_norm = re.sub(r'\s+', '', ln)
                            if ln_norm and ln_norm not in md_norm and ln not in used:
                                recovered.append(ln)
                                used.add(ln)
                            elif ln_norm in md_norm:
                                break
                        break
                if recovered:
                    break

            if not recovered:
                for ml in all_missing:
                    if ml not in used:
                        recovered = [ml]
                        used.add(ml)
                        break

            replacement = '\n'.join(recovered) if recovered else ''
            result = result.replace(original, replacement, 1)
            if recovered:
                recovered_count += 1

        if recovered_count:
            logger.info("已从 PDF 文字层恢复 %d 处被误识别为图片的文本内容", recovered_count)

        return result

    @staticmethod
    def _inject_image_ocr_to_markdown(md_text: str, pages: list) -> str:
        """将图片 OCR 文本替换到 pymupdf4llm 生成的 markdown 中"""
        ocr_texts = []
        for page in pages:
            for img in page.images:
                ocr_texts.append(img.ocr_text.strip() if img.ocr_text else "")

        pattern = re.compile(r'\*\*==> picture \[.*?\] intentionally omitted <==\*\*')

        ocr_idx = 0
        def replacer(match):
            nonlocal ocr_idx
            if ocr_idx < len(ocr_texts) and ocr_texts[ocr_idx]:
                text = ocr_texts[ocr_idx]
                ocr_idx += 1
                return f"**[图片内容]**\n\n{text}"
            ocr_idx += 1
            return match.group(0)

        return pattern.sub(replacer, md_text)

    @staticmethod
    def _fallback_markdown(pages: list) -> str:
        """降级方案：从已有 OCR 结果生成简单 Markdown"""
        lines: list[str] = []

        for page in pages:
            lines.append(f"## 第 {page.page_num} 页\n")
            if page.text.strip():
                lines.append(page.text.strip())
                lines.append("")
            # 表格
            for tbl in page.tables:
                if tbl.title:
                    lines.append(f"**{tbl.title}**")
                    lines.append("")
                lines.append(tbl.markdown_text)
                lines.append("")
            # 图片：仅输出水印/签章检测结果，不再贴图片引用
            # （image_only 的图片内容已被整页 OCR 覆盖，贴引用属冗余；
            #  text/mixed 的图片内容已通过 _inject_image_ocr_to_page_markdown 注入）
            if page.images:
                tags = []
                for img in page.images:
                    if img.has_watermark:
                        tags.append("含水印")
                    if img.has_stamp:
                        tags.append("含签章")
                if tags:
                    # 去重后保留顺序
                    seen = set()
                    unique_tags = []
                    for t in tags:
                        if t not in seen:
                            seen.add(t)
                            unique_tags.append(t)
                    lines.append(f"**检测到图片标记：** {', '.join(unique_tags)}")
                    lines.append("")

        return "\n".join(lines)


# ============================================================
# 便捷函数
# ============================================================
def parse_pdf(source: Union[str, bytes], output_dir: Optional[str] = None,
              ocr_backend: OCRBackend = "rapidocr",
              ocr_images: bool = True, file_name: str = None,
              detect_tables: bool = True) -> PDFResult:
    """一键解析 PDF 文档

    Args:
        source: 本地路径 / 远程 URL / S3 URI / 字节流
        output_dir: 输出目录
        ocr_backend: OCR 后端
        ocr_images: 是否对图片做 OCR
        file_name: 文件名（仅 bytes 输入时需要）
        detect_tables: 是否启用表格检测
    """
    parser = PDFParser(output_dir=output_dir,
                       ocr_backend=ocr_backend, ocr_images=ocr_images,
                       detect_tables=detect_tables)
    result = parser.parse(source, file_name=file_name)

    if output_dir:
        stem = Path(result.file_name).stem
        save_markdown(result, os.path.join(output_dir, f"{stem}.md"))
        save_text(result, os.path.join(output_dir, f"{stem}.txt"))

    return result


def parse_directory(dir_path: str, output_dir: Optional[str] = None,
                    recursive: bool = False,
                    ocr_backend: OCRBackend = "rapidocr",
                    ocr_images: bool = True,
                    detect_tables: bool = True) -> list[PDFResult]:
    """批量解析目录下的所有 PDF"""
    parser = PDFParser(output_dir=output_dir,
                       ocr_backend=ocr_backend, ocr_images=ocr_images,
                       detect_tables=detect_tables)
    return parser.parse_directory(dir_path, recursive=recursive)