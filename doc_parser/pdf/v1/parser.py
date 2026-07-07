"""
PDF 解析器主类
==============
支持文本/混合/扫描版 PDF 解析，可选 PaddleOCR 或 RapidOCR 后端
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

from ai_sag.doc_parser.pdf.v1.models import ImageInfo, PageContent, PDFResult
from ai_sag.doc_parser.pdf.v1.detector import PDFTypeDetector
from ai_sag.doc_parser.pdf.v1.image_processor import ImageProcessor
from ai_sag.doc_parser.pdf.v1.formatter import save_markdown, save_text
from ai_sag.doc_parser.image.ocr import OCRBackend, BaseOCREngine, get_ocr_engine
from ai_sag.doc_parser.image.parser import ocr_for_pymupdf
from ai_sag.doc_parser.pdf.v1.config import OCR_PAGE_DPI, OCR_PYMUPDF_DPI, setup_logging

# 初始化日志（模块级别，确保整个包可用）
setup_logging()

logger = logging.getLogger(__name__)


# ============================================================
# PDF 解析器
# ============================================================
class PDFParser:
    """PDF 解析器主类"""

    def __init__(self, output_dir: Optional[str] = None, extract_images: bool = True,
                 ocr_backend: OCRBackend = "rapidocr",
                 ocr_images: bool = True, ocr_image_min_area: int = 10000,
                 markdown_mode: str = "direct"):
        """
        Args:
            output_dir: 输出目录，传值则保存解析结果（md/txt/图片）到磁盘，
                        传 None 则不写磁盘，仅返回内存中的解析结果
            extract_images: 是否提取并处理图片（水印检测、签章检测、OCR）
            ocr_backend: OCR 后端引擎
            ocr_images: 是否对 text/mixed 类型 PDF 中提取的图片执行 OCR
            ocr_image_min_area: 图片最小面积（像素²），低于此值的图片不做 OCR，
                                避免对小图标/装饰图浪费算力
            markdown_mode: 扫描件 markdown 生成模式，可选值：
                - "pymupdf4llm" (方案A)：用 pymupdf4llm + 自适应字号生成，
                  排版更紧凑，但 OCR 重复执行且部分内容可能被截断/误识别
                - "direct" (默认，方案B)：直接用已有 OCR 结果构建，内容更完整准确，
                  避免重复 OCR 和截断风险，排版为逐行排列
        """
        self.output_dir = output_dir
        self.extract_images = extract_images
        self.ocr_backend = ocr_backend
        self.ocr_images = ocr_images
        self.ocr_image_min_area = ocr_image_min_area
        self.markdown_mode = markdown_mode
        self._ocr: "BaseOCREngine | None" = None  # 延迟初始化

    @property
    def ocr(self) -> BaseOCREngine:
        """延迟获取 OCR 引擎（首次使用时才初始化，避免 import 时卡顿）"""
        if self._ocr is None:
            self._ocr = get_ocr_engine(self.ocr_backend)
        return self._ocr

    # ---------- 公开方法 ----------
    def parse(self, source: Union[str, bytes], file_name: Optional[str] = None) -> PDFResult:
        """
        解析单个 PDF 文档

        Args:
            source: PDF 数据源，支持以下格式：
                    - 本地文件路径 (str): "/path/to/file.pdf"
                    - 远程 URL (str): "https://..." 或 "s3://bucket/key.pdf"
                    - 字节流 (bytes): PDF 原始字节数据
            file_name: 文件名（用于输出文件命名），仅 bytes 输入时需要，
                       本地路径和 URL 会自动提取文件名

        Returns:
            PDFResult 解析结果
        """
        # ---- 统一转换为 (pdf_bytes, display_name) ----
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
                # 本地文件
                if not os.path.exists(source):
                    raise FileNotFoundError(f"PDF 文件不存在: {source}")
                with open(source, "rb") as f:
                    pdf_bytes = f.read()
                display_name = file_name or os.path.basename(source)
        else:
            raise TypeError(f"不支持的 source 类型: {type(source)}，期望 str 或 bytes")

        logger.info(f"开始解析: {display_name} ({len(pdf_bytes)} 字节)")

        # ---- 从字节流打开 PDF ----
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

            if pdf_type == "image_only":
                self._parse_image_only(doc, result)
            else:
                self._parse_text_or_mixed(doc, result)

            # 在文档仍然打开的状态下生成 Markdown（pymupdf4llm 可直接用 doc，
            # 避免关闭后重新打开文件）
            result.markdown_text = self._to_markdown(doc, pdf_type, result.pages)
        finally:
            doc.close()

        # 合并文本
        result.full_text = "\n\n".join(
            p.text for p in result.pages if p.text.strip()
        )

        logger.info(f"解析完成: {len(result.pages)} 页, 文本长度 {len(result.full_text)} 字符")
        return result

    # ---------- 远程文件获取 ----------
    @staticmethod
    def _fetch_url(url: str) -> bytes:
        """通过 HTTP/HTTPS 下载文件到内存"""
        import urllib.request
        response = urllib.request.urlopen(url)
        return response.read()

    @staticmethod
    def _fetch_s3(s3_uri: str) -> bytes:
        """从 S3 下载文件到内存

        需要安装 boto3: pip install boto3
        环境变量配置: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
        """
        try:
            import boto3
        except ImportError:
            raise ImportError("S3 支持需要 boto3，请安装: pip install boto3")

        # 解析 s3://bucket/key.pdf
        parts = s3_uri.replace("s3://", "").split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"S3 URI 格式错误: {s3_uri}，期望 s3://bucket/key")
        bucket, key = parts

        s3_client = boto3.client("s3")
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()

    @staticmethod
    def _extract_filename_from_url(url: str) -> str:
        """从 URL 中提取文件名"""
        # 去掉查询参数
        path = url.split("?")[0].split("#")[0]
        name = os.path.basename(path)
        # 如果提取不到合理的文件名，用默认值
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

    # ---------- 内部解析方法 ----------
    def _parse_text_or_mixed(self, doc: fitz.Document, result: PDFResult) -> None:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            has_images = bool(page.get_images(full=True))

            # 纯文本页：跳过图片提取和 OCR，直接记录
            if not has_images or not self.extract_images:
                result.pages.append(PageContent(
                    page_num=page_num,
                    text=text.strip(),
                    images=[],
                    is_image_only=False,
                ))
                continue

            images = self._extract_images_from_page(page, page_num, result.file_name,
                                                     page_text=text)

            # 将图片 OCR 文本追加到页面文本中
            image_ocr_parts = []
            for img in images:
                if img.ocr_text.strip():
                    image_ocr_parts.append(img.ocr_text.strip())
            if image_ocr_parts:
                combined_text = text.strip() + "\n\n" + "\n\n".join(image_ocr_parts) \
                    if text.strip() else "\n\n".join(image_ocr_parts)
            else:
                combined_text = text.strip()

            result.pages.append(PageContent(
                page_num=page_num,
                text=combined_text,
                images=images,
                is_image_only=(len(text.strip()) == 0 and len(images) > 0),
            ))

    def _parse_image_only(self, doc: fitz.Document, result: PDFResult) -> None:
        logger.info("使用 OCR 策略解析扫描版 PDF")
        for page_num, page in enumerate(doc, start=1):
            # image_only 页面无文字层，传空字符串，不会触发背景水印跳过逻辑
            # do_ocr=False：整页 OCR 已覆盖图片内容，无需对嵌入图片重复 OCR
            images = self._extract_images_from_page(page, page_num, result.file_name,
                                                     page_text="", do_ocr=False) \
                if self.extract_images else []
            text = self._ocr_page(page, page_num)

            result.pages.append(PageContent(
                page_num=page_num, text=text, images=images, is_image_only=True,
            ))

    # ---------- 图片提取 ----------
    def _extract_images_from_page(self, page: fitz.Page, page_num: int,
                                  file_name: str, page_text: str = "",
                                  do_ocr: Optional[bool] = None) -> list[ImageInfo]:
        """提取页面中的所有图片并做水印/签章检测和 OCR

        Args:
            page: PyMuPDF 页面对象
            page_num: 页码（1-based）
            file_name: PDF 文件名（用于输出文件命名）
            page_text: 页面 PDF 文字层内容，用于判断是否可安全跳过背景水印图的 OCR
            do_ocr: 是否对图片做 OCR。None 时用 self.ocr_images；
                    image_only 分支应传 False（整页 OCR 已覆盖图片内容）。
        """
        images: list[ImageInfo] = []
        stem = Path(file_name).stem  # 提前计算，避免重复
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

                # 转为 cv2 数组进行图像处理（水印/签章检测、OCR）
                img_array = ImageProcessor.image_to_cv2_array(image_bytes)
                if img_array is not None:
                    watermark_result = ImageProcessor.detect_watermark(img_array)
                    img_info_obj.has_watermark = watermark_result['has_watermark']
                    img_info_obj.has_stamp = ImageProcessor.detect_stamp(img_array)['has_stamp']

                    # [2025-06 改动] 跳过背景水印图的 OCR 和水印去除
                    # 缘由：背景水印图（几乎全白、仅有浅灰水印文字和彩色logo）OCR 出来的
                    # 都是水印文字（噪音），且页面文字层已包含有效内容，无需浪费 OCR 算力。
                    # 逻辑：水印类型为 background 且页面文字层非空 → 跳过该图片的全部后续处理
                    # 兜底：文字层为空时不跳过，避免丢失唯一内容来源
                    if (watermark_result['has_watermark']
                            and watermark_result['type'] == 'background'
                            and page_text.strip()):
                        logger.info(f"页 {page_num} 图片 {img_index}: 背景水印图，"
                                    f"页面已有文字层，跳过 OCR 和水印去除")
                        # 仍保存原图，但不做 OCR、不去水印、不保存 cleaned
                        if self.output_dir:
                            img_filename = f"{stem}_page{page_num}_img{img_index}.{image_ext}"
                            img_path = os.path.join(self.output_dir, "images", img_filename)
                            img_info_obj.extracted_path = ImageProcessor.save_image(image_bytes, img_path)
                        images.append(img_info_obj)
                        continue

                    # 选择用于 OCR 的图像：有水印/签章时用清洗后的图，否则用原图
                    ocr_source = img_array
                    if img_info_obj.has_watermark or img_info_obj.has_stamp:
                        cleaned = ImageProcessor.preprocess_for_ocr(img_array, grayscale=False)
                        ocr_source = cleaned
                        logger.info(f"页 {page_num} 图片 {img_index}: "
                                    f"水印={img_info_obj.has_watermark}, 签章={img_info_obj.has_stamp}")

                        # 仅在指定 output_dir 时保存清洗后的图片
                        if self.output_dir:
                            cleaned_filename = f"{stem}_page{page_num}_img{img_index}_cleaned.png"
                            cleaned_path = os.path.join(self.output_dir, "images", cleaned_filename)
                            _, encoded = cv2.imencode('.png', cleaned)
                            ImageProcessor.save_image(encoded.tobytes(), cleaned_path)
                            img_info_obj.cleaned_path = cleaned_path

                    # 对大图执行 OCR（避免对小图标浪费算力）
                    # 注意：若 ocr_source 已经是 preprocess_for_ocr 的输出（有水印/签章时），
                    # 则不再重复预处理
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

                # 仅在指定 output_dir 时保存原图到磁盘
                if self.output_dir:
                    img_filename = f"{stem}_page{page_num}_img{img_index}.{image_ext}"
                    img_path = os.path.join(self.output_dir, "images", img_filename)
                    img_info_obj.extracted_path = ImageProcessor.save_image(image_bytes, img_path)

                images.append(img_info_obj)
            except Exception as e:
                logger.warning(f"提取图片失败 (page {page_num}, img {img_index}): {e}")
        return images

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

        # 主引擎 OCR
        try:
            blocks = self.ocr.recognize_with_positions(img_array, preprocess=True)
            ocr_text = self._merge_blocks_to_lines(blocks)
            if ocr_text:
                logger.info(f"页面 {page_num} {self.ocr_backend} 识别成功: {len(ocr_text)} 字符")
                return ocr_text
        except Exception as e:
            logger.warning(f"{self.ocr_backend} 识别失败: {e}")

        # 备选：pymupdf 内置 OCR
        try:
            textpage = page.get_textpage_ocr(flags=0)
            if textpage:
                text = textpage.extractText()
                if text and text.strip():
                    return text.strip()
        except Exception:
            pass

        return ""

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

    # ---------- Markdown 生成 ----------
    def _to_markdown(self, doc: fitz.Document, pdf_type: str, pages: list) -> str:
        """生成 Markdown

        扫描件(image_only)有两种模式（由 markdown_mode 控制）：
        - "pymupdf4llm" (方案A)：通过 ocr_function 将 OCR 文字插入页面 → pymupdf4llm 提取
          排版更紧凑，但 OCR 重复执行、部分内容可能截断或误识别
        - "direct" (方案B)：直接用已有 OCR 结果构建，内容更完整准确，排版为逐行排列

        文本型/混合型始终使用 pymupdf4llm 提取文字层，并注入图片 OCR 文本
        """
        # 扫描件：根据 markdown_mode 选择生成方式
        if pdf_type == "image_only":
            if self.markdown_mode == "direct":
                # 方案B：直接用 OCR 结果构建，避免重复 OCR 和截断风险
                return self._fallback_markdown(pages)
            else:
                # 方案A (默认)：pymupdf4llm + 自适应字号 OCR 回调
                try:
                    kwargs = dict(
                        doc=doc,
                        write_images=False,
                    )
                    if self.output_dir:
                        kwargs["image_path"] = os.path.join(self.output_dir, "images")
                    kwargs["ocr_dpi"] = OCR_PYMUPDF_DPI
                    kwargs["ocr_function"] = lambda page, **kw: ocr_for_pymupdf(
                        page, self.ocr, **kw
                    )

                    md_text = pymupdf4llm.to_markdown(**kwargs)
                    return md_text
                except Exception as e:
                    logger.warning(f"pymupdf4llm 转换失败: {e}，降级为 OCR 结果")
                    return self._fallback_markdown(pages)

        # 文本型/混合型：pymupdf4llm 直接提取文字层
        try:
            kwargs = dict(
                doc=doc,
                write_images=False,
            )
            if self.output_dir:
                kwargs["image_path"] = os.path.join(self.output_dir, "images")
            kwargs["use_ocr"] = False

            md_text = pymupdf4llm.to_markdown(**kwargs)

            # 对 text/mixed 类型，将图片 OCR 文本注入到 markdown 中
            md_text = self._inject_image_ocr_to_markdown(md_text, pages)

            # 恢复被 pymupdf4llm 误识别为图片的文本内容
            md_text = self._recover_missing_text(md_text, doc)

            return md_text
        except Exception as e:
            logger.warning(f"pymupdf4llm 转换失败: {e}，降级为 OCR 结果")
            return self._fallback_markdown(pages)

    def _recover_missing_text(self, md_text: str, doc: fitz.Document) -> str:
        """恢复被 pymupdf4llm 误识别为图片占位符的文本内容。

        pymupdf4llm 有时会将文本元素（如表格内文字、表单字段、行内图形旁的文字）
        误识别为图片，生成 ``**==> picture [W x H] intentionally omitted <==**``
        占位符，导致 markdown 内容丢失。

        本方法通过对比 PDF 原始文字层与真实图片列表，识别"伪图片占位符"（不对应
        任何真实嵌入图片的占位符），并从文字层中恢复缺失的文本。

        判定逻辑：
        1. 收集 PDF 每页所有真实嵌入图片的边界框（来自 page.get_images）
        2. 对 markdown 中剩余的占位符，检查其尺寸是否与某页真实图片近似匹配
        3. 无法匹配任何真实图片的占位符 → 判定为"伪图片"，需要恢复文本
        4. 能匹配真实图片的占位符 → 保留不动

        Args:
            md_text: 经过 OCR 注入后的 markdown 文本
            doc: 仍然打开的 PyMuPDF Document 对象

        Returns:
            恢复缺失文本后的 markdown 文本
        """
        placeholder_pat = re.compile(
            r'\*\*==> picture \[(\d+) x (\d+)\] intentionally omitted <==\*\*'
        )

        remaining = list(placeholder_pat.finditer(md_text))
        if not remaining:
            return md_text

        # ---- 步骤 1：收集每页所有真实嵌入图片的 (宽, 高) ----
        # pymupdf4llm 的占位符尺寸 [W x H] 单位为 pt（点），
        # 而 page.get_images + get_image_bbox 得到的也是 pt 单位
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

        # ---- 步骤 2：收集每页的文字层行 ----
        page_lines_map: dict[int, list[str]] = {}
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            lines = [l.strip() for l in page.get_text("text").split('\n')]
            page_lines_map[page_idx] = lines

        # ---- 步骤 3：识别"伪图片占位符" ----
        # 占位符尺寸不与任何真实图片匹配 → 伪图片，需要恢复
        # 使用容差匹配：尺寸差 ≤ 20% 视为同一图片（pymupdf4llm 与 get_image_bbox
        # 可能有舍入差异）
        _TOLERANCE = 0.20

        def _is_real_image(w: int, h: int) -> bool:
            """判断占位符尺寸是否对应 PDF 中的真实嵌入图片"""
            for rw, rh in real_img_sizes:
                if (abs(w - rw) <= max(rw * _TOLERANCE, 5)
                        and abs(h - rh) <= max(rh * _TOLERANCE, 5)):
                    return True
            return False

        # 将占位符分为：伪图片（需恢复）和真实图片（保留）
        phantom_matches: list[re.Match] = []  # 不对应真实图片的占位符
        for m in remaining:
            w, h = int(m.group(1)), int(m.group(2))
            if not _is_real_image(w, h):
                phantom_matches.append(m)

        if not phantom_matches:
            return md_text

        logger.info(f"检测到 {len(phantom_matches)} 个伪图片占位符"
                    f"（不对应真实嵌入图片，可能为文本误识别）")

        # ---- 步骤 4：构建 markdown 已有内容的索引（去空白后） ----
        md_stripped = placeholder_pat.sub('', md_text)
        md_norm = re.sub(r'\s+', '', md_stripped)

        # 收集"页面文字层中有、但 markdown 中缺失"的行
        all_missing: list[str] = []
        for page_idx in sorted(page_lines_map):
            for line in page_lines_map[page_idx]:
                norm = re.sub(r'\s+', '', line)
                if norm and norm not in md_norm:
                    all_missing.append(line)

        if not all_missing:
            # 没有缺失内容，仅移除伪图片占位符（它们不对应任何真实图片，
            # 也不代表任何缺失文本，属于 pymupdf4llm 的冗余输出）
            result = md_text
            for m in phantom_matches:
                result = result.replace(m.group(0), '', 1)
            if result != md_text:
                logger.info("已移除 %d 个无内容缺失的伪图片占位符",
                            len(phantom_matches))
            return result

        # ---- 步骤 5：基于上下文恢复缺失文本 ----
        # 对每个伪图片占位符，找到其前方文本在页面文字层中的位置，
        # 然后提取紧随其后的缺失行
        # 注意：不能直接用 match.start()/match.end()，因为替换会改变偏移量。
        # 改用逐个查找替换的方式
        result = md_text
        used: set[str] = set()
        recovered_count = 0

        for m in phantom_matches:
            original = m.group(0)  # 完整的占位符字符串
            w, h = int(m.group(1)), int(m.group(2))

            # 在当前 result 中找到该占位符的位置
            pos = result.find(original)
            if pos == -1:
                continue  # 理论上不会发生，但防御性处理

            # 取占位符前方最后一段非空文本作为上下文
            before = result[:pos].rstrip()
            ctx_lines = [l.strip() for l in before.split('\n') if l.strip()]
            ctx = ctx_lines[-1] if ctx_lines else ''
            ctx_norm = re.sub(r'\s+', '', ctx)

            # 在页面文字层中搜索上下文，定位缺失行的位置
            recovered: list[str] = []
            for page_idx in sorted(page_lines_map):
                plines = page_lines_map[page_idx]
                for i, pl in enumerate(plines):
                    if ctx_norm and ctx_norm in re.sub(r'\s+', '', pl):
                        # 找到上下文行，收集其后的缺失行
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
                # 兜底：按顺序取下一个未使用的缺失行
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
            logger.info("已从 PDF 文字层恢复 %d 处被误识别为图片的文本内容",
                        recovered_count)

        return result

    @staticmethod
    def _inject_image_ocr_to_markdown(md_text: str, pages: list) -> str:
        """将图片 OCR 文本替换到 pymupdf4llm 生成的 markdown 中

        pymupdf4llm 对 text/mixed 类型生成 `**==> picture [...] intentionally omitted <==**`
        占位符表示图片位置。此方法按顺序将占位符替换为对应的图片 OCR 文本。
        """
        # 收集所有页面的图片 OCR 文本（按页面和图片顺序）
        ocr_texts = []
        for page in pages:
            for img in page.images:
                ocr_texts.append(img.ocr_text.strip() if img.ocr_text else "")

        # 按顺序替换 "intentionally omitted" 占位符
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
            # 图片：仅输出水印/签章检测结果，不再贴图片引用
            # （image_only 的图片内容已被整页 OCR 覆盖，贴引用属冗余；
            #  text/mixed 的图片内容已通过 _inject_image_ocr_to_markdown 注入）
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
              markdown_mode: str = "direct") -> PDFResult:
    """一键解析 PDF 文档

    Args:
        source: 本地路径 / 远程 URL / S3 URI / 字节流
        output_dir: 输出目录，传值则保存解析结果到磁盘，传 None 则仅返回内存结果
        ocr_backend: OCR 后端
        ocr_images: 是否对图片做 OCR
        file_name: 文件名（仅 bytes 输入时需要）
        markdown_mode: 扫描件 markdown 生成模式，可选 "direct"(默认) 或 "pymupdf4llm"。
            image_only 类型 PDF 默认用 "direct"，跳过 pymupdf4llm 二次版面分析，
            避免 multi-line 表头表格列错位、列表误判、内容截断等问题。
    """
    parser = PDFParser(output_dir=output_dir,
                       ocr_backend=ocr_backend, ocr_images=ocr_images,
                       markdown_mode=markdown_mode)
    result = parser.parse(source, file_name=file_name)

    if output_dir:
        stem = Path(result.file_name).stem
        save_markdown(result, os.path.join(output_dir, f"{stem}.md"))
        save_text(result, os.path.join(output_dir, f"{stem}.txt"))

    return result


def parse_directory(dir_path: str, output_dir: Optional[str] = None,
                    recursive: bool = False,
                    ocr_backend: OCRBackend = "rapidocr",
                    ocr_images: bool = True) -> list[PDFResult]:
    """批量解析目录下的所有 PDF"""
    parser = PDFParser(output_dir=output_dir,
                       ocr_backend=ocr_backend, ocr_images=ocr_images)
    return parser.parse_directory(dir_path, recursive=recursive)