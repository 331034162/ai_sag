"""表格行切分器：针对 Excel V6 CSV 文本按数据行切分，每行带表头列名。

设计动机：Excel 表格（如需求清单）入库后，若按句子/语义切分，表格行会被打散，
导致"创建人"等列名与值分离，LLM 无法正确识别人名实体（如"汪晨"被漏抽）。

本切分器针对 V6 产出的 CSV 文本（多 sheet 用 "=== sheet名 ===" 分隔）：

结构识别标注模式（V6 开启 ENABLE_STRUCTURE_DETECTION 时）：
- V6 在每行 CSV 行首加角色前缀：#TITLE#/#FORM#/#SIGNING#/#GROUP_HEADER#/#DATA#
- 本切分器据此：
  - 跳过 TITLE/FORM/SIGNING/GROUP_HEADER 行，不作为数据行
  - 取第一个 #DATA# 行作真表头（叶子表头，跳过其上方的标题/表单/分组表头）
  - 提取 TITLE 作 chunk 标题、FORM 作表单上下文，注入到每个 chunk（跨chunk不丢失）
- 解决问题：复杂表格（采购入库单/供应商资质信息表）的标题/表单行被误当数据行、
  分组表头被误当列名、表单上下文跨chunk丢失等问题。

兼容模式（V6 关闭结构识别时，CSV 无前缀）：
- 退化为原行为：rows[0] 当表头，rows[1:] 当数据行

通用能力：
- 数据行按字符数窗口聚合，单行不切断（保证行内实体完整）
- 每行以"列名: 值"格式呈现，列名上下文内嵌于每行，LLM 可直接识别实体归属
"""
from __future__ import annotations

import csv
import io
import re
import uuid
from typing import Iterator

from ..base import Chunk, LoadedDocument
from .base import BaseSplitter

_SHEET_SEP = re.compile(r'^=== (.+?) ===$', re.MULTILINE)
# 匹配 V6 输出的角色前缀，如 #TITLE#供应链原材料采购入库单,,,,,,,
_ROLE_RE = re.compile(r'^#(TITLE|FORM|SIGNING|GROUP_HEADER|DATA)#')

# 汇总行关键词：首列含这些词的行视为聚合行（非个体记录），加 [汇总] 标记避免误抽实体
_SUMMARY_KEYWORDS = ("合计", "小计", "总计", "总和", "汇总", "累计")

# 数据行角色：会被切分入库的行
_DATA_ROLE = "data"


class TableSplitter(BaseSplitter):
    """表格行切分器。

    Args:
        chunk_size: 单个 chunk 最大字符数
        chunk_overlap: 切分重叠（表格行切分通常不需要重叠，默认 0）
    """

    def __init__(self, chunk_size: int = 8192, chunk_overlap: int = 0) -> None:
        self.chunk_size = max(chunk_size, 512)
        self.chunk_overlap = chunk_overlap

    def split(self, doc: LoadedDocument, source_id: str, document_id: str) -> list[Chunk]:
        if not doc.content or not doc.content.strip():
            return []

        chunks: list[Chunk] = []
        rank = 0

        for sheet_name, csv_text in self._split_sheets(doc.content):
            parsed = self._parse_annotated_rows(csv_text)
            if not parsed:
                continue

            # 提取标题和表单上下文（跨chunk注入，避免后续chunk丢失表单信息）
            title = self._extract_title(parsed)
            form_fields = self._extract_form(parsed)
            signing_info = self._extract_signing(parsed)

            # 定位真表头：第一个 DATA 行（V6 已将分组表头识别为 GROUP_HEADER）
            header_idx = self._find_header_index(parsed)
            if header_idx is None:
                continue

            header = parsed[header_idx][1]
            # 数据行：真表头之后的所有 DATA 行（跳过中间的 GROUP_HEADER/SIGNING 等非数据行）
            data_rows = [fields for role, fields in parsed[header_idx + 1:]
                         if role == _DATA_ROLE]

            for chunk_text in self._window_rows(sheet_name, title, form_fields, signing_info, header, data_rows):
                if chunk_text.strip():
                    chunks.append(Chunk(
                        id=str(uuid.uuid4()),
                        document_id=document_id,
                        source_id=source_id,
                        rank_index=rank,
                        heading=sheet_name or doc.title or "表格",
                        content=chunk_text,
                        metadata={"sheet": sheet_name, "node_id": f"table-{rank}"},
                    ))
                    rank += 1

        return chunks

    @staticmethod
    def _split_sheets(content: str) -> Iterator[tuple[str, str]]:
        matches = list(_SHEET_SEP.finditer(content))
        if not matches:
            yield ("", content)
            return
        for i, m in enumerate(matches):
            sheet_name = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            csv_text = content[start:end].strip()
            if csv_text:
                yield (sheet_name, csv_text)

    @staticmethod
    def _parse_annotated_rows(csv_text: str) -> list[tuple[str, list[str]]]:
        """解析 CSV 文本为 [(role, fields), ...]。

        支持 V6 结构识别标注（#TITLE#/#FORM#/.../#DATA#）：
        - 行首匹配到角色前缀时，剥离前缀，剩余部分作为 CSV 字段
        - 无前缀时（兼容模式）视为 data 行
        """
        rows = list(csv.reader(io.StringIO(csv_text)))
        result: list[tuple[str, list[str]]] = []
        for row in rows:
            if not row:
                continue
            first = row[0]
            m = _ROLE_RE.match(first)
            if m:
                role = m.group(1).lower()
                remaining = first[m.end():]
                fields = [remaining] + row[1:]
            else:
                role = _DATA_ROLE
                fields = row
            result.append((role, fields))
        return result

    @staticmethod
    def _extract_title(parsed: list[tuple[str, list[str]]]) -> str:
        """提取标题：第一个 TITLE 行的第一个非空字段"""
        for role, fields in parsed:
            if role == "title":
                for v in fields:
                    v = (v or "").strip()
                    if v:
                        return v
                break
        return ""

    @staticmethod
    def _extract_form(parsed: list[tuple[str, list[str]]]) -> str:
        """提取表单字段：所有 FORM 行的非空字段，用 ' | ' 拼接"""
        parts: list[str] = []
        for role, fields in parsed:
            if role == "form":
                for v in fields:
                    v = (v or "").strip()
                    if v:
                        parts.append(v)
        return " | ".join(parts)

    @staticmethod
    def _extract_signing(parsed: list[tuple[str, list[str]]]) -> str:
        """提取签章信息：所有 SIGNING 行的非空字段，用 ' | ' 拼接。

        签章行含人名（如"申请人签字：张三"），跳过会导致人名实体丢失，
        故提取后注入到每个chunk，保留签章人名供实体抽取。
        """
        parts: list[str] = []
        for role, fields in parsed:
            if role == "signing":
                for v in fields:
                    v = (v or "").strip()
                    if v:
                        parts.append(v)
        return " | ".join(parts)

    @staticmethod
    def _find_header_index(parsed: list[tuple[str, list[str]]]) -> int | None:
        """定位真表头索引：第一个 DATA 行。

        V6 已将分组表头识别为 GROUP_HEADER，叶子表头（如"大类,物料编码,..."）
        是第一个 DATA 行。兼容模式下（无前缀）第一行即视为表头。
        """
        for i, (role, _) in enumerate(parsed):
            if role == _DATA_ROLE:
                return i
        return None

    def _window_rows(self, sheet_name: str, title: str, form_fields: str,
                     signing_info: str, header: list[str],
                     data_rows: list[list[str]]) -> Iterator[str]:
        """按字符数窗口聚合数据行，每个chunk带 sheet名+标题+表单+签章+表头 上下文"""
        prefix_parts: list[str] = []
        if sheet_name:
            prefix_parts.append(f"## {sheet_name}")
        if title:
            prefix_parts.append(f"标题: {title}")
        if form_fields:
            prefix_parts.append(f"表单: {form_fields}")
        if signing_info:
            prefix_parts.append(f"签章: {signing_info}")
        header_line = "表头: " + " | ".join(h for h in header if h and h.strip())
        prefix_parts.append(header_line)
        prefix = "\n".join(prefix_parts) + "\n\n"
        prefix_size = len(prefix)

        current_lines: list[str] = []
        current_size = prefix_size

        for row in data_rows:
            row_text = self._format_row(header, row)
            if not row_text.strip():
                continue
            # 汇总行检测：首列含"合计/小计/总计"等 → 加 [汇总] 标记，避免"合计"被误抽为实体
            if self._is_summary_row(row):
                row_text = "[汇总] " + row_text
            row_size = len(row_text) + 1

            if current_lines and current_size + row_size > self.chunk_size:
                yield prefix + "".join(current_lines)
                current_lines = []
                current_size = prefix_size

            current_lines.append(row_text)
            current_size += row_size

        if current_lines:
            yield prefix + "".join(current_lines)

    @staticmethod
    def _format_row(header: list[str], row: list[str]) -> str:
        lines = []
        for i, val in enumerate(row):
            col_name = header[i] if i < len(header) else f"列{i + 1}"
            col_name = col_name.strip() if col_name else ""
            val = str(val).strip() if val else ""
            if not val:
                continue
            # 列名与值完全相同（如纵向合并填充的分类值"供应商资质信息"）→ 无意义冗余，跳过
            # 场景：A1:A16纵向合并填充"供应商资质信息"到叶子表头列名和数据行，导致每行出现
            # "供应商资质信息: 供应商资质信息"，该分类上下文已由sheet名/标题承载，此处跳过省token
            if col_name and col_name == val:
                continue
            display_name = col_name if col_name else f"列{i + 1}"
            lines.append(f"{display_name}: {val}")
        return "\n".join(lines) + "\n\n" if lines else ""

    @staticmethod
    def _is_summary_row(row: list[str]) -> bool:
        """汇总行检测：首列非空值含"合计/小计/总计/汇总/累计"等关键词。

        汇总行是聚合数据（非个体记录），如"合计 | ¥3,361.00"，
        加 [汇总] 标记让 LLM 明确这是聚合值，避免"合计"被误抽为公司/人名实体。
        """
        for val in row:
            val = (val or "").strip()
            if not val:
                continue
            return any(kw in val for kw in _SUMMARY_KEYWORDS)
        return False