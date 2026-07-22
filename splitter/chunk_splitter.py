"""文档切分器：基于 LlamaIndex NodeParser 实现，支持 Markdown/句子/Token/代码/语义切分。

这是本模块采用 LlamaIndex 的核心收益点：
- markdown：MarkdownNodeParser，按标题切分并保留层级
- sentence：SentenceSplitter，按句子边界 + chunk_size 窗口
- token：TokenTextSplitter，按 token 精确切分
- code：CodeSplitter，按语法结构切分（支持多种语言）
- semantic：SemanticSplitterNodeParser，按语义相似度切分（chunk 语义完整，适合 SAG 事件抽取）

切分后产出 Chunk 列表，供 extractor 抽取事件。
"""
from __future__ import annotations

import os
import re
import uuid
from typing import Literal

from llama_index.core import Document
from llama_index.core.node_parser import (
    CodeSplitter,
    MarkdownNodeParser,
    SemanticSplitterNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
)

from ..base import Chunk, LoadedDocument
from .base import BaseSplitter

SplitterMode = Literal["markdown", "sentence", "token", "code", "semantic"]


class ChunkSplitter(BaseSplitter):
    def __init__(self, mode: SplitterMode = "markdown", chunk_size: int = 512,
                 chunk_overlap: int = 100, language: str = "python",
                 embed_model=None,
                 breakpoint_percentile_threshold: int = 95) -> None:
        self.mode = mode
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.language = language
        self._embed_model = embed_model
        self._breakpoint_percentile = breakpoint_percentile_threshold
        self._last_heading: str = ""
        self._parser = self._build_parser()

    def _build_parser(self):
        if self.mode == "markdown":
            return MarkdownNodeParser()
        if self.mode == "sentence":
            return SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        if self.mode == "token":
            return TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        if self.mode == "code":
            return CodeSplitter(language=self.language, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        if self.mode == "semantic":
            if self._embed_model is None:
                raise ValueError("semantic 模式需要 embed_model 参数")
            return SemanticSplitterNodeParser(
                embed_model=self._embed_model,
                breakpoint_percentile_threshold=self._breakpoint_percentile,
            )
        raise ValueError(f"未知切分模式: {self.mode}")

    def split(self, doc: LoadedDocument, source_id: str, document_id: str) -> list[Chunk]:
        if not doc.content or not doc.content.strip():
            return []
        # 每次切分新文档时重置沿用上一标题的状态
        self._last_heading = ""
        llama_doc = Document(text=doc.content, metadata={"title": doc.title})
        nodes = self._parser.get_nodes_from_documents([llama_doc])
        chunks: list[Chunk] = []
        for i, node in enumerate(nodes):
            heading = self._extract_heading(node)
            text = node.text or ""
            if not text.strip():
                continue
            chunks.append(Chunk(
                id=str(uuid.uuid4()), document_id=document_id, source_id=source_id,
                rank_index=i, heading=heading, content=text,
            ))
        return chunks

    def _extract_heading(self, node) -> str:
        meta = getattr(node, "metadata", {}) or {}
        text = getattr(node, "text", "") or ""

        # L1: 优先从 Markdown 元数据取标题（MarkdownNodeParser 会填充 Header_1/2/3）
        for key in ("Header_1", "Header_2", "Header_3", "header"):
            val = meta.get(key)
            if val and str(val).strip():
                heading = str(val).strip()
                self._last_heading = heading
                return heading

        # L2: 从文本开头提取结构化标题（覆盖 semantic 模式标题行被切散的情况）
        heading_from_text = self._heading_from_text(text)
        if heading_from_text:
            self._last_heading = heading_from_text
            return heading_from_text

        # L3: 沿用上一个有效标题（同一章节被切分成多个 chunk 时继承标题）
        if self._last_heading:
            return self._last_heading

        # L4: 兜底用文件名（去后缀），再不行返回 "Introduction"
        fallback = meta.get("title")
        if fallback and str(fallback).strip():
            name = str(fallback).strip()
            # 去掉常见的文件扩展名（.md / .pdf / .docx / .txt 等）
            root, _ = os.path.splitext(name)
            return root if root else name
        return "Introduction"

    @staticmethod
    def _heading_from_text(text: str) -> str:
        """从 chunk 文本开头提取章节标题。

        识别规则（按优先级，只看前 5 行，避免误判正文里的编号）：
        1. Markdown 标题：# / ## / ### 开头
        2. 中文章节：第X章/第X节/第X部分
        3. 中文序号：一、 二、 等
        4. 数字序号：1. / 1.1 / 1.1.1 等（标题需含中文且较短，排除列表项）
        """
        if not text:
            return ""
        for line in text.split("\n")[:5]:
            line = line.strip()
            if not line:
                continue
            # 1) Markdown 标题：## 标题 / # 标题
            m = re.match(r'^(#{1,6})\s+(.+)$', line)
            if m:
                return m.group(2).strip()[:80]
            # 2) 中文章节：第一章 / 第一节 / 第一部分
            m = re.match(r'^第[一二三四五六七八九十百]+[章节部分][\s、．.]?\s*(.*)$', line)
            if m:
                title = m.group(1).strip()
                if title:
                    return title[:80]
                # 如果没标题文字（如只有"第一章"），原样返回
                return line[:80]
            # 3) 中文序号：一、 标题
            m = re.match(r'^[一二三四五六七八九十]+[、]\s*(.+)$', line)
            if m:
                return m.group(1).strip()[:80]
            # 4) 数字序号：1. 标题 / 1.1 标题 / 1.1.1 标题（标题需含中文且较短）
            m = re.match(r'^(\d+)(\.\d+)*[\s、．.]\s*(.+)$', line)
            if m:
                candidate = m.group(3).strip()
                # 排除纯数字行、过长的正文行、明显是列表项（全是名词且很短的也算列表）
                if 2 <= len(candidate) <= 60 and re.search(r'[\u4e00-\u9fa5]', candidate):
                    return candidate[:80]
        return ""