"""向量库抽象基类：定义统一接口，便于切换 ChromaDB/Milvus/FAISS。

同步方法供入库等同步场景使用；
异步方法（a 前缀）供异步检索流程使用。

默认实现：基类用 asyncio.to_thread 包装同步实现，兼容 ChromaDB 等无异步 SDK 的后端。
纯协程实现：Milvus / PGVector 后端重写了所有 a* 方法，使用 LlamaIndex 原生异步接口
（AsyncMilvusClient / async SQLAlchemy），为真正的协程，无需线程池。
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Literal

Collection = Literal["chunks", "event_titles", "event_contents", "event_summaries", "entities"]


class BaseVectorStore(ABC):
    """向量库后端抽象。管理 4 个逻辑集合，提供写入与查询。"""

    # ---------------- 同步接口 ----------------

    @abstractmethod
    def add(self, name: Collection, ids: list[str], texts: list[str],
            embeddings: list[list[float]], metadatas: list[dict] | None = None) -> None:
        ...

    @abstractmethod
    def query(self, name: Collection, query_embedding: list[float], top_k: int,
              similarity_threshold: float = 0.0,
              source_ids: list[str] | None = None,
              document_ids: list[str] | None = None) -> list[tuple[str, float]]:
        ...

    @abstractmethod
    def delete_by_source(self, source_id: str) -> None:
        ...

    @abstractmethod
    def delete_by_document(self, source_id: str, document_id: str) -> None:
        """按 (source_id, document_id) 删除向量。"""
        ...

    @abstractmethod
    def delete_entities_by_ids(self, entity_ids: list[str]) -> None:
        """按 entity_id 删除实体向量（用于清理孤儿实体）。"""
        ...

    @abstractmethod
    def get_embeddings(self, name: Collection, ids: list[str]) -> dict[str, list[float]]:
        """按 id 批量取已存向量，返回 {id: embedding}（仅含找到的）。"""
        ...

    # ---------------- 异步接口（默认用 to_thread 包装同步实现）----------------

    async def aadd(self, name: Collection, ids: list[str], texts: list[str],
                   embeddings: list[list[float]], metadatas: list[dict] | None = None) -> None:
        if not ids:
            return
        await asyncio.to_thread(self.add, name, ids, texts, embeddings, metadatas)

    async def aquery(self, name: Collection, query_embedding: list[float], top_k: int,
                     similarity_threshold: float = 0.0,
                     source_ids: list[str] | None = None,
                     document_ids: list[str] | None = None) -> list[tuple[str, float]]:
        return await asyncio.to_thread(
            self.query, name, query_embedding, top_k, similarity_threshold, source_ids, document_ids)

    async def adelete_by_source(self, source_id: str) -> None:
        await asyncio.to_thread(self.delete_by_source, source_id)

    async def adelete_by_document(self, source_id: str, document_id: str) -> None:
        await asyncio.to_thread(self.delete_by_document, source_id, document_id)

    async def adelete_entities_by_ids(self, entity_ids: list[str]) -> None:
        if not entity_ids:
            return
        await asyncio.to_thread(self.delete_entities_by_ids, entity_ids)

    async def aget_embeddings(self, name: Collection, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}
        return await asyncio.to_thread(self.get_embeddings, name, ids)

    # ---- 语义化便捷方法（同步）----

    def add_chunks(self, items: list[tuple[str, str, list[float]]],
                   source_id: str | None = None,
                   document_id: str | None = None) -> None:
        if items:
            metas = self._build_metas(len(items), source_id, document_id)
            self.add("chunks", [i[0] for i in items], [i[1] for i in items],
                     [i[2] for i in items], metas)

    def add_events(self, items: list[tuple[str, str, list[float]]],
                   source_id: str | None = None,
                   document_id: str | None = None) -> None:
        if items:
            metas = self._build_metas(len(items), source_id, document_id)
            self.add("event_titles", [i[0] for i in items], [i[1] for i in items],
                     [i[2] for i in items], metas)

    def add_event_contents(self, items: list[tuple[str, str, list[float]]],
                           source_id: str | None = None,
                           document_id: str | None = None) -> None:
        if items:
            metas = self._build_metas(len(items), source_id, document_id)
            self.add("event_contents", [i[0] for i in items], [i[1] for i in items],
                     [i[2] for i in items], metas)

    def add_event_summaries(self, items: list[tuple[str, str, list[float]]],
                            source_id: str | None = None,
                            document_id: str | None = None) -> None:
        if items:
            metas = self._build_metas(len(items), source_id, document_id)
            self.add("event_summaries", [i[0] for i in items], [i[1] for i in items],
                     [i[2] for i in items], metas)

    def add_entities(self, items: list[tuple[str, str, list[float]]]) -> None:
        if items:
            self.add("entities", [i[0] for i in items], [i[1] for i in items], [i[2] for i in items])

    def query_chunks(self, qe: list[float], top_k: int, similarity_threshold: float = 0.0,
                     source_ids: list[str] | None = None,
                     document_ids: list[str] | None = None) -> list[tuple[str, float]]:
        return self.query("chunks", qe, top_k, similarity_threshold, source_ids, document_ids)

    def query_event_titles(self, qe: list[float], top_k: int, similarity_threshold: float = 0.0,
                           source_ids: list[str] | None = None,
                           document_ids: list[str] | None = None) -> list[tuple[str, float]]:
        return self.query("event_titles", qe, top_k, similarity_threshold, source_ids, document_ids)

    def query_event_contents(self, qe: list[float], top_k: int, similarity_threshold: float = 0.0,
                             source_ids: list[str] | None = None,
                             document_ids: list[str] | None = None) -> list[tuple[str, float]]:
        return self.query("event_contents", qe, top_k, similarity_threshold, source_ids, document_ids)

    def query_event_summaries(self, qe: list[float], top_k: int, similarity_threshold: float = 0.0,
                              source_ids: list[str] | None = None,
                              document_ids: list[str] | None = None) -> list[tuple[str, float]]:
        return self.query("event_summaries", qe, top_k, similarity_threshold, source_ids, document_ids)

    def query_entities(self, qe: list[float], top_k: int, similarity_threshold: float = 0.0,
                       source_ids: list[str] | None = None,
                       document_ids: list[str] | None = None) -> list[tuple[str, float]]:
        return self.query("entities", qe, top_k, similarity_threshold, None, None)

    # ---- 语义化便捷方法（异步）----

    async def aadd_chunks(self, items: list[tuple[str, str, list[float]]],
                          source_id: str | None = None,
                          document_id: str | None = None) -> None:
        if items:
            metas = self._build_metas(len(items), source_id, document_id)
            await self.aadd("chunks", [i[0] for i in items], [i[1] for i in items],
                            [i[2] for i in items], metas)

    async def aadd_events(self, items: list[tuple[str, str, list[float]]],
                          source_id: str | None = None,
                          document_id: str | None = None) -> None:
        if items:
            metas = self._build_metas(len(items), source_id, document_id)
            await self.aadd("event_titles", [i[0] for i in items], [i[1] for i in items],
                            [i[2] for i in items], metas)

    async def aadd_event_contents(self, items: list[tuple[str, str, list[float]]],
                                  source_id: str | None = None,
                                  document_id: str | None = None) -> None:
        if items:
            metas = self._build_metas(len(items), source_id, document_id)
            await self.aadd("event_contents", [i[0] for i in items], [i[1] for i in items],
                            [i[2] for i in items], metas)

    async def aadd_event_summaries(self, items: list[tuple[str, str, list[float]]],
                                   source_id: str | None = None,
                                   document_id: str | None = None) -> None:
        if items:
            metas = self._build_metas(len(items), source_id, document_id)
            await self.aadd("event_summaries", [i[0] for i in items], [i[1] for i in items],
                            [i[2] for i in items], metas)

    async def aadd_entities(self, items: list[tuple[str, str, list[float]]]) -> None:
        if items:
            await self.aadd("entities", [i[0] for i in items], [i[1] for i in items],
                            [i[2] for i in items])

    async def aquery_chunks(self, qe: list[float], top_k: int, similarity_threshold: float = 0.0,
                            source_ids: list[str] | None = None,
                            document_ids: list[str] | None = None) -> list[tuple[str, float]]:
        return await self.aquery("chunks", qe, top_k, similarity_threshold, source_ids, document_ids)

    async def aquery_event_titles(self, qe: list[float], top_k: int, similarity_threshold: float = 0.0,
                                  source_ids: list[str] | None = None,
                                  document_ids: list[str] | None = None) -> list[tuple[str, float]]:
        return await self.aquery("event_titles", qe, top_k, similarity_threshold, source_ids, document_ids)

    async def aquery_event_contents(self, qe: list[float], top_k: int, similarity_threshold: float = 0.0,
                                    source_ids: list[str] | None = None,
                                    document_ids: list[str] | None = None) -> list[tuple[str, float]]:
        return await self.aquery("event_contents", qe, top_k, similarity_threshold, source_ids, document_ids)

    async def aquery_event_summaries(self, qe: list[float], top_k: int, similarity_threshold: float = 0.0,
                                     source_ids: list[str] | None = None,
                                     document_ids: list[str] | None = None) -> list[tuple[str, float]]:
        return await self.aquery("event_summaries", qe, top_k, similarity_threshold, source_ids, document_ids)

    async def aquery_entities(self, qe: list[float], top_k: int, similarity_threshold: float = 0.0,
                              source_ids: list[str] | None = None,
                              document_ids: list[str] | None = None) -> list[tuple[str, float]]:
        return await self.aquery("entities", qe, top_k, similarity_threshold, None, None)

    @staticmethod
    def _build_metas(count: int, source_id: str | None, document_id: str | None) -> list[dict] | None:
        """构造向量 metadata：同时写入 source_id 和 document_id。

        每项返回独立的 dict 实例（避免共享引用被下游意外修改）。
        """
        if not source_id and not document_id:
            return None
        result: list[dict] = []
        for _ in range(count):
            m: dict = {}
            if source_id:
                m["source_id"] = source_id
            if document_id:
                m["document_id"] = document_id
            result.append(m)
        return result