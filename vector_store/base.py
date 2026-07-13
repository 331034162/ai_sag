"""向量库抽象基类：定义统一接口，便于切换 ChromaDB/Milvus/FAISS。

同步方法供入库等同步场景使用；
异步方法（a 前缀）供异步检索流程使用，默认用 asyncio.to_thread 包装同步实现。
子类若有原生异步能力（如 Milvus async）可重写异步方法。
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
              source_ids: list[str] | None = None) -> list[tuple[str, float]]:
        ...

    @abstractmethod
    def delete_by_source(self, source_id: str) -> None:
        ...

    @abstractmethod
    def delete_entities_by_ids(self, entity_ids: list[str]) -> None:
        """按 entity_id 删除实体向量（用于清理孤儿实体）。"""
        ...

    @abstractmethod
    def delete_event_ids(self, event_ids: list[str]) -> None:
        """按 event_id 删除 event_titles / event_contents 向量（硬删除软删事件）。"""
        ...

    @abstractmethod
    def get_embeddings(self, name: Collection, ids: list[str]) -> dict[str, list[float]]:
        """按 id 批量取已存向量，返回 {id: embedding}（仅含找到的）。"""
        ...

    @abstractmethod
    def list_source_ids(self) -> list[str]:
        """列出向量库中所有已知的 source_id（从 chunks collection 的 metadata 提取）。"""
        ...

    @abstractmethod
    def list_all_entity_ids(self) -> list[str]:
        """列出 entities collection 中所有 entity_id（供对账清理孤儿实体向量）。"""
        ...

    # ---------------- 异步接口（默认用 to_thread 包装同步实现）----------------

    async def aadd(self, name: Collection, ids: list[str], texts: list[str],
                   embeddings: list[list[float]], metadatas: list[dict] | None = None) -> None:
        if not ids:
            return
        await asyncio.to_thread(self.add, name, ids, texts, embeddings, metadatas)

    async def aquery(self, name: Collection, query_embedding: list[float], top_k: int,
                     similarity_threshold: float = 0.0,
                     source_ids: list[str] | None = None) -> list[tuple[str, float]]:
        return await asyncio.to_thread(
            self.query, name, query_embedding, top_k, similarity_threshold, source_ids)

    async def adelete_by_source(self, source_id: str) -> None:
        await asyncio.to_thread(self.delete_by_source, source_id)

    async def adelete_entities_by_ids(self, entity_ids: list[str]) -> None:
        if not entity_ids:
            return
        await asyncio.to_thread(self.delete_entities_by_ids, entity_ids)

    async def adelete_event_ids(self, event_ids: list[str]) -> None:
        if not event_ids:
            return
        await asyncio.to_thread(self.delete_event_ids, event_ids)

    async def aget_embeddings(self, name: Collection, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}
        return await asyncio.to_thread(self.get_embeddings, name, ids)

    async def alist_source_ids(self) -> list[str]:
        return await asyncio.to_thread(self.list_source_ids)

    async def alist_all_entity_ids(self) -> list[str]:
        return await asyncio.to_thread(self.list_all_entity_ids)

    # ---- 语义化便捷方法（同步）----

    def add_chunks(self, items: list[tuple[str, str, list[float]]],
                   source_id: str | None = None) -> None:
        if items:
            metas = [{"source_id": source_id} for _ in items] if source_id else None
            self.add("chunks", [i[0] for i in items], [i[1] for i in items],
                     [i[2] for i in items], metas)

    def add_events(self, items: list[tuple[str, str, list[float]]],
                   source_id: str | None = None) -> None:
        if items:
            metas = [{"source_id": source_id} for _ in items] if source_id else None
            self.add("event_titles", [i[0] for i in items], [i[1] for i in items],
                     [i[2] for i in items], metas)

    def add_event_contents(self, items: list[tuple[str, str, list[float]]],
                           source_id: str | None = None) -> None:
        if items:
            metas = [{"source_id": source_id} for _ in items] if source_id else None
            self.add("event_contents", [i[0] for i in items], [i[1] for i in items],
                     [i[2] for i in items], metas)

    def add_event_summaries(self, items: list[tuple[str, str, list[float]]],
                            source_id: str | None = None) -> None:
        if items:
            metas = [{"source_id": source_id} for _ in items] if source_id else None
            self.add("event_summaries", [i[0] for i in items], [i[1] for i in items],
                     [i[2] for i in items], metas)

    def add_entities(self, items: list[tuple[str, str, list[float]]]) -> None:
        if items:
            self.add("entities", [i[0] for i in items], [i[1] for i in items], [i[2] for i in items])

    def query_chunks(self, qe: list[float], top_k: int, similarity_threshold: float = 0.0,
                     source_ids: list[str] | None = None) -> list[tuple[str, float]]:
        return self.query("chunks", qe, top_k, similarity_threshold, source_ids)

    def query_event_titles(self, qe: list[float], top_k: int, similarity_threshold: float = 0.0,
                           source_ids: list[str] | None = None) -> list[tuple[str, float]]:
        return self.query("event_titles", qe, top_k, similarity_threshold, source_ids)

    def query_event_contents(self, qe: list[float], top_k: int, similarity_threshold: float = 0.0,
                             source_ids: list[str] | None = None) -> list[tuple[str, float]]:
        return self.query("event_contents", qe, top_k, similarity_threshold, source_ids)

    def query_event_summaries(self, qe: list[float], top_k: int, similarity_threshold: float = 0.0,
                              source_ids: list[str] | None = None) -> list[tuple[str, float]]:
        return self.query("event_summaries", qe, top_k, similarity_threshold, source_ids)

    def query_entities(self, qe: list[float], top_k: int, similarity_threshold: float = 0.0,
                       source_ids: list[str] | None = None) -> list[tuple[str, float]]:
        return self.query("entities", qe, top_k, similarity_threshold, None)

    # ---- 语义化便捷方法（异步）----

    async def aadd_chunks(self, items: list[tuple[str, str, list[float]]],
                          source_id: str | None = None) -> None:
        if items:
            metas = [{"source_id": source_id} for _ in items] if source_id else None
            await self.aadd("chunks", [i[0] for i in items], [i[1] for i in items],
                            [i[2] for i in items], metas)

    async def aadd_events(self, items: list[tuple[str, str, list[float]]],
                          source_id: str | None = None) -> None:
        if items:
            metas = [{"source_id": source_id} for _ in items] if source_id else None
            await self.aadd("event_titles", [i[0] for i in items], [i[1] for i in items],
                            [i[2] for i in items], metas)

    async def aadd_event_contents(self, items: list[tuple[str, str, list[float]]],
                                  source_id: str | None = None) -> None:
        if items:
            metas = [{"source_id": source_id} for _ in items] if source_id else None
            await self.aadd("event_contents", [i[0] for i in items], [i[1] for i in items],
                            [i[2] for i in items], metas)

    async def aadd_event_summaries(self, items: list[tuple[str, str, list[float]]],
                                   source_id: str | None = None) -> None:
        if items:
            metas = [{"source_id": source_id} for _ in items] if source_id else None
            await self.aadd("event_summaries", [i[0] for i in items], [i[1] for i in items],
                            [i[2] for i in items], metas)

    async def aadd_entities(self, items: list[tuple[str, str, list[float]]]) -> None:
        if items:
            await self.aadd("entities", [i[0] for i in items], [i[1] for i in items],
                            [i[2] for i in items])

    async def aquery_chunks(self, qe: list[float], top_k: int, similarity_threshold: float = 0.0,
                            source_ids: list[str] | None = None) -> list[tuple[str, float]]:
        return await self.aquery("chunks", qe, top_k, similarity_threshold, source_ids)

    async def aquery_event_titles(self, qe: list[float], top_k: int, similarity_threshold: float = 0.0,
                                  source_ids: list[str] | None = None) -> list[tuple[str, float]]:
        return await self.aquery("event_titles", qe, top_k, similarity_threshold, source_ids)

    async def aquery_event_contents(self, qe: list[float], top_k: int, similarity_threshold: float = 0.0,
                                    source_ids: list[str] | None = None) -> list[tuple[str, float]]:
        return await self.aquery("event_contents", qe, top_k, similarity_threshold, source_ids)

    async def aquery_event_summaries(self, qe: list[float], top_k: int, similarity_threshold: float = 0.0,
                                     source_ids: list[str] | None = None) -> list[tuple[str, float]]:
        return await self.aquery("event_summaries", qe, top_k, similarity_threshold, source_ids)

    async def aquery_entities(self, qe: list[float], top_k: int, similarity_threshold: float = 0.0,
                              source_ids: list[str] | None = None) -> list[tuple[str, float]]:
        return await self.aquery("entities", qe, top_k, similarity_threshold, None)
