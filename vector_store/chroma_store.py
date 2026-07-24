"""ChromaDB 后端：基于 LlamaIndex ChromaVectorStore。"""
from __future__ import annotations

import chromadb
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.chroma import ChromaVectorStore

from ..base import Config
from ..base.logger import get_logger
from .base import BaseVectorStore, Collection

log = get_logger()


class ChromaVectorStoreBackend(BaseVectorStore):
    def __init__(self, cfg: Config) -> None:
        self._client = chromadb.PersistentClient(path=cfg.vector_store.chroma_path)
        self._stores: dict[Collection, ChromaVectorStore] = {}
        for name in ("chunks", "event_titles", "event_contents", "event_summaries", "entities"):
            collection = self._client.get_or_create_collection(name=name)
            self._stores[name] = ChromaVectorStore(chroma_collection=collection)

    def _store(self, name: Collection) -> ChromaVectorStore:
        return self._stores[name]

    def add(self, name: Collection, ids: list[str], texts: list[str],
            embeddings: list[list[float]], metadatas: list[dict] | None = None) -> None:
        if not ids:
            return
        
        # 1. 先对输入的 IDs 进行去重，确保本次添加内部没有重复
        seen = set()
        unique_indices = []
        for i, cid in enumerate(ids):
            if cid not in seen:
                seen.add(cid)
                unique_indices.append(i)
        
        # 2. 只保留唯一的记录
        unique_ids = [ids[i] for i in unique_indices]
        unique_texts = [texts[i] for i in unique_indices]
        unique_embs = [embeddings[i] for i in unique_indices]
        unique_metas = [(metadatas[i] if metadatas else {}) or {} for i in unique_indices] if metadatas else None
        
        if not unique_ids:
            return
        
        # 3. 再检查哪些 ID 已存在于向量库中，跳过重复的
        col = self._store(name)._collection
        existing = set()
        try:
            result = col.get(ids=unique_ids, include=[])
            existing = set(result.get("ids", []))
        except Exception:
            pass
        
        # 4. 构建最终要添加的节点
        nodes = []
        for i, (cid, text, emb) in enumerate(zip(unique_ids, unique_texts, unique_embs)):
            if cid in existing:
                continue
            meta = unique_metas[i] if unique_metas else {}
            nodes.append(TextNode(id_=cid, text=text, embedding=emb, metadata=meta))
        
        skipped_dup = len(ids) - len(unique_ids)
        skipped_existing = len(unique_ids) - len(nodes)
        skipped_total = skipped_dup + skipped_existing
        if nodes:
            self._store(name).add(nodes)
            log.info("向量写入 collection={} 传入={} 跳过(内部ID重复={}/已存在={})={} 实际写入={}",
                     name, len(ids), skipped_dup, skipped_existing, skipped_total, len(nodes))
        else:
            log.info("向量写入 collection={} 传入={} 跳过(内部ID重复={}/已存在={})={} 实际写入=0（全部已存在）",
                     name, len(ids), skipped_dup, skipped_existing, skipped_total)

    def query(self, name: Collection, query_embedding: list[float], top_k: int,
              similarity_threshold: float = 0.0,
              source_ids: list[str] | None = None,
              document_ids: list[str] | None = None) -> list[tuple[str, float]]:
        # entities 跨 source 共享，不按 source_id / document_id 过滤（P1-15 修复）
        if name != "entities":
            conditions = []
            if source_ids:
                conditions.append({"source_id": {"$in": source_ids}})
            if document_ids:
                conditions.append({"document_id": {"$in": document_ids}})
            if len(conditions) > 1:
                where = {"$and": conditions}
            elif len(conditions) == 1:
                where = conditions[0]
            else:
                where = None
        else:
            where = None
        vsq = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=top_k,
            mode="default",
        )
        kwargs = {"where": where} if where else {}
        result = self._store(name).query(vsq, **kwargs)
        hits: list[tuple[str, float]] = []
        if result.nodes and result.similarities is not None:
            for node, sim in zip(result.nodes, result.similarities):
                if sim >= similarity_threshold:
                    hits.append((node.node_id, float(sim)))
        source_filter = (source_ids is not None and name != "entities")
        doc_filter = (document_ids is not None and name != "entities")
        log.debug("向量查询 collection={} top_k={} 阈值={} 来源过滤={} 文档过滤={} 结果数={}",
                  name, top_k, similarity_threshold, source_filter, doc_filter, len(hits))
        return hits

    def delete_by_source(self, source_id: str) -> None:
        """按 source_id 删除 chunks/event_titles/event_contents/event_summaries 四个 collection 的向量。

        注意：entities collection 不按 source_id 删（实体跨 source 共享，去重），
        孤儿实体通过 delete_entities_by_ids 单独清理。
        """
        for name in ("chunks", "event_titles", "event_contents", "event_summaries"):
            try:
                col = self._store(name)._collection
                before = col.count()
                col.delete(where={"source_id": source_id})
                after = col.count()
                log.info("向量删除 collection={} source_id={} 删除前={} 删除后={}",
                         name, source_id, before, after)
            except Exception as e:
                log.warning("向量删除失败 collection={} source_id={} err={}", name, source_id, e)

    def delete_by_document(self, source_id: str, document_id: str) -> None:
        """按 (source_id, document_id) 删除四个 collection 的向量。

        entities collection 不按 document_id 删（实体跨 document 共享）。
        """
        where = {"$and": [{"source_id": source_id}, {"document_id": document_id}]}
        for name in ("chunks", "event_titles", "event_contents", "event_summaries"):
            try:
                col = self._store(name)._collection
                before = col.count()
                col.delete(where=where)
                after = col.count()
                log.info("向量删除 collection={} source_id={} document_id={} 删除前={} 删除后={}",
                         name, source_id, document_id, before, after)
            except Exception as e:
                log.warning("向量删除失败 collection={} source_id={} document_id={} err={}",
                            name, source_id, document_id, e)

    def delete_entities_by_ids(self, entity_ids: list[str]) -> None:
        if not entity_ids:
            return
        col = self._store("entities")._collection
        before = col.count()
        col.delete(ids=entity_ids)
        after = col.count()
        log.info("实体向量删除 ids数量={} 删除前={} 删除后={}", len(entity_ids), before, after)

    def get_embeddings(self, name: Collection, ids: list[str]) -> dict[str, list[float]]:
        if not ids:
            return {}
        col = self._store(name)._collection
        result = col.get(ids=ids, include=["embeddings"])
        emb_list = result.get("embeddings")
        if emb_list is None:
            emb_list = []
        # 确保 embedding 是 list[float]，ChromaDB 可能返回 numpy 数组
        return {id_: list(emb) for id_, emb in zip(result.get("ids", []), emb_list)
                if emb is not None}