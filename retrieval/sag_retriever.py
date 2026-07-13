"""SAG 异步检索器：实现论文流程图的完整检索逻辑。

全链路异步：MySQL 用 aiomysql，LLM 用 LlamaIndex 原生异步方法，
Embedder/VectorStore 用异步接口（a 前缀方法）。
"""
from __future__ import annotations

import asyncio
import re
from typing import Literal

from llama_index.core.llms import ChatMessage, LLM, MessageRole
from llama_index.core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..base import (
    Config,
    QUERY_REWRITE_SYSTEM_PROMPT,
    QUERY_REWRITE_USER_PROMPT,
    RERANK_CANDIDATE_FORMAT,
    RERANK_PROMPT_TEMPLATE,
    RERANK_SYSTEM_PROMPT,
    RetrievedSection,
    SearchResult,
    SearchTrace,
    query_extract_system_prompt,
)
from ..base.logger import get_logger
from ..embeddings import BaseEmbedder
from ..storage import MysqlStore
from ..vector_store import BaseVectorStore

log = get_logger()

FusionMode = Literal["supplement", "concat"]


class _QueryEntities(BaseModel):
    entities: list[str] = Field(default_factory=list, description="从查询中抽取的关键实体名")


class _RerankOrder(BaseModel):
    order: list[int] = Field(default_factory=list, description="按相关性降序排列的事件编号列表")


class SagRetriever:
    def __init__(self, cfg: Config, db: MysqlStore, vectors: BaseVectorStore,
                 embedder: BaseEmbedder, llm: LLM) -> None:
        self.cfg = cfg
        self.db = db
        self.vectors = vectors
        self.embedder = embedder
        self.llm = llm

    async def search(self, query: str, source_ids: list[str] | None = None, *,
                     fusion: FusionMode | None = None,
                     history: list[dict] | None = None) -> SearchResult:
        fusion = fusion or self.cfg.search.fusion
        trace = SearchTrace(query=query)

        if not query or not query.strip():
            return SearchResult(sections=[], trace=trace)

        # 0. 多轮对话时，结合历史重写查询
        rewritten = await self._rewrite_query(query, history or [])
        if rewritten and rewritten != query:
            log.info("查询重写 raw={!r} -> rewritten={!r}", query, rewritten)
        eff_query = rewritten or query

        # 用重写后的 query 生成向量
        query_vec = await self.embedder.aembed_text(eff_query)

        # 1. 提取查询实体 + 3. 种子事件向量召回（并发执行，无依赖关系）
        query_entities_task = self._extract_query_entities(eff_query)
        seed_recall = self.cfg.search.seed_recall

        if seed_recall == "mixed":
            # 双路并发：title + summary，合并去重取高分数
            title_task = self.vectors.aquery_event_titles(
                query_vec, top_k=self.cfg.search.max_events,
                similarity_threshold=self.cfg.search.similarity_threshold,
                source_ids=source_ids,
            )
            summary_task = self.vectors.aquery_event_summaries(
                query_vec, top_k=self.cfg.search.max_events,
                similarity_threshold=self.cfg.search.similarity_threshold,
                source_ids=source_ids,
            )
            query_entities, title_hits, summary_hits = await asyncio.gather(
                query_entities_task, title_task, summary_task)
            # 合并：去重保留 max 分数，按分降序
            merged: dict[str, float] = {}
            for eid, score in title_hits:
                merged[eid] = score
            for eid, score in summary_hits:
                merged[eid] = max(merged.get(eid, 0.0), score)
            path_b_hits = sorted(merged.items(), key=lambda x: x[1], reverse=True)
            log.info("种子召回策略=mixed title={} summary={} 合并={}",
                     len(title_hits), len(summary_hits), len(path_b_hits))
        elif seed_recall == "summary":
            summary_task = self.vectors.aquery_event_summaries(
                query_vec, top_k=self.cfg.search.max_events,
                similarity_threshold=self.cfg.search.similarity_threshold,
                source_ids=source_ids,
            )
            query_entities, path_b_hits = await asyncio.gather(query_entities_task, summary_task)
            log.info("种子召回策略=summary 结果数={}", len(path_b_hits))
        else:
            # 默认 title 策略（兼容旧版）
            title_task = self.vectors.aquery_event_titles(
                query_vec, top_k=self.cfg.search.max_events,
                similarity_threshold=self.cfg.search.similarity_threshold,
                source_ids=source_ids,
            )
            query_entities, path_b_hits = await asyncio.gather(query_entities_task, title_task)
            log.info("种子召回策略=title 结果数={}", len(path_b_hits))

        trace.query_entities = query_entities
        seed_vector_event_ids = [h[0] for h in path_b_hits]
        # Path B 种子召回已有分数，复用避免粗排阶段重复拉 embedding 重算
        path_b_scores: dict[str, float] = {h[0]: h[1] for h in path_b_hits}

        # 2. 分支1：实体召回事件（依赖步骤1的实体抽取结果）
        entity_ids: list[str] = []
        if query_entities:
            entity_ids = await self._recall_entities(query_entities, source_ids)
        trace.expanded_query_entities = entity_ids  # 记录 Ûq

        entity_event_ids = (await self.db.get_event_ids_by_entity_ids(entity_ids, source_ids)
                            if entity_ids else [])
        log.info("实体→事件联表查询 实体数={} 命中事件数={}", len(entity_ids), len(entity_event_ids))

        # 4. 合并事件池
        seed_ids = list(dict.fromkeys(entity_event_ids + seed_vector_event_ids))
        trace.seed_event_ids = seed_ids

        if not seed_ids:
            return await self._fallback(query_vec, source_ids, trace, fusion)

        # 5. 多跳 BFS 扩展（优化 3：缓存事件供精排复用）
        event_cache: dict[str, object] = {}
        expanded_ids = await self._expand(seed_ids, entity_ids, source_ids,
                                          query_vec, event_cache=event_cache)
        trace.expanded_event_ids = expanded_ids

        # 6. 粗排（复用 Path B 已有分数，省去重复拉 embedding）
        ranked_ids, coarse_scores = await self._coarse_rank(
            expanded_ids, query_vec, source_ids, pre_scores=path_b_scores)
        # 兜底：如果所有事件都没有向量（极端情况），保留扩展集前 N 个
        if not ranked_ids and expanded_ids:
            log.warning("粗排无有效分数，兜底使用扩展集前 N 个事件")
            ranked_ids = expanded_ids[: self.cfg.search.max_events]
            coarse_scores = {eid: 0.0 for eid in ranked_ids}
        top_n = ranked_ids[: self.cfg.search.max_events]

        # 7. LLM 重排
        rerank_input = top_n[: self.cfg.search.rerank_candidate_limit]
        trace.rerank_candidate_ids = rerank_input  # 记录 Ê（精排输入）
        # Bug 2 + 优化 3：透传粗排分数与事件缓存
        reranked_ids = await self._rerank(eff_query, rerank_input,
                                          query_entity_ids=entity_ids, source_ids=source_ids,
                                          coarse_scores=coarse_scores, events_cache=event_cache)
        trace.reranked_ids = reranked_ids

        # 8. 事件回取切片 ChunkA（用粗排真实分数替换假分数）
        sections_a = await self._sections_for_events(
            reranked_ids, source_ids, coarse_scores=coarse_scores)

        # 双路融合（论文设计：A路/B路各自独立配额，合并去重后给 LLM）
        # max_sections 语义 = 每路配额；A路5 + B路5 去重后最多 2×max_sections 个
        sections_b: list[RetrievedSection] = []
        per_path = self.cfg.search.max_sections
        if fusion == "concat":
            # A路（事件路）：精排结果，最多 per_path 个
            sections_a = sections_a[:per_path]
            a_count = len(sections_a)
            # Bug修复：A路因多事件映射同一chunk或chunk_id为空导致不足per_path时，
            # B路多取差额补足，保证最终给LLM的信息量不缩水
            b_need = per_path if a_count >= per_path else per_path + (per_path - a_count)
            sections_b = await self._vector_sections(query_vec, source_ids, b_need)
            # 合并去重，上限 2×per_path（A优先，B补足，去重）
            sections = self._merge_dedupe(sections_a, sections_b, per_path * 2)
        else:
            # supplement 模式：A路优先占满 per_path，不足时 B路补足到 per_path
            sections = sections_a[:per_path]
            if len(sections) < per_path:
                await self._supplement(sections, query_vec, source_ids, per_path)

        # 填充 source_name，供 LLM 生成答案时引用可读来源
        sections = await self._attach_source_names(sections)

        # 诊断日志：记录最终片段的来源路与标题，便于排查"片段未被引用"
        def _sec_brief(sections: list[RetrievedSection]) -> list[dict]:
            result = []
            for s in sections:
                c = s.content or ""
                result.append({"rank": s.rank, "score": round(s.score, 3),
                               "heading": (s.heading or "")[:30],
                               "content": c[:50] + ("..." if len(c) > 50 else "")})
            return result
        log.info("最终片段 query={!r} fusion={} 最终={} A路={} B路={} A路详情={} B路详情={}",
                 trace.query, fusion, len(sections),
                 len(sections_a[:per_path]), len(sections_b),
                 _sec_brief(sections_a[:per_path]),
                 _sec_brief(sections_b))
        return SearchResult(sections=sections, trace=trace)

    # ---------------- 步骤实现 ----------------

    async def _rewrite_query(self, query: str, history: list[dict]) -> str:
        if not history:
            return query
        history_text = self._format_history(history, self.cfg.search.rewrite_max_rounds)
        if not history_text:
            return query
        try:
            return await self._llm_rewrite(history_text, query)
        except Exception as e:
            log.warning("查询重写失败，回退原 query err={}", e)
            return query

    async def _llm_rewrite(self, history_text: str, query: str) -> str:
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=QUERY_REWRITE_SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER,
                        content=QUERY_REWRITE_USER_PROMPT.format(history=history_text, query=query)),
        ]
        resp = await self.llm.achat(messages)
        text = str(resp).strip()
        text = text.strip("\"'""''")
        for prefix in ("重写后：", "重写后:", "重写：", "重写:",
                       "assistant:", "assistant：", "Assistant:", "Assistant：",
                       "user:", "user：", "User:", "User：",
                       "助手：", "助手:", "用户：", "用户:"):
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        return text or query

    @staticmethod
    def _format_history(history: list[dict], max_rounds: int) -> str:
        valid = [h for h in history
                 if h.get("role") in ("user", "assistant") and h.get("content", "").strip()]
        max_msgs = max_rounds * 2
        if len(valid) > max_msgs:
            valid = valid[-max_msgs:]
        if not valid:
            return ""
        role_label = {"user": "用户", "assistant": "助手"}
        return "\n".join(f"{role_label[h['role']]}：{h['content']}" for h in valid)

    async def _extract_query_entities(self, query: str) -> list[str]:
        try:
            prompt = ChatPromptTemplate(message_templates=[
                ChatMessage(role=MessageRole.SYSTEM, content=query_extract_system_prompt()),
                ChatMessage(role=MessageRole.USER, content="查询：{query}\n请返回实体名列表。"),
            ])
            result = await self.llm.astructured_predict(
                _QueryEntities,
                prompt,
                query=query,
            )
            entities = [e.strip() for e in result.entities if e.strip()]
            log.info("实体抽取 query={!r} entities={}", query[:80], entities)
            return entities
        except Exception as e:
            log.warning("实体抽取失败 query={} error={}", query[:50], str(e), exc_info=True)
            return []

    async def _recall_entities(self, names: list[str],
                               source_ids: list[str] | None) -> list[str]:
        exact = await self.db.search_entities_by_name(names, source_ids)
        exact_ids = [e.id for e in exact]
        vec_ids: list[str] = []
        if names:
            name_embs = await self.embedder.aembed_texts(names)
            # 并发查询每个实体名的向量召回（P1 修复：串行→并发）
            hit_lists = await asyncio.gather(*[
                self.vectors.aquery_entities(
                    name_emb, top_k=5,
                    similarity_threshold=self.cfg.search.entity_expand_threshold,
                )
                for name_emb in name_embs
            ])
            for hits in hit_lists:
                vec_ids.extend(h[0] for h in hits)
        candidate_ids = list(dict.fromkeys(exact_ids + vec_ids))
        if not candidate_ids:
            return []
        return await self.db.filter_entity_ids_by_sources(candidate_ids, source_ids)

    async def _expand(self, seed_ids: list[str], initial_entity_ids: list[str],
                      source_ids: list[str] | None,
                      query_vec: list[float],
                      event_cache: dict[str, object] | None = None) -> list[str]:
        """沿 event_entities 超边做 BFS 多跳扩展。

        子策略：
        - multi：固定跳数扩展，收集全部可达新事件。
        - hopllm：每跳用向量相似度对新事件重排，仅保留 topK 作为下一跳种子，
          且当最高相似度低于阈值时动态停止，避免引入噪声。
        """
        tracked_events = set(seed_ids)
        tracked_entities = set(initial_entity_ids)
        current_event_ids = list(seed_ids)

        log.info("BFS启动 种子事件={} 初始实体={} 最大跳数={} 子策略={}",
                 len(seed_ids), len(initial_entity_ids), self.cfg.search.max_hops,
                 self.cfg.search.sub_strategy)

        for hop in range(self.cfg.search.max_hops):
            if not current_event_ids:
                log.info("BFS第{}跳 无当前事件，提前终止", hop)
                break
            events = await self.db.get_events_by_ids(current_event_ids, source_ids)
            # 优化 3：缓存事件供精排复用
            if event_cache is not None:
                for ev in events:
                    event_cache[ev.id] = ev
            current_entity_ids = set()
            for ev in events:
                current_entity_ids.update(ev.entity_ids)
            new_entities = [eid for eid in current_entity_ids if eid not in tracked_entities]
            if not new_entities:
                break
            # 论文 Section 4.4：entity frontier pruning budget=50，限制每跳边界实体数
            budget = self.cfg.search.entity_frontier_budget
            if len(new_entities) > budget:
                new_entities = new_entities[:budget]
            tracked_entities.update(new_entities)
            new_event_ids = await self.db.get_event_ids_by_entity_ids(
                new_entities, source_ids, exclude=list(tracked_events))
            log.info("BFS第{}跳 实体→事件联表 新增实体={} 新事件={}", hop, len(new_entities), len(new_event_ids))
            if not new_event_ids:
                log.info("BFS第{}跳 无新事件，终止扩展", hop)
                break
            tracked_events.update(new_event_ids)

            if self.cfg.search.sub_strategy == "hopllm":
                emb_map = await self.vectors.aget_embeddings("event_contents", new_event_ids)
                scored = self._cosine_scores(query_vec, new_event_ids, emb_map)
                scored.sort(key=lambda x: x[1], reverse=True)
                if not scored:
                    log.info("BFS第{}跳 hopllm评分无结果，终止", hop)
                    break
                best_score = scored[0][1]
                if best_score < self.cfg.search.hop_relevance_threshold:
                    log.info("BFS第{}跳 hopllm最佳分={:.4f}<阈值={}，动态停止",
                             hop, best_score, self.cfg.search.hop_relevance_threshold)
                    break  # 动态停止：新跳事件与查询已不相关
                current_event_ids = [eid for eid, _ in scored[: self.cfg.search.hop_seed_topk]]
                log.info("BFS第{}跳 hopllm重排 候选={} 取top{} 最佳分={:.4f}",
                         hop, len(scored), len(current_event_ids), best_score)
            else:
                current_event_ids = new_event_ids

        log.info("BFS结束 总事件数={} (种子={})", len(tracked_events), len(seed_ids))
        return list(tracked_events)

    async def _coarse_rank(self, event_ids: list[str], query_vec: list[float],
                           source_ids: list[str] | None = None,
                           pre_scores: dict[str, float] | None = None,
                           ) -> tuple[list[str], dict[str, float]]:
        """粗排：按 query 向量与 event content 向量的余弦相似度排序 + 阈值过滤。

        论文 Fig 2 标注 "Similarity Threshold Filtering"：
        排序后过滤低于 coarse_threshold 的事件，再截断到 top max_events。
        阈值默认 0.15，设为 0 则只排序截断不做阈值过滤。

        pre_scores：Path B 种子召回已算出的分数，复用避免重复拉 embedding。
        """
        if not event_ids:
            return [], {}
        pre_scores = pre_scores or {}
        # 拆分：已有分数的事件直接复用，没有的才拉 embedding 计算
        pre_scored = [(eid, pre_scores[eid]) for eid in event_ids if eid in pre_scores]
        new_ids = [eid for eid in event_ids if eid not in pre_scores]
        if new_ids:
            emb_map = await self.vectors.aget_embeddings("event_contents", new_ids)
            new_scored = self._cosine_scores(query_vec, new_ids, emb_map)
        else:
            new_scored = []
        scored = pre_scored + new_scored
        # 过滤掉完全无向量的事件（分数为0），避免纯噪声进入精排
        scored = [(eid, s) for eid, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        # 论文 Fig 2：Similarity Threshold Filtering
        threshold = self.cfg.search.coarse_threshold
        if threshold > 0:
            scored = [(eid, s) for eid, s in scored if s >= threshold]
        scored = scored[: self.cfg.search.max_events]
        score_map = {eid: s for eid, s in scored}
        return [eid for eid, _ in scored], score_map

    @staticmethod
    def _cosine_scores(query_vec: list[float], ids: list[str],
                       emb_map: dict[str, list[float]]) -> list[tuple[str, float]]:
        # 优化 1：改为 numpy 矩阵运算，O(N) → 一次矩阵乘
        import numpy as np
        valid_ids = [eid for eid in ids if eid in emb_map]
        if not valid_ids:
            return [(eid, 0.0) for eid in ids]
        emb_matrix = np.asarray([emb_map[eid] for eid in valid_ids], dtype=np.float32)
        q = np.asarray(query_vec, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm <= 0:
            return [(eid, 0.0) for eid in ids]
        emb_norms = np.linalg.norm(emb_matrix, axis=1)
        dots = emb_matrix @ q
        denom = q_norm * emb_norms + 1e-8
        scores = np.where(emb_norms > 0, dots / denom, 0.0)
        score_dict = dict(zip(valid_ids, scores.tolist()))
        return [(eid, score_dict.get(eid, 0.0)) for eid in ids]

    async def _rerank(self, query: str, event_ids: list[str], *,
                      query_entity_ids: list[str] | None = None,
                      source_ids: list[str] | None = None,
                      coarse_scores: dict[str, float] | None = None,
                      events_cache: dict[str, object] | None = None) -> list[str]:
        if not event_ids:
            return []
        # 优化 3：优先复用多跳扩展阶段缓存的事件，缺失的再查
        events: list = []
        if events_cache:
            missing = []
            for eid in event_ids:
                ev = events_cache.get(eid)
                if ev is not None:
                    events.append(ev)
                else:
                    missing.append(eid)
            if missing:
                missing_events = await self.db.get_events_by_ids(missing, source_ids)
                for ev in missing_events:
                    events_cache[ev.id] = ev
                events.extend(missing_events)
            # 按 event_ids 顺序重排
            ev_map = {ev.id: ev for ev in events}
            events = [ev_map[eid] for eid in event_ids if eid in ev_map]
        else:
            events = await self.db.get_events_by_ids(event_ids, source_ids)
        if not events:
            return []
        try:
            selected = await self._llm_rerank(query, events, query_entity_ids=query_entity_ids or [])
            result = selected[: self.cfg.search.rerank_top_k]
        except Exception:
            # Bug 2 修复：精排失败时按粗排分数降级排序，而非丢顺序
            log.warning("LLM精排失败，降级使用粗排分数排序")
            if coarse_scores:
                sorted_ids = sorted(event_ids,
                                    key=lambda eid: coarse_scores.get(eid, 0.0),
                                    reverse=True)
                result = sorted_ids[: self.cfg.search.rerank_top_k]
            else:
                result = event_ids[: self.cfg.search.rerank_top_k]

        # 精排结果日志
        ev_map = {ev.id: ev for ev in events}
        brief: list[dict] = []
        for eid in result:
            ev = ev_map.get(eid)
            if ev:
                summary_snip = (ev.summary or "")[:80]
                brief.append({"title": (ev.title or "")[:60],
                              "summary": summary_snip + ("..." if len(ev.summary or "") > 80 else ""),
                              "coarse_score": round(coarse_scores.get(eid, 0.0), 4) if coarse_scores else None})
        log.info("LLM精排结果 输入={} 输出={} 事件={}",
                 len(event_ids), len(result), brief)
        return result

    async def _llm_rerank(self, query: str, events: list, *,
                          query_entity_ids: list[str]) -> list[str]:
        query_entity_set = set(query_entity_ids)
        entity_names = (await self.db.get_entity_names_by_ids(query_entity_ids)
                        if query_entity_ids else {})

        lines = []
        for i, ev in enumerate(events):
            roles_text = self._format_entity_roles(ev, query_entity_set, entity_names)
            lines.append(RERANK_CANDIDATE_FORMAT.format(
                i=i, event_id=ev.id, title=ev.title, summary=ev.summary,
                roles=roles_text,
            ))
        candidates = "\n\n".join(lines)
        try:
            prompt = ChatPromptTemplate(message_templates=[
                ChatMessage(role=MessageRole.SYSTEM, content=RERANK_SYSTEM_PROMPT),
                ChatMessage(role=MessageRole.USER, content=RERANK_PROMPT_TEMPLATE),
            ])
            rerank_result = await self.llm.astructured_predict(
                _RerankOrder,
                prompt,
                query=query,
                candidates=candidates,
            )
            order = rerank_result.order
        except Exception:
            fallback = RERANK_SYSTEM_PROMPT + "\n\n" + RERANK_PROMPT_TEMPLATE.format(query=query, candidates=candidates)
            resp = await self.llm.acomplete(fallback)
            text = str(resp).strip()
            first_line = text.split("\n")[0].strip()
            order = [int(x) for x in re.findall(r"\d+", first_line)]
        seen = set()
        result = []
        for idx in order:
            if 0 <= idx < len(events) and events[idx].id not in seen:
                seen.add(events[idx].id)
                result.append(events[idx].id)
        for ev in events:
            if ev.id not in seen:
                result.append(ev.id)
        return result

    @staticmethod
    def _format_entity_roles(ev, query_entity_set: set[str],
                             entity_names: dict[str, str]) -> str:
        if not query_entity_set or not ev.entity_roles:
            return ""
        hit_roles = []
        for eid, role in ev.entity_roles.items():
            if eid in query_entity_set:
                name = entity_names.get(eid, eid[:8])
                if role:
                    hit_roles.append(f"{name}({role})")
                else:
                    hit_roles.append(f"{name}(角色未标注)")
        if not hit_roles:
            return ""
        total = len(query_entity_set)
        hit_count = sum(1 for eid in ev.entity_roles if eid in query_entity_set)
        return f"命中查询实体：{'、'.join(hit_roles)}  [命中 {hit_count}/{total}]"

    async def _sections_for_events(self, event_ids: list[str],
                                   source_ids: list[str] | None,
                                   coarse_scores: dict[str, float] | None = None,
                                   ) -> list[RetrievedSection]:
        if not event_ids:
            return []
        ev_chunk_map = await self.db.get_chunk_ids_by_event_ids(event_ids)
        chunk_ids = [ev_chunk_map[eid] for eid in event_ids if eid in ev_chunk_map]
        if not chunk_ids:
            return []
        chunks = await self.db.get_chunks_by_ids(chunk_ids)
        chunk_map = {c.id: c for c in chunks}
        sections = []
        for eid in event_ids:
            cid = ev_chunk_map.get(eid)
            if cid is None:
                continue
            c = chunk_map.get(cid)
            if c is None:
                continue
            sections.append(RetrievedSection(
                chunk_id=c.id, source_id=c.source_id, document_id=c.document_id,
                heading=c.heading, content=c.content,
                rank=len(sections),
                score=float((coarse_scores or {}).get(eid, 0.0)),
            ))
        return sections

    # ---------------- 双路融合 ----------------

    async def _fetch_vector_chunks(self, query_vec: list[float], source_ids: list[str] | None,
                                  top_k: int,
                                  existing_chunk_ids: set[str] | None = None,
                                  ) -> list[RetrievedSection]:
        # 优化 2 + Bug 3：抽取共享查询逻辑，并删除 source_ids 重复过滤
        hits = await self.vectors.aquery_chunks(query_vec, top_k=top_k * 2,
                                                similarity_threshold=self.cfg.search.similarity_threshold,
                                                source_ids=source_ids)
        if not hits:
            return []
        existing = existing_chunk_ids or set()
        chunk_ids = [h[0] for h in hits if h[0] not in existing]
        if not chunk_ids:
            return []
        chunks = await self.db.get_chunks_by_ids(chunk_ids)
        chunk_map = {c.id: c for c in chunks}
        sections = []
        for rank, (cid, score) in enumerate(hits):
            c = chunk_map.get(cid)
            if c is None or c.id in existing:
                continue
            sections.append(RetrievedSection(
                chunk_id=c.id, source_id=c.source_id, document_id=c.document_id,
                heading=c.heading, content=c.content, rank=rank, score=float(score),
            ))
            if len(sections) >= top_k:
                break
        return sections

    async def _vector_sections(self, query_vec: list[float], source_ids: list[str] | None,
                               top_k: int) -> list[RetrievedSection]:
        return await self._fetch_vector_chunks(query_vec, source_ids, top_k)

    async def _supplement(self, sections: list[RetrievedSection], query_vec: list[float],
                          source_ids: list[str] | None, max_sections: int) -> None:
        if len(sections) >= max_sections:
            return
        existing = {s.chunk_id for s in sections}
        need = max_sections - len(sections)
        new_sections = await self._fetch_vector_chunks(
            query_vec, source_ids, need, existing_chunk_ids=existing)
        for s in new_sections:
            s.rank = len(sections)
            sections.append(s)
            if len(sections) >= max_sections:
                break

    @staticmethod
    def _merge_dedupe(a: list[RetrievedSection], b: list[RetrievedSection],
                      max_sections: int) -> list[RetrievedSection]:
        # 保留事件路优先顺序（精排结果质量更高），仅去重和截断
        seen = set()
        merged = []
        for s in a + b:
            if s.chunk_id in seen:
                continue
            seen.add(s.chunk_id)
            merged.append(s)
            if len(merged) >= max_sections:
                break
        for i, s in enumerate(merged):
            s.rank = i
        return merged

    async def _fallback(self, query_vec: list[float], source_ids: list[str] | None,
                        trace: SearchTrace, fusion: FusionMode) -> SearchResult:
        trace.fallback = "vector_only"
        sections = await self._vector_sections(query_vec, source_ids, self.cfg.search.max_sections)
        sections = await self._attach_source_names(sections)
        return SearchResult(sections=sections, trace=trace)

    async def _attach_source_names(self, sections: list[RetrievedSection]) -> list[RetrievedSection]:
        """批量查询 source 名称，填充到每个 section 的 source_name 字段。"""
        if not sections:
            return sections
        source_ids = list({s.source_id for s in sections})
        names = await self.db.get_source_names_by_ids(source_ids)
        for s in sections:
            s.source_name = names.get(s.source_id, "")
        return sections