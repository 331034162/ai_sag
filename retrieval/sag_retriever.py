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
        # 记录各路命中明细，供观测日志诊断 title/summary 贡献度
        title_hits: list = []
        summary_hits: list = []

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
        # 观测日志：种子召回事件明细（标题+文件名+分数+命中类型）
        # 命中类型用于诊断 title/summary 召回的相对贡献
        if seed_vector_event_ids:
            # mixed 模式下记录每个事件是 title 命中、summary 命中还是两者都命中
            title_hit_ids = {h[0] for h in title_hits}
            summary_hit_ids = {h[0] for h in summary_hits}
            seed_ev_map = {ev.id: ev for ev in
                           await self.db.get_events_by_ids(seed_vector_event_ids, source_ids)}
            seed_src_ids = {ev.source_id for ev in seed_ev_map.values()}
            seed_src_names = await self.db.get_source_names_by_ids(list(seed_src_ids))
            seed_brief = [{"score": round(path_b_scores.get(ev_id, 0.0), 4),
                           "hit": ("both" if ev_id in title_hit_ids and ev_id in summary_hit_ids
                                   else "title" if ev_id in title_hit_ids
                                   else "summary" if ev_id in summary_hit_ids
                                   else "?"),
                           "title": (seed_ev_map[ev_id].title or "")[:50],
                           "source": seed_src_names.get(seed_ev_map[ev_id].source_id, "?")}
                          for ev_id in seed_vector_event_ids if ev_id in seed_ev_map]
            log.info("[观测] 种子召回事件 query={!r} 数量={} 明细={}",
                     eff_query[:50], len(seed_vector_event_ids), seed_brief)

        # 2. 分支1：实体召回事件（依赖步骤1的实体抽取结果）
        entity_ids: list[str] = []
        if query_entities:
            entity_ids = await self._recall_entities(query_entities, source_ids)
        trace.expanded_query_entities = entity_ids  # 记录 Ûq

        entity_event_ids = (await self.db.get_event_ids_by_entity_ids(entity_ids, source_ids)
                            if entity_ids else [])
        # 观测日志：实体召回事件明细 + 种子实体名
        if entity_event_ids:
            ent_ev_map = {ev.id: ev for ev in
                          await self.db.get_events_by_ids(entity_event_ids, source_ids)}
            ent_src_ids = {ev.source_id for ev in ent_ev_map.values()}
            ent_src_names = await self.db.get_source_names_by_ids(list(ent_src_ids))
            ent_names_map = await self.db.get_entity_names_by_ids(entity_ids)
            ent_brief = [{"title": (ent_ev_map[ev_id].title or "")[:50],
                          "source": ent_src_names.get(ent_ev_map[ev_id].source_id, "?")}
                         for ev_id in entity_event_ids if ev_id in ent_ev_map]
            log.info("[观测] 实体召回事件 实体名={} 命中事件数={} 明细={}",
                     [ent_names_map.get(eid, "?") for eid in entity_ids],
                     len(entity_event_ids), ent_brief)
        else:
            log.info("实体→事件联表查询 实体数={} 命中事件数={}", len(entity_ids), len(entity_event_ids))

        # 4. 合并事件池
        seed_ids = list(dict.fromkeys(entity_event_ids + seed_vector_event_ids))
        trace.seed_event_ids = seed_ids

        if not seed_ids:
            return await self._fallback(query_vec, source_ids, trace, fusion)

        # 5. 多跳 BFS 扩展（优化 3：缓存事件供精排复用）
        event_cache: dict[str, object] = {}
        expanded_ids, event_depth_map = await self._expand(
            seed_ids, entity_ids, source_ids,
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
        # 透传粗排分数、事件缓存、事件深度图（种子=0/BFS第1轮=1...）供精排计算调整后相关度。
        rerank_input = top_n[: self.cfg.search.rerank_candidate_limit]
        trace.rerank_candidate_ids = rerank_input  # 记录 Ê（精排输入）
        reranked_ids = await self._rerank(eff_query, rerank_input,
                                          query_entity_ids=entity_ids, source_ids=source_ids,
                                          coarse_scores=coarse_scores, events_cache=event_cache,
                                          event_depth_map=event_depth_map)
        trace.reranked_ids = reranked_ids

        # 8. 事件回取切片 ChunkA（使用深度衰减调整后的分数）
        sections_a = await self._sections_for_events(
            reranked_ids, source_ids, coarse_scores=coarse_scores,
            event_depth_map=event_depth_map)

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
        # 不打印 content 内容，避免日志过长；改打印文档名和标题定位
        def _sec_brief(sections: list[RetrievedSection]) -> list[dict]:
            result = []
            for s in sections:
                result.append({"rank": s.rank, "score": round(s.score, 3),
                               "doc": (s.source_name or "")[:40],
                               "heading": (s.heading or "")[:30]})
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
        exact_names = [e.name for e in exact]
        vec_ids: list[str] = []
        vec_brief: list[dict] = []
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
            # 收集向量扩展结果，并按查询实体名分组记录分数（便于观测哪个实体名拉来哪些近邻）
            vec_ent_ids_all = list({h[0] for hits in hit_lists for h in hits})
            vec_ent_names = (await self.db.get_entity_names_by_ids(vec_ent_ids_all)
                             if vec_ent_ids_all else {})
            for qname, hits in zip(names, hit_lists):
                for eid, score in hits:
                    vec_ids.append(eid)
                    vec_brief.append({"query": qname,
                                      "entity": vec_ent_names.get(eid, "?"),
                                      "score": round(float(score), 3)})
        candidate_ids = list(dict.fromkeys(exact_ids + vec_ids))
        if not candidate_ids:
            log.info("[观测] 种子实体召回 查询实体名={} 精确匹配=0 向量扩展=0 最终=0",
                     names)
            return []
        filtered_ids = await self.db.filter_entity_ids_by_sources(candidate_ids, source_ids)
        filtered_names = (await self.db.get_entity_names_by_ids(filtered_ids)
                          if filtered_ids else {})
        log.info("[观测] 种子实体召回 查询实体名={} 精确匹配={} 向量扩展={} 最终={}\n"
                 "  精确匹配实体名={}\n"
                 "  向量扩展明细={}\n"
                 "  最终种子实体名={}",
                 names, len(exact_ids), len(vec_ids), len(filtered_ids),
                 exact_names,
                 vec_brief,
                 [filtered_names.get(eid, "?") for eid in filtered_ids])
        return filtered_ids

    async def _expand(self, seed_ids: list[str], initial_entity_ids: list[str],
                      source_ids: list[str] | None,
                      query_vec: list[float],
                      event_cache: dict[str, object] | None = None
                      ) -> tuple[list[str], dict[str, int]]:
        """沿 event_entities 超边做 BFS 多跳扩展。

        子策略：
        - multi：固定跳数扩展，每跳新事件经 budget 截断后作为下一跳种子。
        - hopllm：每跳用向量相似度对新事件重排，仅保留 topK 作为下一跳种子，
          且当最高相似度低于阈值时动态停止，避免引入噪声。

        事件深度语义（精排调整后相关度 = 向量相关度 × decay^event_depth）：
        深度公式 depth = hop + 1（BFS hop 从 0 计起）：
        - depth=0: 种子事件（实体召回 + 向量召回，可信度最高，系数 1.0）
        - depth=1: BFS hop=0 发现的新事件（第1轮扩展，系数 decay）
        - depth=2: BFS hop=1 发现的新事件（第2轮扩展，系数 decay²）
        - depth=3: BFS hop=2 发现的新事件（第3轮扩展，系数 decay³）

        返回:
            (tracked_event_ids, event_depth_map)
            event_depth_map: event_id -> BFS 发现深度，供精排计算调整后相关度。
        """
        tracked_events: set[str] = set(seed_ids)
        # 事件深度：种子事件 depth=0，BFS 各跳新事件 depth=hop+1
        event_depth_map: dict[str, int] = {eid: 0 for eid in seed_ids}
        # 实体去重集合：初始化为 initial_entity_ids（查询召回的实体），
        # 目的是避免 BFS 重复去查这些实体的事件——它们的事件关联已在种子召回阶段处理过。
        # 仅用于 BFS 去重，不记录深度。
        tracked_entities: set[str] = set(initial_entity_ids)
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
                log.info("BFS第{}跳 无新实体（边界实体均已遍历），终止扩展", hop)
                break
            # 论文 Section 4.4：entity frontier pruning budget=100，限制每跳边界实体数
            budget = self.cfg.search.entity_frontier_budget
            if self.cfg.search.entity_frontier_filter:
                # 度数硬过滤（主）+ 综合评分截断（辅），抑制高频枢纽实体桥接噪声。
                # 注意：度数过滤独立于 budget，只要开启就执行（即使 new_entities < budget），
                # 否则像"众邦银行"这种高频实体会直接通过，桥接大量无关事件。
                new_entities, degree_rejected = await self._filter_frontier_entities(
                    new_entities, query_vec, budget)
                # 被度数硬过滤剔除的实体永久屏蔽：度数是固有属性，不随 batch 变化，
                # 下一跳若再次出现这些 hub 实体，重复评估纯属浪费且可能因 batch 分布
                # 变化而误通过。综合评分截断剔除的实体不屏蔽（评分依赖 batch 归一化）。
                tracked_entities.update(degree_rejected)
            elif len(new_entities) > budget:
                new_entities = new_entities[:budget]
            tracked_entities.update(new_entities)

            new_event_ids = await self.db.get_event_ids_by_entity_ids(
                new_entities, source_ids, exclude=tracked_events)
            log.info("BFS第{}跳 实体→事件联表 新增实体={} 新事件={}",
                     hop, len(new_entities), len(new_event_ids))
            if not new_event_ids:
                log.info("BFS第{}跳 无新事件，终止扩展", hop)
                break
            # 一次性查新事件并缓存（避免观测日志与 hopllm 重排重复查询）
            new_events = await self.db.get_events_by_ids(new_event_ids, source_ids)
            if event_cache is not None:
                for ev in new_events:
                    event_cache[ev.id] = ev
            new_ev_map = {ev.id: ev for ev in new_events}
            new_src_ids = {ev.source_id for ev in new_events}
            new_src_names = await self.db.get_source_names_by_ids(list(new_src_ids))
            new_ent_names = await self.db.get_entity_names_by_ids(new_entities)
            new_ent_set = set(new_entities)
            bfs_brief = [{"title": (new_ev_map[ev_id].title or "")[:50],
                          "source": new_src_names.get(new_ev_map[ev_id].source_id, "?"),
                          "via": [new_ent_names.get(eid, "?")
                                  for eid in new_ev_map[ev_id].entity_ids
                                  if eid in new_ent_set][:5]}
                         for ev_id in new_event_ids if ev_id in new_ev_map]
            log.info("[观测] BFS第{}跳 新实体名={} 新事件明细={}",
                     hop,
                     [new_ent_names.get(eid, "?") for eid in new_entities[:20]],
                     bfs_brief)
            tracked_events.update(new_event_ids)
            # 记录事件深度：BFS hop=0 发现的新事件 depth=1，hop=1 depth=2 ...
            for ev_id in new_event_ids:
                event_depth_map[ev_id] = hop + 1

            if self.cfg.search.sub_strategy == "hopllm":
                emb_map = await self.vectors.aget_embeddings("event_contents", new_event_ids)
                scored = self._cosine_scores(query_vec, new_event_ids, emb_map)
                scored.sort(key=lambda x: x[1], reverse=True)
                if not scored:
                    log.info("BFS第{}跳 hopllm评分无结果，终止", hop)
                    break
                best_score = scored[0][1]
                if best_score < self.cfg.search.hop_relevance_threshold:
                    log.info("BFS第{}跳 hopllm最佳分={:.4f}<阈值={}，停止扩展（已发现事件保留待粗排/精排判断）",
                             hop, best_score, self.cfg.search.hop_relevance_threshold)
                    break  # 停止扩展：新事件整体相似度过低，继续扩展只会引入更多噪声。
                    # 注意：已 track 的事件不丢弃——它们的去留由粗排/精排决定，
                    # hop_relevance_threshold 是"扩展停止信号"而非"候选过滤阈值"。
                current_event_ids = [eid for eid, _ in scored[: self.cfg.search.hop_seed_topk]]
                # 观测日志：hopllm 保留的种子事件明细（复用已查过的事件，不再二次 DB 查询）
                scored_map = dict(scored)
                hop_brief = [{"score": round(scored_map.get(ev_id, 0.0), 4),
                              "title": (new_ev_map[ev_id].title or "")[:50],
                              "source": new_src_names.get(new_ev_map[ev_id].source_id, "?")}
                             for ev_id in current_event_ids if ev_id in new_ev_map]
                log.info("BFS第{}跳 hopllm重排 候选={} 取top{} 最佳分={:.4f} 明细={}",
                         hop, len(scored), len(current_event_ids), best_score, hop_brief)
            else:
                # multi 子策略：对新事件做 budget 截断（防止事件数量指数爆炸）
                # 与 hopllm 不同，不做相似度过滤，仅按数量限制下一跳种子规模。
                # 注意：new_event_ids 顺序由 SQL SELECT DISTINCT 决定（无 ORDER BY），
                # 截断本质是"随机抽样"——这是 multi 策略的设计缺陷，生产环境建议用 hopllm。
                event_budget = self.cfg.search.hop_seed_topk
                if len(new_event_ids) > event_budget:
                    current_event_ids = new_event_ids[:event_budget]
                    log.info("BFS第{}跳 multi截断 新事件={} 取top{}", hop, len(new_event_ids), event_budget)
                else:
                    current_event_ids = new_event_ids

        log.info("BFS结束 总事件数={} (种子={}) 事件深度分布={}",
                 len(tracked_events), len(seed_ids),
                 {f"depth{d}": sum(1 for v in event_depth_map.values() if v == d)
                  for d in sorted(set(event_depth_map.values()))})
        return list(tracked_events), event_depth_map

    @staticmethod
    def _degree_threshold_percentile(degree_values: list[int], pct: float) -> float:
        """分位数法：剔除 batch 内度数 > P{pct} 的实体。
        自适应 batch 分布，无需假设分布形态。"""
        import numpy as np
        return float(np.percentile(degree_values, pct))

    @staticmethod
    def _degree_threshold_mad(degree_values: list[int], k: float) -> float:
        """MAD（绝对中位差）法：threshold = median + k*MAD/0.6745。
        对长尾分布鲁棒，k=3.0 对应 99.7% 置信。MAD=0 时退化为 median*2。"""
        import numpy as np
        arr = np.asarray(degree_values, dtype=np.float64)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        if mad <= 0:
            return median * 2.0
        return median + k * mad / 0.6745

    @staticmethod
    def _degree_threshold_tukey(degree_values: list[int], k: float) -> float:
        """Tukey 篱笆法（箱线图）：threshold = Q3 + k*IQR。
        k=1.5 为经典箱线图离群点判据。IQR=0 时退化为 Q3*2。"""
        import numpy as np
        arr = np.asarray(degree_values, dtype=np.float64)
        q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        iqr = q3 - q1
        if iqr <= 0:
            return q3 * 2.0
        return q3 + k * iqr

    @staticmethod
    def _degree_threshold_otsu(degree_values: list[int]) -> float:
        """Otsu 大津法：数据驱动自动找使类间方差最大的二分阈值。
        无需手动设分位数，经典图像分割方法。退化时返回 max(degree)。"""
        import numpy as np
        arr = np.asarray(degree_values, dtype=np.float64)
        if len(arr) < 2 or arr.max() == arr.min():
            return float(arr.max())
        bins = min(20, len(arr))
        hist, edges = np.histogram(arr, bins=bins)
        total = len(arr)
        sum_total = float(arr.sum())
        sum_b, w_b, max_var, threshold = 0.0, 0, 0.0, float(arr.max())
        for i in range(len(hist)):
            w_b += int(hist[i])
            if w_b == 0:
                continue
            w_f = total - w_b
            if w_f == 0:
                break
            bin_center = (edges[i] + edges[i + 1]) / 2
            sum_b += hist[i] * bin_center
            m_b = sum_b / w_b
            m_f = (sum_total - sum_b) / w_f
            var_between = w_b * w_f * (m_b - m_f) ** 2
            if var_between > max_var:
                max_var = var_between
                threshold = float(bin_center)
        return threshold

    def _compute_degree_threshold(
        self, degree_values: list[int], method: str
    ) -> tuple[float, str]:
        """根据配置方法计算度数离群阈值，返回 (threshold, 方法描述)。"""
        k = self.cfg.search.entity_degree_outlier_k
        pct = self.cfg.search.entity_degree_percentile
        if method == "mad":
            return self._degree_threshold_mad(degree_values, k), f"MAD(k={k})"
        if method == "tukey":
            return self._degree_threshold_tukey(degree_values, k), f"Tukey(k={k})"
        if method == "otsu":
            return self._degree_threshold_otsu(degree_values), "Otsu"
        if method == "none":
            return float('inf'), "None(仅绝对上限)"
        return self._degree_threshold_percentile(degree_values, pct), f"P{pct}"

    async def _filter_frontier_entities(
        self, entity_ids: list[str], query_vec: list[float], budget: int
    ) -> tuple[list[str], set[str]]:
        """BFS 边界实体过滤：度数硬过滤（主，多方法可配置）+ 综合评分截断（辅）。

        返回:
            (保留的实体列表, 被度数硬过滤剔除的实体集合)
            被剔除的实体集合供调用方加入 tracked_entities 永久屏蔽，
            避免后续跳重复评估同一 hub 实体（度数是固有属性，不随 batch 变化）。
            综合评分截断剔除的实体不返回（其评分依赖 batch 内归一化，重评有意义）。

        度数硬过滤（从源头切断"众邦银行"等枢纽实体的桥接路径）：
          1. 绝对上限：degree > entity_degree_abs_max 直接剔除（所有方法共用兜底，0=关闭）
          2. 离群检测（主判据，方法由 entity_degree_method 配置）：
             - percentile：剔除度数 > batch 内 P{percentile} 的实体（默认，自适应分布）
             - mad：剔除度数 > median + k*MAD/0.6745 的实体（长尾鲁棒）
             - tukey：剔除度数 > Q3 + k*IQR 的实体（经典箱线图）
             - otsu：数据驱动自动找最优二分阈值（无需手动设分位数）
             - none：关闭离群检测，仅用绝对上限（最早的行为）
             仅当 batch 大小 >= entity_degree_min_batch 时启用（小 batch 统计量不稳定）
          3. 过度过滤保护：若离群过滤后实体数 < 输入数/2 且 abs_max 确实过滤过，
             放宽到 abs_max 结果；否则保留离群结果（注意 abs_max=0 时无放宽目标）

        综合评分（辅，对硬过滤后的实体做 budget 截断）：
          综合分 = (1-α)*IDF + α*query相似度
          - IDF：1/(1+degree)，度数越高分越低
          - query相似度：实体向量与 query 向量的余弦相似度，保留与问题语义相关的实体
          - α = entity_frontier_query_weight，默认 0.6
        两个分量均在 batch 内归一化到 [0,1] 再加权，取 top budget。
        """
        if not entity_ids:
            return [], set()
        degrees = await self.db.get_entity_degrees(entity_ids)

        # === 度数硬过滤（主判据，从源头剔除高频枢纽）===
        abs_max = self.cfg.search.entity_degree_abs_max
        method = self.cfg.search.entity_degree_method
        min_batch = self.cfg.search.entity_degree_min_batch

        # 条件1：绝对上限过滤（所有方法共用兜底）
        if abs_max > 0:
            after_abs = [eid for eid in entity_ids if degrees.get(eid, 0) <= abs_max]
        else:
            after_abs = list(entity_ids)

        # 条件2：离群检测过滤（仅当 after_abs batch 足够大且方法非 none）
        # 注意：① 阈值基于 after_abs 计算（剔除 abs_max 实体后的真实参与样本）
        #       ② min_batch 判断也基于 after_abs，避免 abs_max 已大量剔除后仍硬上统计
        use_method = method != "none" and len(after_abs) >= min_batch
        if use_method:
            degree_values = [degrees.get(eid, 0) for eid in after_abs]
            threshold, method_desc = self._compute_degree_threshold(degree_values, method)
            after_method = [eid for eid in after_abs if degrees.get(eid, 0) <= threshold]
        else:
            threshold = float('inf')
            method_desc = f"{method}(batch<{min_batch}跳过)"
            after_method = list(after_abs)

        # 条件3：过度过滤保护（过滤后太少则放宽到只用绝对上限）
        # min_keep 基于输入数量而非 budget：度数过滤独立于 budget 执行，
        # 若用 budget//2 会因 budget=100 而 min_keep=50，导致小输入（如34个）过滤后总触发放宽。
        # 仅当 abs_max 确实剔除了实体（after_abs < entity_ids）时才放宽到 after_abs，
        # 否则 abs_max=0 → after_abs=原始，放宽无意义，应保留离群结果避免回退到完全无过滤。
        min_keep = max(len(entity_ids) // 2, 5)
        abs_filtered = len(after_abs) < len(entity_ids)
        if len(after_method) < min_keep and abs_filtered:
            filtered_ids = after_abs
            relaxed = True
        else:
            filtered_ids = after_method
            relaxed = False

        # 过滤后为空则回退到原始候选（避免 BFS 完全断链）
        if not filtered_ids:
            filtered_ids = list(entity_ids)
            relaxed = True

        # 计算被度数硬过滤剔除的实体（供调用方永久屏蔽，避免后续跳重复评估）
        # 仅屏蔽被 abs_max（固定阈值）剔除的实体——abs_max 是硬上限，任何 batch 下
        # degree>abs_max 都是 hub，永久屏蔽安全。
        # 离群检测（percentile/mad/tukey/otsu）剔除的实体不屏蔽——其阈值是 batch 内
        # 动态统计量，下一跳 batch 分布不同时重评可能得到不同结果，重评有意义。
        # 综合评分截断剔除的实体同理不屏蔽（评分依赖 batch 归一化）。
        # filtered_ids 可能因"过度过滤保护"或"空集回退"放宽到 after_abs/原始，
        # 此时被放宽"救回"的实体不算剔除。
        filtered_set = set(filtered_ids)
        degree_rejected = {eid for eid in entity_ids
                           if eid not in filtered_set
                           and degrees.get(eid, 0) > abs_max > 0}

        filtered_degrees = [degrees.get(eid, 0) for eid in filtered_ids] if filtered_ids else [0]
        log.info("[观测] 边界实体度数过滤 候选={} 方法={} 阈值={} 绝对上限={} 启用离群={} "
                 "硬过滤后={} 过度放宽={} 度数范围={}-{}",
                 len(entity_ids), method_desc,
                 round(threshold, 2) if use_method else "跳过",
                 abs_max, use_method, len(filtered_ids), relaxed,
                 min(filtered_degrees), max(filtered_degrees))

        # 如果硬过滤后已不足 budget，直接返回，跳过综合评分（无意义）
        # budget 下限为 1，避免 0 导致 BFS 断链
        if len(filtered_ids) <= max(budget, 1):
            return filtered_ids, degree_rejected

        # === 综合评分截断（辅，对硬过滤后的实体做 budget 截断）===
        # 方案1：IDF 评分（度数倒数，本身在 [0,1]）
        idf_raw = {eid: 1.0 / (1.0 + degrees.get(eid, 0)) for eid in filtered_ids}
        # 方案3：query 相似度评分
        entity_embs = await self.vectors.aget_embeddings("entities", filtered_ids)
        sim_pairs = self._cosine_scores(query_vec, filtered_ids, entity_embs)
        sim_raw = {eid: max(0.0, s) for eid, s in sim_pairs}  # 截断负相似度
        # batch 内归一化到 [0,1]，消除量纲差异。
        # 注意：纯 min-max 对异常最值敏感——一个离群小值会把其余样本压到接近 1，
        # 相对差异被压缩。改用 winsorize+min-max：先按分位数裁剪极端值再做归一化，
        # 保留主体分布的相对差异。
        def _winsor_minmax(d: dict[str, float], lo_pct: float = 0.1, hi_pct: float = 0.9
                           ) -> dict[str, float]:
            if not d:
                return d
            vals = sorted(d.values())
            n = len(vals)
            if n < 4:  # 样本太少不做 winsorize
                lo, hi = vals[0], vals[-1]
            else:
                lo = vals[int(n * lo_pct)]
                hi = vals[int(n * hi_pct)]
            rng = hi - lo
            if rng < 1e-8:
                return {k: 1.0 for k in d}  # 全相同则等权
            return {k: min(max((v - lo) / rng, 0.0), 1.0) for k, v in d.items()}
        idf_norm = _winsor_minmax(idf_raw)
        sim_norm = _winsor_minmax(sim_raw)
        # 综合分加权
        alpha = self.cfg.search.entity_frontier_query_weight
        combined = {eid: (1.0 - alpha) * idf_norm.get(eid, 0.0) + alpha * sim_norm.get(eid, 0.0)
                    for eid in filtered_ids}
        # 按综合分降序取 top budget（budget 下限 1，避免 BFS 断链）
        ranked = sorted(filtered_ids, key=lambda eid: combined[eid], reverse=True)
        result = ranked[:max(budget, 1)]
        log.info("[观测] 边界实体综合评分截断 输入={} 保留={} α={} 最佳分={:.4f} 度数范围={}-{}",
                 len(filtered_ids), len(result), alpha,
                 combined[result[0]] if result else 0.0,
                 min(degrees.get(eid, 0) for eid in result) if result else 0,
                 max(degrees.get(eid, 0) for eid in result) if result else 0)
        # 综合评分截断剔除的实体（filtered_ids - result）不加入 degree_rejected，
        # 因为其评分依赖 batch 内归一化，下一跳重评可能得到不同结果。
        return result, degree_rejected

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
        # 观测日志：粗排结果明细（标题+文件名+分数）
        if scored:
            coarse_ev_map = {ev.id: ev for ev in
                             await self.db.get_events_by_ids([eid for eid, _ in scored], source_ids)}
            coarse_src_ids = {ev.source_id for ev in coarse_ev_map.values()}
            coarse_src_names = await self.db.get_source_names_by_ids(list(coarse_src_ids))
            coarse_brief = [{"score": round(s, 4),
                             "title": (coarse_ev_map[eid].title or "")[:50],
                             "source": coarse_src_names.get(coarse_ev_map[eid].source_id, "?")}
                            for eid, s in scored if eid in coarse_ev_map]
            log.info("[观测] 粗排结果 输入={} 保留={} top5明细={}",
                     len(event_ids), len(scored), coarse_brief[:5])
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
                      events_cache: dict[str, object] | None = None,
                      event_depth_map: dict[str, int] | None = None) -> list[str]:
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
        # 观测日志：精排输入候选明细（标题+文件名）
        rerank_src_ids = [ev.source_id for ev in events]
        rerank_src_names = await self.db.get_source_names_by_ids(rerank_src_ids)
        rerank_in_brief = [{"title": (ev.title or "")[:50],
                            "source": rerank_src_names.get(ev.source_id, "?")}
                           for ev in events]
        log.info("[观测] 精排输入候选 数量={} 明细={}", len(events), rerank_in_brief)
        try:
            selected = await self._llm_rerank(query, events, query_entity_ids=query_entity_ids or [],
                                             coarse_scores=coarse_scores,
                                             event_depth_map=event_depth_map or {})
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

        # 精排结果日志（补充文件名）
        ev_map = {ev.id: ev for ev in events}
        result_src_ids = [ev_map[eid].source_id for eid in result if eid in ev_map]
        result_src_names = (await self.db.get_source_names_by_ids(result_src_ids)
                            if result_src_ids else {})
        brief: list[dict] = []
        for eid in result:
            ev = ev_map.get(eid)
            if ev:
                summary_snip = (ev.summary or "")[:80]
                brief.append({"title": (ev.title or "")[:60],
                              "summary": summary_snip + ("..." if len(ev.summary or "") > 80 else ""),
                              "source": result_src_names.get(ev.source_id, "?"),
                              "coarse_score": round(coarse_scores.get(eid, 0.0), 4) if coarse_scores else None})
        log.info("[观测] LLM精排结果 输入={} 输出={} 事件={}",
                 len(event_ids), len(result), brief)
        return result

    async def _llm_rerank(self, query: str, events: list, *,
                          query_entity_ids: list[str],
                          coarse_scores: dict[str, float] | None = None,
                          event_depth_map: dict[str, int] | None = None) -> list[str]:
        query_entity_set = set(query_entity_ids)
        entity_names = (await self.db.get_entity_names_by_ids(query_entity_ids)
                        if query_entity_ids else {})
        depth_map = event_depth_map or {}
        decay = self.cfg.search.rerank_event_decay
        # 召回轮次名称映射（depth→可读标签），供候选格式展示
        round_labels = {0: "种子", 1: "BFS第1轮", 2: "BFS第2轮", 3: "BFS第3轮"}
        # 种子事件数（depth=0）用于日志区分种子 vs BFS 扩展
        # 注意：统计的是精排输入 events 中的种子数，而非整个 depth_map
        # （粗排可能过滤掉部分种子，导致 depth_map 数量 > 精排输入数量）
        ev_ids_in_rerank = {ev.id for ev in events}
        seed_ev = sum(1 for eid, d in depth_map.items()
                      if d == 0 and eid in ev_ids_in_rerank)

        # IDF 降权：查查询实体的 degree，计算 IDF = log(总事件数 / degree)。
        # 解决 hub 实体（如"新一代信贷系统"关联 8+ 事件）让所有事件都"强命中"的问题。
        # 有效信号 = weight × IDF_norm，hub 实体 weight 高但 IDF 低，有效信号被压低。
        # IDF 归一化到 [0,1]：IDF_norm = IDF / IDF_max，保证 effective = weight × IDF_norm
        # 仍在 [0,1] 范围，复用原有 weight_strong/weight_weak 阈值无需调整。
        total_events = len(ev_ids_in_rerank)
        entity_idf: dict[str, float] = {}
        if query_entity_ids and total_events > 0:
            degrees = await self.db.get_entity_degrees(query_entity_ids)
            import math
            raw_idf = {}
            for eid in query_entity_ids:
                deg = degrees.get(eid, 0)
                # degree=0 理论上不该出现（查询实体必有事件关联），兜底给高 IDF
                raw_idf[eid] = math.log(total_events / deg) if deg > 0 else math.log(total_events)
            # 归一化到 [0,1]：除以最大 IDF（最稀有的实体得 1.0）
            max_idf = max(raw_idf.values()) if raw_idf else 1.0
            if max_idf > 0:
                entity_idf = {eid: v / max_idf for eid, v in raw_idf.items()}

        lines = []
        rerank_brief = []
        total_qe = len(query_entity_set)
        for i, ev in enumerate(events):
            roles_text, n_strong, n_weak = self._format_entity_roles(
                ev, query_entity_set, entity_names,
                weight_strong=self.cfg.search.rerank_weight_strong,
                weight_weak=self.cfg.search.rerank_weight_weak,
                entity_idf=entity_idf,
            )
            coarse = coarse_scores.get(ev.id, 0.0) if coarse_scores else 0.0
            ev_depth = depth_map.get(ev.id, 0)
            # 调整后相关度 = 向量相关度 × decay^事件深度
            adjusted = coarse * (decay ** ev_depth)
            round_label = round_labels.get(ev_depth, f"BFS第{ev_depth}轮")
            lines.append(RERANK_CANDIDATE_FORMAT.format(
                i=i, event_id=ev.id, title=ev.title, summary=ev.summary,
                score=f"{coarse:.3f}",
                adjusted=f"{adjusted:.3f}",
                round=round_label,
                roles=roles_text,
            ))
            # 观测统计：命中查询实体数/强信号数/弱信号数/事件深度/调整后相关度/命中实体weight+IDF明细
            hit_count = sum(1 for eid in ev.entity_roles if eid in query_entity_set)
            # 命中实体的 weight × IDF 明细，用于确认 hub 实体是否被有效降权
            hit_signals = {entity_names.get(eid, eid[:8]): {
                "weight": round(ev.entity_weights.get(eid, 0.5), 2),
                "idf": round(entity_idf.get(eid, 0.0), 2),
                "signal": round(ev.entity_weights.get(eid, 0.5) * entity_idf.get(eid, 0.0), 2),
            } for eid in ev.entity_roles if eid in query_entity_set}
            rerank_brief.append({
                "i": i,
                "title": (ev.title or "")[:40],
                "depth": ev_depth,
                "coarse": round(coarse, 3),
                "adjusted": round(adjusted, 3),
                "hit": f"{hit_count}/{total_qe}",
                "strong": n_strong,
                "weak": n_weak,
                "signals": hit_signals,
            })
        candidates = "\n\n".join(lines)
        log.info("[观测] 精排候选命中明细 decay={} 候选数={} 事件(种子={}/扩展={}/总={}) 明细={}",
                 decay, len(events), seed_ev, len(events) - seed_ev, len(events), rerank_brief)
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
                             entity_names: dict[str, str],
                             *,
                             weight_strong: float = 0.7,
                             weight_weak: float = 0.4,
                             entity_idf: dict[str, float] | None = None) -> tuple[str, int, int]:
        """格式化命中查询实体的角色与强弱信号。

        强弱信号基于 weight × IDF（有效信号）：
        - weight：实体在该事件中的关联权重（LLM 入库时输出，0.1-1.0）
        - IDF：实体的逆文档频率，log(总事件数 / 该实体关联事件数)
        - 有效信号 = weight × IDF，解决 hub 实体（weight 高但区分度低）让所有事件都"强命中"的问题

        阈值（基于 weight × IDF 的乘积）：
        - effective >= weight_strong 为强信号（核心关联 + 高区分度）
        - weight_weak <= effective < weight_strong 为中信号
        - effective < weight_weak 为弱信号（背景/泛命中）

        注意：引入 IDF 后，hub 实体（如"新一代信贷系统"关联 8+ 事件）的 IDF 低，
        即便 weight=1.0，有效信号也可能降到中/弱，让真正有区分度的低频实体凸显。

        返回:
            (roles_text, n_strong, n_weak)
        """
        if not query_entity_set or not ev.entity_roles:
            return "", 0, 0
        idf_map = entity_idf or {}
        hit_roles = []
        n_strong = 0
        n_weak = 0
        for eid, role in ev.entity_roles.items():
            if eid not in query_entity_set:
                continue
            name = entity_names.get(eid, eid[:8])
            weight = ev.entity_weights.get(eid, 0.5)
            idf = idf_map.get(eid, 0.0)
            effective = weight * idf  # 有效信号 = weight × IDF
            if effective >= weight_strong:
                signal = "强"
                n_strong += 1
            elif effective >= weight_weak:
                signal = "中"
            else:
                signal = "弱"
                n_weak += 1
            if role:
                hit_roles.append(f"{name}({role},{signal},w{weight:.1f},idf{idf:.1f})")
            else:
                hit_roles.append(f"{name}(角色未标注,{signal},w{weight:.1f},idf{idf:.1f})")
        if not hit_roles:
            return "", 0, 0
        total = len(query_entity_set)
        hit_count = sum(1 for eid in ev.entity_roles if eid in query_entity_set)
        return (f"命中查询实体：{'、'.join(hit_roles)}  "
                f"[命中 {hit_count}/{total} 强{n_strong}弱{n_weak}]",
                n_strong, n_weak)

    async def _sections_for_events(self, event_ids: list[str],
                                   source_ids: list[str] | None,
                                   coarse_scores: dict[str, float] | None = None,
                                   event_depth_map: dict[str, int] | None = None,
                                   ) -> list[RetrievedSection]:
        if not event_ids:
            return []
        ev_chunk_map = await self.db.get_chunk_ids_by_event_ids(event_ids)
        chunk_ids = [ev_chunk_map[eid] for eid in event_ids if eid in ev_chunk_map]
        if not chunk_ids:
            return []
        chunks = await self.db.get_chunks_by_ids(chunk_ids)
        chunk_map = {c.id: c for c in chunks}
        depth_map = event_depth_map or {}
        decay = self.cfg.search.rerank_event_decay
        sections = []
        for eid in event_ids:
            cid = ev_chunk_map.get(eid)
            if cid is None:
                continue
            c = chunk_map.get(cid)
            if c is None:
                continue
            coarse = (coarse_scores or {}).get(eid, 0.0)
            ev_depth = depth_map.get(eid, 0)
            adjusted = coarse * (decay ** ev_depth)
            sections.append(RetrievedSection(
                chunk_id=c.id, source_id=c.source_id, document_id=c.document_id,
                heading=c.heading, content=c.content,
                rank=len(sections),
                score=float(adjusted),
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
        # rank 按"实际保留顺序"重新编号，避免 hits 中被跳过的项导致 rank 跳号
        # （否则日志里会看到 rank=0,1,2,5 这种不连续现象）
        for cid, score in hits:
            c = chunk_map.get(cid)
            if c is None or c.id in existing:
                continue
            sections.append(RetrievedSection(
                chunk_id=c.id, source_id=c.source_id, document_id=c.document_id,
                heading=c.heading, content=c.content,
                rank=len(sections), score=float(score),
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