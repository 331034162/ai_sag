"""异步入库编排：loader → cleaner → splitter → extractor → storage。

全链路异步：DB 用 aiomysql，LLM 用 LlamaIndex 原生异步，
Embedder/VectorStore 用异步接口（a 前缀方法）。
"""
from __future__ import annotations

import asyncio
import re
import uuid

from llama_index.core.llms import LLM

from ..base import Chunk, Config, LoadedDocument, SUPPORTED_GENRES
from ..base.logger import get_logger
from ..cleaner import TextCleaner
from ..embeddings import create_embedder
from ..extractor import EventExtractor
from ..llm import LlmFactory
from ..loader import DocumentLoader
from ..splitter import create_splitter
from ..storage import MysqlStore
from ..vector_store import create_vector_store

log = get_logger()

# V6 解析器在 Excel CSV 行首添加的角色前缀（结构识别标注）：
#   #TITLE# / #FORM# / #SIGNING# / #GROUP_HEADER# / #DATA#
# TableSplitter 切分时据此识别行角色并剥离；但 LoadedDocument.content 会原样写入
# aisag_documents 表，导致 document 与 chunk 口径不一致。此正则用于入库前剥离行首前缀，
# 仅对 xlsx 文件应用，保证 documents 表与 chunks 表内容口径一致。
_ROLE_PREFIX_RE = re.compile(r'^#(?:TITLE|FORM|SIGNING|GROUP_HEADER|DATA)#', re.MULTILINE)


def _strip_excel_role_prefixes(text: str) -> str:
    """剥离 Excel V6 角色前缀，仅用于写 documents 表前的清理。

    清理范围：行首的 #TITLE#/#FORM#/#SIGNING#/#GROUP_HEADER#/#DATA# 五种前缀。
    不影响 CSV 内部的 # 字符（仅匹配行首固定前缀），不影响其他文件类型。
    """
    return _ROLE_PREFIX_RE.sub('', text)


class IngestPipeline:
    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self.loader = DocumentLoader.default(config=self.cfg)
        self.cleaner = TextCleaner(
            strip_html=self.cfg.cleaner.strip_html,
            merge_hard_breaks=self.cfg.cleaner.merge_hard_breaks,
            collapse_blank_lines=self.cfg.cleaner.collapse_blank_lines,
            normalize_whitespace=self.cfg.cleaner.normalize_whitespace,
            protect_list_items=self.cfg.cleaner.protect_list_items,
        )
        self.embedder = create_embedder(self.cfg)
        # semantic 模式需要 embed_model，复用已有的 embedder
        embed_model = getattr(self.embedder, '_model', None)
        self.splitter = create_splitter(self.cfg, embed_model=embed_model)
        # LlmFactory 支持按场景返回不同 LLM 实例（体裁分类/事件抽取等按场景选用）
        self._llm_factory = LlmFactory(self.cfg)
        # self.llm 作为兜底 LLM：用 GENRE_CLASSIFY 场景（pipeline 内部按需 _llm_for 选场景）
        self.llm: LLM = self._llm_factory.get("GENRE_CLASSIFY")
        # 事件抽取器使用 EVENT_EXTRACT 场景的 LLM
        self.extractor = EventExtractor(
            self._llm_factory.get("EVENT_EXTRACT"),
            max_retries=self.cfg.ingest.extract_max_retries,
            summary_max_chars=self.cfg.ingest.summary_max_chars,
            title_max_chars=self.cfg.ingest.title_max_chars,
        )
        self.db = MysqlStore(
            host=self.cfg.mysql.host, port=self.cfg.mysql.port,
            user=self.cfg.mysql.user, password=self.cfg.mysql.password,
            database=self.cfg.mysql.database,
            pool_size=self.cfg.mysql.pool_size,
            max_overflow=self.cfg.mysql.max_overflow,
            pool_timeout=self.cfg.mysql.pool_timeout,
            pool_recycle=self.cfg.mysql.pool_recycle,
        )
        self.vectors = create_vector_store(self.cfg)
        self._reconcile_task: asyncio.Task | None = None
        # 并发入库信号量：限制同时入库的文档数，防止 LLM API rate limit / embedding OOM
        self._ingest_semaphore = asyncio.Semaphore(self.cfg.ingest.concurrency)

    async def init(self) -> None:
        """异步初始化：建表 + 启动对账（含硬删除软删事件）。需在 async 上下文中调用。"""
        await self.db.ensure_schema()
        await self._reconcile()
        self._start_reconcile_loop()

    async def ingest_file(self, path: str, *, source_name: str | None = None,
                          source_id: str | None = None,
                          title: str | None = None,
                          md5: str = "",
                          ocr_images: bool | None = None,
                          ocr_backend: str | None = None) -> str:
        # heading 修复：透传 title，避免 chunk.heading 退化为临时文件名
        # ocr_images / ocr_backend：请求级覆盖 config 默认值（None=用配置默认）
        doc = self.loader.load(path, title=title, ocr_images=ocr_images, ocr_backend=ocr_backend)
        return await self.ingest_document(doc, source_name=source_name, source_id=source_id, md5=md5)

    async def ingest_text(self, title: str, content: str, *, source_name: str | None = None,
                          source_id: str | None = None,
                          md5: str = "") -> str:
        doc = self.loader.load_text(title, content)
        return await self.ingest_document(doc, source_name=source_name, source_id=source_id, md5=md5)

    async def ingest_document(self, doc: LoadedDocument, *, source_name: str | None = None,
                              source_id: str | None = None,
                              md5: str = "") -> str:
        async with self._ingest_semaphore:
            return await self._ingest_document_impl(doc, source_name=source_name, source_id=source_id, md5=md5)

    async def _ingest_document_impl(self, doc: LoadedDocument, *, source_name: str | None = None,
                                    source_id: str | None = None,
                                    md5: str = "") -> str:
        source_id = source_id or str(uuid.uuid4())
        source_name = source_name or doc.title
        document_id = str(uuid.uuid4())

        log.info("入库开始 source={} doc={}", source_name, doc.title)
        cleaned = self.cleaner.clean(doc)
        # 文档级文体识别（方案 A+B 前置步骤）：单文档单标签，驱动抽取时的判别规则和 role 词汇表
        # 失败时降级为 generic，不阻断入库；用开关控制可关闭以省一次 LLM 调用
        genre = await self._detect_genre(cleaned)
        log.info("文体识别完成 doc={} genre={}", cleaned.title, genre)

        chunks = self.splitter.split(cleaned, source_id=source_id, document_id=document_id)
        if not chunks:
            raise ValueError("切分后无有效切片，请检查文档内容")
        log.info("切分完成 chunk数={} 总字符={}", len(chunks), sum(len(c.content) for c in chunks))

        events = await self._extract_events(chunks, cleaned.title, genre=genre)
        total_entities = sum(len(e.entities) for e in events)
        fallback_count = sum(1 for e in events if not e.entities)
        log.info("事件抽取完成 事件数={} 实体总数={} 空实体事件数={}",
                 len(events), total_entities, fallback_count)

        await self._persist(source_id, source_name, document_id, cleaned, chunks, events, md5=md5)
        log.info("入库完成 source_id={}", source_id)
        return source_id

    async def _extract_events(self, chunks: list[Chunk], doc_title: str, *,
                              genre: str = "generic"):
        """事件抽取：顺序模式传递前文摘要用于代词消解，并行模式独立抽取。

        顺序模式：通过 extract_batch 逐 chunk 传递前一个 chunk 的事件摘要，
        LLM 可利用 previous_context 消解"该公司""上述协议"等指代。
        并行模式：各 chunk 独立并发，无法传递跨 chunk 上下文（一致性优先于速度时建议关并行）。
        任一 chunk 抽取失败（ExtractionError）均向上传播，终止入库，不写入低质量数据。
        genre 为文档级文体标签，透传给 extractor 驱动方案 A+B 文体感知增强。

        表格文体（tabular）强制走并行：表格 chunk 之间无语义依赖（每行独立记录），
        顺序模式的 prev 摘要代词消解对表格无效，串行只会拖慢入库，故默认并发抽取。
        """
        from ..base.models import ExtractionError as _ExtractionError

        # 表格文体强制并发；其他文体按全局 extract_parallel 开关
        use_parallel = self.cfg.ingest.extract_parallel or genre == "tabular"
        if use_parallel:
            loop = asyncio.get_running_loop()
            tasks = [
                loop.run_in_executor(None, self.extractor.extract, chunk, doc_title, "", genre)
                for chunk in chunks
            ]
            # return_exceptions=True：先收集所有结果，再统一检查，避免 gather 抛异常后
            # 已提交的线程池任务成为无法追踪的"幽灵任务"。
            results = await asyncio.gather(*tasks, return_exceptions=True)
            errors: list[tuple[int, str]] = []
            for i, r in enumerate(results):
                if isinstance(r, _ExtractionError):
                    errors.append((i, str(r)))
                elif isinstance(r, BaseException):
                    errors.append((i, f"抽取异常 chunk_idx={i}: {r}"))
            if errors:
                log.error("事件抽取失败，终止入库 doc={} 失败数={}/{} 详情={}",
                          doc_title, len(errors), len(chunks), [e[1] for e in errors[:3]])
                raise _ExtractionError(
                    chunks[errors[0][0]].id,
                    self.cfg.ingest.extract_max_retries,
                    "; ".join(e[1] for e in errors[:3]))
            return results  # type: ignore[return-value]
        return await asyncio.to_thread(
            self.extractor.extract_batch, chunks, doc_title=doc_title,
            parallel=self.cfg.ingest.extract_parallel,
            max_workers=self.cfg.ingest.extract_parallel_workers,
            genre=genre)

    async def _detect_genre(self, doc: LoadedDocument) -> str:
        """文档级文体识别：用 LLM 从文档标题+首尾样本判断，返回 SUPPORTED_GENRES 之一。

        设计要点：
        - 文档级而非 chunk 级，降低判断误差和调用成本（每文档仅 1 次 LLM 调用）
        - 只取标题+首尾文本样本，避免全文输入浪费 token
        - 输出限定为 SUPPORTED_GENRES 枚举，防止 LLM 自创标签
        - 失败或关闭开关时返回 generic，不阻断入库
        - 查询抽取侧不调用此方法（查询无文体属性），保证入库/查询口径一致
        """
        if not self.cfg.ingest.genre_detect:
            return "generic"
        # 短路：文件类型已确定文体的，直接返回，省一次 LLM 调用
        # xlsx/csv 均为表格数据，转 markdown 表格后走 tabular 文体抽取
        if doc.file_type in ("xlsx", "csv"):
            return "tabular"
        # 首尾各取 800 字作为样本，兼顾开头定调与结尾收束
        sample = doc.title + "\n" + doc.content[:800]
        if len(doc.content) > 800:
            sample += "\n...\n" + doc.content[-500:]
        prompt = (
            "判断以下文本的主要文体，从下列选项中选最贴切的一个：\n"
            f"{', '.join(SUPPORTED_GENRES)}\n"
            "选项含义：news=新闻报道，contract=合同/协议，financial_report=财务报告，"
            "legal=法律文书，meeting=会议纪要，manual=说明书/操作手册，"
            "academic=学术论文/研究报告，report=通用报告（工作/调研/年度/政务报告，非学术），"
            "fiction=小说/虚构叙事，poetry=诗歌/散文诗，"
            "tabular=表格数据，generic=通用兜底。\n"
            "判别提示：学术报告（含实验/文献综述/方法论）归 academic，"
            "工作/调研/年度/政务报告归 report；"
            "虚构叙事（小说/故事/章回）归 fiction，韵文（诗/词/散曲）归 poetry；"
            "以 markdown 表格为主体（多行多列数据）归 tabular。\n"
            "只返回选项名（小写英文），不要解释、不要标点。\n\n文本：\n" + sample
        )
        try:
            resp = await self._llm_factory.get("GENRE_CLASSIFY").acomplete(prompt)
            genre = str(resp).strip().strip("\"'""''").lower()
            # 容错：LLM 可能返回带说明的文本，取第一个匹配的 genre 词
            for g in SUPPORTED_GENRES:
                if g in genre:
                    return g
            return "generic"
        except Exception as e:
            log.warning("文体识别失败，降级为 generic err={}", e)
            return "generic"

    async def _persist(self, source_id: str, source_name: str, document_id: str,
                       doc: LoadedDocument, chunks: list[Chunk],
                       extracted_events: list, *,
                       md5: str = "") -> None:
        await self.db.ping()

        log.info("生成 chunk 向量 数量={}", len(chunks))
        chunk_texts = [c.content for c in chunks]
        chunk_embs = await self.embedder.aembed_texts(chunk_texts) if chunk_texts else []

        # 写库前清理 documents 表 content：仅对 Excel 剥离 V6 角色前缀
        # 时机：chunks 已切分完成（TableSplitter 用带前缀的 doc.content 识别角色），
        #       此处只清理写 documents 表用的副本，不影响 chunks 列表与其他文件类型
        # 判定：开关开启 + file_type 限定 xlsx/xls（当前只有 xlsx 能进 ExcelReader，xls 预留扩展）
        #       + content 确实含 V6 前缀（防御未来 ExcelReader 切到无前缀的 V2 等版本，避免无谓清理）
        needs_clean = (self.cfg.ingest.strip_excel_role_prefix
                       and doc.file_type in ("xlsx", "xls")
                       and _ROLE_PREFIX_RE.search(doc.content) is not None)
        doc_content = _strip_excel_role_prefixes(doc.content) if needs_clean else doc.content

        log.info("MySQL 事务写入 source/document/chunks/events/entities")
        event_records, seen_entities = await self.db.persist_source(
            source_id=source_id, source_name=source_name, file_type=doc.file_type,
            document_id=document_id, doc_title=doc.title, doc_content=doc_content,
            chunks=chunks, events=extracted_events,
            md5=md5,
        )
        log.info("MySQL 写入完成 事件={} 实体引用={}（含跨文档共享复用，向量库仅写新增实体）",
                 len(event_records), len(seen_entities))

        try:
            log.info("写入 chunks 向量 数量={}", len(chunks))
            chunk_vec_items = [(c.id, c.content, e) for c, e in zip(chunks, chunk_embs)]
            await self.vectors.aadd_chunks(chunk_vec_items, source_id=source_id, document_id=document_id)

            title_texts = [e.title for e in event_records]
            content_texts = [e.content for e in event_records]
            if title_texts:
                log.info("生成事件标题向量 数量={}", len(title_texts))
                title_embs = await self.embedder.aembed_texts(title_texts)
                await self.vectors.aadd_events(
                    [(e.id, e.title, emb) for e, emb in zip(event_records, title_embs)],
                    source_id=source_id, document_id=document_id)
            if content_texts:
                log.info("生成事件内容向量 数量={}", len(content_texts))
                content_embs = await self.embedder.aembed_texts(content_texts)
                await self.vectors.aadd_event_contents(
                    [(e.id, e.content, emb) for e, emb in zip(event_records, content_embs)],
                    source_id=source_id, document_id=document_id)
            if self.cfg.ingest.embed_summary:
                summary_texts = [e.summary for e in event_records]
                if summary_texts:
                    log.info("生成事件摘要向量 数量={}", len(summary_texts))
                    summary_embs = await self.embedder.aembed_texts(summary_texts)
                    await self.vectors.aadd_event_summaries(
                        [(e.id, e.summary, emb) for e, emb in zip(event_records, summary_embs)],
                        source_id=source_id, document_id=document_id)
            if seen_entities:
                name_lookup: dict[tuple[str, str], str] = {}
                for ev in extracted_events:
                    for e in ev.entities:
                        key = (e.type, self.db._normalize(e.name))
                        if key not in name_lookup:
                            name_lookup[key] = e.name
                entity_pairs: list[tuple[str, str]] = []
                entity_names: list[str] = []
                seen_eids = set()
                for (etype, norm), eid in seen_entities.items():
                    if eid not in seen_eids:
                        seen_eids.add(eid)
                        ename = name_lookup.get((etype, norm), norm)
                        entity_pairs.append((eid, ename))
                        entity_names.append(ename)
                entity_embs = await self.embedder.aembed_texts(entity_names) if entity_names else []
                entity_vec_items = [
                    (pair[0], pair[1], emb) for pair, emb in zip(entity_pairs, entity_embs)]
                log.info("生成实体向量 数量={}", len(entity_vec_items))
                await self.vectors.aadd_entities(entity_vec_items)
            await self.db.mark_vector_synced(source_id, True)
        except Exception as vec_err:
            log.error("向量库写入失败，回滚 MySQL 数据 source_id={} err={}", source_id, vec_err)
            try:
                _, orphan_ids = await self.db.delete_by_source(source_id)
                await self.vectors.adelete_by_source(source_id)
                if orphan_ids:
                    await self.vectors.adelete_entities_by_ids(orphan_ids)
                log.info("MySQL 回滚完成 source_id={} 清理孤儿实体={}", source_id, len(orphan_ids))
            except Exception as rollback_err:
                log.error("MySQL 回滚失败 source_id={} err={}（需手动清理）", source_id, rollback_err)
            raise

    async def delete_source(self, source_id: str) -> tuple[int, list[str]]:
        """删除 source：先删向量库 → 再删 MySQL → 最后删孤儿实体向量。

        顺序设计（先向量后 MySQL）：
          - 向量库先删 → 检索立即查不到该文档（检索依赖向量库）
          - MySQL 后删 → 页面稍后消失（页面依赖 MySQL）
          - 避免"页面显示已删除但检索还能搜到"的窗口
        
        跨库一致性：
          1. 标记 vector_synced=False（表示删除中，对账兜底）
          2. 向量库删除 chunks/titles/contents（失败不中断，靠对账兜底）
          3. MySQL 事务删除（含查孤儿实体 id）
          4. 孤儿实体向量删除（失败不中断，靠对账第三类兜底）
        """
        # 1. 标记删除中，对账任务会发现并兜底清理
        await self.db.mark_vector_synced(source_id, False)

        # 2. 先删向量库（检索立即不可见，失败靠对账兜底）
        try:
            await self.vectors.adelete_by_source(source_id)
        except Exception as e:
            log.error("向量库删除失败，靠对账兜底 source_id={} err={}", source_id, e)

        # 3. MySQL 事务删除（含查孤儿实体 id）
        n, orphan_entity_ids = await self.db.delete_by_source(source_id)

        # 4. 删孤儿实体向量（失败靠对账第三类兜底）
        if orphan_entity_ids:
            log.info("清理孤儿实体向量 数量={} ids={}", len(orphan_entity_ids), orphan_entity_ids)
            try:
                await self.vectors.adelete_entities_by_ids(orphan_entity_ids)
            except Exception as e:
                log.error("孤儿实体向量删除失败，靠对账兜底 ids={} err={}", orphan_entity_ids, e)
        return n, orphan_entity_ids

    async def close(self) -> None:
        if self._reconcile_task and not self._reconcile_task.done():
            self._reconcile_task.cancel()
            try:
                await self._reconcile_task
            except asyncio.CancelledError:
                pass
        await self.db.close()

    async def _reconcile(self) -> None:
        """对账 + 硬删除：清理四类数据。

        1. MySQL 中 vector_synced=False 的 source（入库崩溃残留 + 删除中崩溃残留）
        2. 向量库中有但 MySQL 已不存在的 source_id（删除流程中 source 被物理删但向量库删除失败）
        3. 向量库 entities 中有但 MySQL 已不存在的 entity_id（孤儿实体向量，删除步骤4失败残留）
        4. MySQL 中 deleted_at IS NOT NULL 的软删除事件（硬删除 MySQL + 向量库）
        """
        try:
            # 第一类：MySQL 标记未同步的 source
            unsynced = await self.db.find_unsynced_sources()
            for sid in unsynced:
                log.warning("对账：发现未同步向量的 source，清理中 source_id={}", sid)
                orphan_ids_first: list[str] = []
                try:
                    _, orphan_ids_first = await self.db.delete_by_source(sid)
                except Exception as e:
                    log.warning("对账：MySQL 删除 unsynced source 失败（可能已物理删除）source_id={} err={}", sid, e)
                try:
                    await self.vectors.adelete_by_source(sid)
                except Exception as e:
                    log.error("对账：清理向量库失败 source_id={} err={}", sid, e)
                # 同步清理孤儿实体向量（原本靠第三类兜底，现在即时清理缩短不一致窗口）
                if orphan_ids_first:
                    try:
                        await self.vectors.adelete_entities_by_ids(orphan_ids_first)
                    except Exception as e:
                        log.error("对账：清理孤儿实体向量失败 source_id={} ids={} err={}",
                                  sid, orphan_ids_first, e)

            # 第二类：向量库中有但 MySQL 已不存在的孤儿 source_id
            mysql_source_ids = set(await self.db.list_all_source_ids())
            vector_source_ids = set(await self.vectors.alist_source_ids())
            orphan_source_ids = vector_source_ids - mysql_source_ids
            for sid in orphan_source_ids:
                log.warning("对账：发现孤儿向量（MySQL 无此 source），清理 source_id={}", sid)
                try:
                    await self.vectors.adelete_by_source(sid)
                except Exception as e:
                    log.error("对账：清理孤儿向量失败 source_id={} err={}", sid, e)

            # 第三类：向量库 entities 中有但 MySQL 已不存在的孤儿 entity_id
            mysql_entity_ids = set(await self.db.list_all_entity_ids())
            vector_entity_ids = set(await self.vectors.alist_all_entity_ids())
            orphan_entity_ids = vector_entity_ids - mysql_entity_ids
            if orphan_entity_ids:
                log.warning("对账：发现孤儿实体向量 数量={} ids={}", len(orphan_entity_ids), orphan_entity_ids)
                try:
                    await self.vectors.adelete_entities_by_ids(list(orphan_entity_ids))
                except Exception as e:
                    log.error("对账：清理孤儿实体向量失败 err={}", e)

            # 第四类：软删除事件的物理清理
            event_ids, hd_orphan_ids = await self.db.hard_delete_soft_deleted_events()
            if event_ids:
                try:
                    await self.vectors.adelete_event_ids(event_ids)
                except Exception as e:
                    log.error("对账：事件向量硬删除失败 ids={} err={}", event_ids, e)
                if hd_orphan_ids:
                    try:
                        await self.vectors.adelete_entities_by_ids(hd_orphan_ids)
                    except Exception as e:
                        log.error("对账：孤儿实体向量硬删除失败 ids={} err={}", hd_orphan_ids, e)
        except Exception as e:
            log.warning("对账失败（可能首次启动无数据）err={}", e)

    def _start_reconcile_loop(self) -> None:
        interval = self.cfg.ingest.reconcile_interval
        if interval <= 0:
            log.info("定时对账已禁用（AISAG_RECONCILE_INTERVAL=0）")
            return
        self._reconcile_task = asyncio.create_task(
            self._reconcile_loop(interval), name="sag-reconcile")
        log.info("后台定时对账已启动 间隔={}s", interval)

    async def _reconcile_loop(self, interval: int) -> None:
        while True:
            try:
                await asyncio.sleep(interval)
                await self._reconcile()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("定时对账失败 err={}", e)