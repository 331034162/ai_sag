"""API 模块：基于 FastAPI 对外暴露文档上传、下载、删除、更新、列表、检索、问答接口。

全链路异步：所有 DB/检索/入库调用均为 await。

启动：
    python -m ai_sag.api                    # 默认 0.0.0.0:8777
    python -m ai_sag.api --host 0.0.0.0 --port 8777
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
import time
from typing import Any, Literal, get_args

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

try:
    from .base import Config
    from .base.logger import generate_trace_id, get_logger, init_logger, reset_trace_id, set_trace_id
    from .doc_parser.image.ocr import OCRBackend
    from .ingest import IngestPipeline
    from .retrieval.qa_engine import QAEngine
except ImportError:
    import sys
    from pathlib import Path
    _ROOT = str(Path(__file__).resolve().parents[1])
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from ai_sag.base import Config
    from ai_sag.base.logger import generate_trace_id, get_logger, init_logger, reset_trace_id, set_trace_id
    from ai_sag.doc_parser.image.ocr import OCRBackend
    from ai_sag.ingest import IngestPipeline
    from ai_sag.retrieval.qa_engine import QAEngine

init_logger()
log = get_logger()

_ingest: IngestPipeline | None = None
_qa: QAEngine | None = None
_cfg: Config | None = None


def _config() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = Config()
    return _cfg


def _ingest_pipeline() -> IngestPipeline:
    global _ingest
    if _ingest is None:
        _ingest = IngestPipeline(_config())
    return _ingest


def _upload_tmp_dir() -> str | None:
    """获取上传临时目录。

    读取 AISAG_UPLOAD_TMP_DIR 配置（位于 PdfDocParserConfig）：
    - 留空（默认）→ 返回 None，使用系统默认目录（Windows %TEMP%，Linux /tmp）；
    - 指定路径 → 自动创建（exist_ok），返回该路径作为 NamedTemporaryFile(dir=...) 参数；
    - 创建失败 → 回退到 None，保证入库不中断。
    """
    d = (_config().doc_parser.upload_tmp_dir or "").strip()
    if not d:
        return None
    try:
        os.makedirs(d, exist_ok=True)
        return d
    except OSError:
        return None


def _make_upload_tmp_path(filename: str, tmp_dir: str | None) -> str:
    """生成上传临时文件路径：{原文件名前缀}_{日期}_{时分秒毫秒}_{随机数}.{后缀}

    命名规则示例：
        新一代信贷_参数管理操作手册_20260704_115102345_483.docx

    便于：
    - 从文件名即可看出上传时间和原始文档，方便排查问题；
    - 毫秒+随机数保证并发上传不冲突；
    - 出问题后能快速定位是哪次上传的临时文件。

    Args:
        filename: 原始上传文件名（如 "新一代信贷_xxx.docx"）
        tmp_dir: 临时目录（None 时用系统默认 tempfile.gettempdir()）
    """
    import datetime
    import random
    stem = os.path.splitext(os.path.basename(filename))[0]
    # 文件名前缀只保留安全字符（中文/字母/数字/下划线/横线），其余替换为 _
    safe_stem = "".join(c if c.isalnum() or c in "_-" else "_" for c in stem)[:40]
    ext = os.path.splitext(filename)[1].lower()  # 含 .，如 .docx
    now = datetime.datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S") + f"{now.microsecond // 1000:03d}"
    rnd = random.randint(0, 999)
    name = f"{safe_stem}_{ts}_{rnd:03d}{ext}"
    if tmp_dir:
        return os.path.join(tmp_dir, name)
    import tempfile as _tf
    return os.path.join(_tf.gettempdir(), name)


def _qa_engine() -> QAEngine:
    global _qa
    if _qa is None:
        ingest = _ingest_pipeline()
        _qa = QAEngine(_config(), db=ingest.db, vectors=ingest.vectors)
    return _qa


def _parse_bool_form(value: str | None) -> bool | None:
    """解析表单里的布尔字段。

    前端 FormData 只能传字符串，这里统一转换：
    - "true"/"1"/"on"（不区分大小写）→ True
    - "false"/"0"/"off"/""（不区分大小写）→ False
    - None → None（用配置默认值）
    其他无法识别的值也视为 None（用配置默认），宽松容错避免误关 OCR。
    """
    if value is None:
        return None
    v = value.strip().lower()
    if v in ("true", "1", "on"):
        return True
    if v in ("false", "0", "off"):
        return False
    return None


# OCR 后端白名单：从 doc_parser 的 OCRBackend Literal 类型提取合法值集合
# 非法值一律归一化为 None（用配置默认），避免注入未受控参数
_SUPPORTED_OCR_BACKENDS: set[str] = set(get_args(OCRBackend))


def _normalize_ocr_backend(value: str | None) -> str | None:
    """归一化 OCR 后端表单字段。

    - "rapidocr" / "paddleocr"（不区分大小写）→ 原样返回小写
    - None / 空串 / 其他非法值 → None（用配置默认值）
    """
    if value is None:
        return None
    v = value.strip().lower()
    if v in _SUPPORTED_OCR_BACKENDS:
        return v
    return None


# ---------------- 请求/响应模型 ----------------

class TextUploadRequest(BaseModel):
    title: str
    content: str
    source_name: str | None = None


class SearchRequest(BaseModel):
    query: str
    source_ids: list[str] | None = None
    fusion: Literal["supplement", "concat"] | None = None


class AskRequest(BaseModel):
    query: str
    source_ids: list[str] | None = None
    fusion: Literal["supplement", "concat"] | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    query: str
    history: list[ChatMessage] = Field(default_factory=list)
    source_ids: list[str] | None = None
    fusion: Literal["supplement", "concat"] | None = None


class UpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None


# ---------------- 应用工厂 ----------------

def create_app() -> FastAPI:
    app = FastAPI(title="ai_sag API", version="0.2.0",
                  description="SAG 事件-实体关联知识库 API（全链路异步）")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=True,
        allow_methods=["*"], allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        trace_id = request.headers.get("X-Trace-Id") or generate_trace_id()
        token = set_trace_id(trace_id)
        start = time.perf_counter()
        path = request.url.path
        method = request.method
        try:
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - start) * 1000
            log.info("REQ {} {} {} -> {} cost={:.1f}ms",
                     trace_id, method, path, response.status_code, elapsed_ms)
            response.headers["X-Trace-Id"] = trace_id
            return response
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            log.error("REQ {} {} {} -> ERROR cost={:.1f}ms err={}",
                      trace_id, method, path, elapsed_ms, e)
            raise
        finally:
            reset_trace_id(token)

    log.info("ai_sag API 启动完成")

    @app.on_event("startup")
    async def _warmup() -> None:
        log.info("预加载入库流水线与问答引擎（首次会加载 embedding 模型，请稍候）...")
        ingest = _ingest_pipeline()
        await ingest.init()
        _qa_engine()
        log.info("预加载完成，所有接口可立即响应")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        global _qa, _ingest
        if _qa is not None:
            await _qa.close()
            _qa = None
        if _ingest is not None:
            await _ingest.close()
            _ingest = None
        log.info("已关闭入库流水线与问答引擎，释放资源")

    # ---- 文档上传（文件）----
    @app.post("/api/documents", summary="上传文档（文件）")
    async def upload_file(
        file: UploadFile = File(...),
        source_name: str | None = Form(None),
        ocr_images: str | None = Form(None, description="是否对 docx/pdf 中的图片做 OCR：true/false，留空用配置默认"),
        ocr_backend: str | None = Form(None, description="OCR 引擎：rapidocr/paddleocr，留空用配置默认"),
    ):
        suffix = os.path.splitext(file.filename or "")[1].lower().lstrip(".")
        if suffix not in ("md", "markdown", "txt", "docx", "pdf", "xlsx", "xls", "csv"):
            raise HTTPException(400, f"不支持的文件类型: .{suffix}（支持 .md/.txt/.docx/.pdf/.xlsx/.csv）")
        raw = await file.read()
        if not raw:
            raise HTTPException(400, "文件内容为空")

        # 解析 ocr_images 表单字段：true/false 字符串转布尔，其他（含 None）视为用配置默认
        ocr_flag = _parse_bool_form(ocr_images)
        # 归一化 ocr_backend：白名单校验，非法值视为用配置默认
        ocr_engine = _normalize_ocr_backend(ocr_backend)

        # 计算文件 MD5
        file_md5 = hashlib.md5(raw).hexdigest()
        filename = source_name or file.filename

        # 检查重复：相同文件名 + 相同 MD5 的未归档文档已存在则拒绝
        pipe = _ingest_pipeline()
        existing_id = await pipe.db.check_duplicate(filename, file_md5)
        if existing_id:
            log.warning("重复上传拦截 file={} name={} md5={} existing_source_id={}",
                        file.filename, filename, file_md5, existing_id)
            raise HTTPException(409, f"文件已存在（文件名和内容均相同），source_id={existing_id}")

        with open(_make_upload_tmp_path(file.filename, _upload_tmp_dir()), "wb") as f:
            f.write(raw)
            tmp_path = f.name
        try:
            log.info("开始入库 file={} size={}B md5={} ocr_images={} ocr_backend={}",
                     file.filename, len(raw), file_md5, ocr_flag, ocr_engine)
            source_id = await pipe.ingest_file(tmp_path,
                                                source_name=filename,
                                                title=file.filename,
                                                md5=file_md5,
                                                ocr_images=ocr_flag,
                                                ocr_backend=ocr_engine)
            log.info("入库完成 source_id={} file={}", source_id, file.filename)
        except HTTPException:
            raise
        except Exception as e:
            code = getattr(e, 'args', [None])[0]
            if code == 1062:
                log.warning("重复上传拦截 file={} name={} md5={}", file.filename, source_name, file_md5)
                raise HTTPException(409, "文件已存在（文件名和内容均相同）")
            log.exception("入库失败 file={} err={}", file.filename, e)
            raise HTTPException(500, f"入库失败: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return {"source_id": source_id, "filename": file.filename, "size": len(raw)}

    # ---- 文档上传（纯文本）----
    @app.post("/api/documents/text", summary="上传文档（纯文本）")
    async def upload_text(req: TextUploadRequest):
        if not req.content.strip():
            raise HTTPException(400, "内容不能为空")

        content_md5 = hashlib.md5(req.content.encode("utf-8")).hexdigest()
        source_name = req.source_name or req.title

        # 检查重复：相同名称 + 相同 MD5 的未归档文档已存在则拒绝
        pipe = _ingest_pipeline()
        existing_id = await pipe.db.check_duplicate(source_name, content_md5)
        if existing_id:
            log.warning("重复上传拦截（文本） title={} name={} md5={} existing_source_id={}",
                        req.title, source_name, content_md5, existing_id)
            raise HTTPException(409, f"相同内容的文本已存在，source_id={existing_id}")

        try:
            source_id = await pipe.ingest_text(req.title, req.content, source_name=source_name, md5=content_md5)
        except HTTPException:
            raise
        except Exception as e:
            code = getattr(e, 'args', [None])[0]
            if code == 1062:
                log.warning("重复上传拦截（文本） title={} name={} md5={}", req.title, source_name, content_md5)
                raise HTTPException(409, "相同内容的文本已存在")
            raise HTTPException(500, f"入库失败: {e}")
        return {"source_id": source_id, "title": req.title}

    # ---- 文档列表 ----
    @app.get("/api/documents", summary="文档列表")
    async def list_documents(
        keyword: str | None = Query(None),
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
        include_archived: bool = Query(False),
    ):
        db = _ingest_pipeline().db
        items = await db.list_sources(include_archived=include_archived, keyword=keyword,
                                      limit=limit, offset=offset)
        total = await db.count_sources(include_archived=include_archived)
        return {"total": total, "limit": limit, "offset": offset, "items": items}

    # ---- 文档查询（全文搜索文档原文）----
    @app.get("/api/documents/search", summary="文档全文查询")
    async def search_documents(
        q: str = Query(..., min_length=1, description="搜索关键词"),
        source_ids: str | None = Query(None, description="逗号分隔的 source_id 过滤"),
        limit: int = Query(20, ge=1, le=100),
        context_size: int = Query(80, ge=10, le=500, description="匹配片段前后字符数"),
    ):
        ids = [s.strip() for s in source_ids.split(",") if s.strip()] if source_ids else None
        db = _ingest_pipeline().db
        items = await db.search_documents(q, source_ids=ids, limit=limit, context_size=context_size)
        return {"query": q, "total": len(items), "items": items}

    # ---- 文档详情 ----
    @app.get("/api/documents/{source_id}", summary="文档详情")
    async def get_document(source_id: str):
        db = _ingest_pipeline().db
        src = await db.get_source(source_id)
        if not src:
            raise HTTPException(404, "文档不存在")
        doc = await db.get_document_by_source(source_id) or {}
        return {
            "source": src,
            "document": doc,
            "stats": {
                "chunks": await db.count_chunks_by_source(source_id),
                "events": await db.count_events_by_source(source_id),
            },
        }

    # ---- 文档下载 ----
    @app.get("/api/documents/{source_id}/download", summary="下载文档原文")
    async def download_document(source_id: str):
        db = _ingest_pipeline().db
        src = await db.get_source(source_id)
        if not src:
            raise HTTPException(404, "文档不存在")
        doc = await db.get_document_by_source(source_id)
        if not doc:
            raise HTTPException(404, "原文不可用")
        content = doc.get("content") or ""
        title = doc.get("title") or source_id
        buf = io.BytesIO(content.encode("utf-8"))
        headers = {"Content-Disposition": f'attachment; filename="{title}.md"'}
        return StreamingResponse(buf, media_type="text/markdown", headers=headers)

    # ---- 文档更新（元信息）----
    @app.patch("/api/documents/{source_id}", summary="更新文档元信息")
    async def update_document(source_id: str, req: UpdateRequest):
        db = _ingest_pipeline().db
        if not await db.get_source(source_id):
            raise HTTPException(404, "文档不存在")
        ok = await db.update_source(source_id, name=req.name, description=req.description)
        return {"updated": ok, "source_id": source_id}

    # ---- 文档删除 ----
    @app.delete("/api/documents/{source_id}", summary="删除文档")
    async def delete_document(source_id: str):
        pipe = _ingest_pipeline()
        if not await pipe.db.get_source(source_id):
            raise HTTPException(404, "文档不存在")
        n, orphan_ids = await pipe.delete_source(source_id)
        log.info("删除文档 source_id={} 删除事件数={} 清理孤儿实体={}",
                 source_id, n, len(orphan_ids))
        return {"source_id": source_id, "deleted_events": n,
                "deleted_orphan_entities": len(orphan_ids)}

    # ---- 文档重建（更新内容：先入新后删旧，避免空洞窗口）----
    @app.put("/api/documents/{source_id}", summary="更新文档内容（重建）")
    async def rebuild_document(
        source_id: str,
        file: UploadFile = File(...),
        ocr_images: str | None = Form(None, description="是否对 docx/pdf 中的图片做 OCR：true/false，留空用配置默认"),
        ocr_backend: str | None = Form(None, description="OCR 引擎：rapidocr/paddleocr，留空用配置默认"),
    ):
        """重建策略：先用新 source_id 入库，成功后再删旧数据。

        避免先删后入的空洞窗口：旧数据在入库完成前始终可检索，
        入库失败不影响旧数据可用性。
        """
        suffix = os.path.splitext(file.filename or "")[1].lower().lstrip(".")
        if suffix not in ("md", "markdown", "txt", "docx", "pdf", "xlsx", "xls", "csv"):
            raise HTTPException(400, f"不支持的文件类型: .{suffix}")
        raw = await file.read()
        if not raw:
            raise HTTPException(400, "文件内容为空")

        ocr_flag = _parse_bool_form(ocr_images)
        ocr_engine = _normalize_ocr_backend(ocr_backend)
        file_md5 = hashlib.md5(raw).hexdigest()

        with open(_make_upload_tmp_path(file.filename, _upload_tmp_dir()), "wb") as f:
            f.write(raw)
            tmp_path = f.name
        pipe = _ingest_pipeline()
        old_source = await pipe.db.get_source(source_id)
        if not old_source:
            raise HTTPException(404, "文档不存在")

        # 文件内容未变化时无需重建
        if file_md5 and file_md5 == old_source.get("md5", ""):
            return {"source_id": source_id, "rebuilt": False, "message": "文件内容未变化，跳过重建"}

        try:
            # 1. 先用新 source_id 入库（旧数据不受影响）
            new_id = await pipe.ingest_file(tmp_path, source_name=old_source["name"], md5=file_md5,
                                            ocr_images=ocr_flag, ocr_backend=ocr_engine)
            # 2. 入库成功后删除旧数据（失败也不影响新数据）
            await pipe.delete_source(source_id)
        except Exception as e:
            # 入库失败时旧数据仍在，不影响使用
            raise HTTPException(500, f"重建失败: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return {"source_id": new_id, "old_source_id": source_id, "rebuilt": True}

    # ---- 检索 ----
    @app.post("/api/search", summary="检索")
    async def search(req: SearchRequest):
        log.info("检索请求 query={!r} source_ids={} fusion={}",
                 req.query, req.source_ids, req.fusion)
        try:
            result = await _qa_engine().search(req.query, source_ids=req.source_ids, fusion=req.fusion)
            log.info("检索完成 query={!r} 命中chunk={} seed_events={} expanded_events={}",
                     req.query, len(result.sections),
                     len(result.trace.seed_event_ids) if result.trace else 0,
                     len(result.trace.expanded_event_ids) if result.trace else 0)
        except Exception as e:
            log.exception("检索失败 query={!r} err={}", req.query, e)
            raise HTTPException(500, f"检索失败: {e}")
        return await _serialize_search_result(result)

    # ---- 问答 ----
    @app.post("/api/ask", summary="问答")
    async def ask(req: AskRequest):
        log.info("问答请求 query={!r} source_ids={} fusion={}",
                 req.query, req.source_ids, req.fusion)
        try:
            answer, result = await _qa_engine().ask(req.query, source_ids=req.source_ids, fusion=req.fusion)
            log.info("问答完成 query={!r} 命中chunk={} answer_len={}",
                     req.query, len(result.sections), len(answer))
        except Exception as e:
            log.exception("问答失败 query={!r} err={}", req.query, e)
            raise HTTPException(500, f"问答失败: {e}")
        return {"answer": answer, **await _serialize_search_result(result)}

    # ---- 多轮对话 ----
    @app.post("/api/chat", summary="多轮对话")
    async def chat(req: ChatRequest):
        log.info("对话请求 query={!r} history_len={} source_ids={} fusion={}",
                 req.query, len(req.history), req.source_ids, req.fusion)
        try:
            history = [m.model_dump() for m in req.history]
            answer, result = await _qa_engine().chat(
                req.query, history=history,
                source_ids=req.source_ids, fusion=req.fusion,
            )
            log.info("对话完成 query={!r} 命中chunk={} answer_len={}",
                     req.query, len(result.sections), len(answer))
        except Exception as e:
            log.exception("对话失败 query={!r} err={}", req.query, e)
            raise HTTPException(500, f"对话失败: {e}")
        return {"answer": answer, **await _serialize_search_result(result)}

    # ---- 多轮对话（流式） ----
    @app.post("/api/chat/stream", summary="多轮对话（SSE 流式）")
    async def chat_stream(req: ChatRequest):
        log.info("流式对话请求 query={!r} history_len={} source_ids={} fusion={}",
                 req.query, len(req.history), req.source_ids, req.fusion)

        async def event_stream():
            try:
                history = [m.model_dump() for m in req.history]
                result, gen = await _qa_engine().chat_stream(
                    req.query, history=history,
                    source_ids=req.source_ids, fusion=req.fusion,
                )
                log.info("流式对话检索完成 query={!r} 命中chunk={}",
                         req.query, len(result.sections))
                meta = await _serialize_search_result(result)
                yield _sse("meta", {"sections": meta["sections"], "trace": meta["trace"]})
                async for delta in gen:
                    if delta:
                        yield _sse("delta", {"content": delta})
                yield _sse("done", {})
            except Exception as e:
                log.exception("流式对话失败 query={!r} err={}", req.query, e)
                yield _sse("error", {"message": str(e)})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # ---- 统计 ----
    @app.get("/api/stats", summary="统计信息")
    async def stats():
        db = _ingest_pipeline().db
        total = await db.count_sources()
        archived = await db.count_sources(include_archived=True) - total
        return {"sources": total, "archived_sources": archived}

    # ---- 健康检查 ----
    @app.get("/api/health", summary="健康检查")
    async def health():
        try:
            await _ingest_pipeline().db.ping()
            return {"status": "ok"}
        except Exception as e:
            return JSONResponse({"status": "error", "detail": str(e)}, status_code=503)

    return app


async def _serialize_search_result(result) -> dict[str, Any]:
    """序列化检索结果，批量补充 source_name 供前端溯源展示。"""
    sections_raw = [{
        "chunk_id": s.chunk_id, "source_id": s.source_id, "document_id": s.document_id,
        "heading": s.heading, "content": s.content, "rank": s.rank, "score": s.score,
    } for s in result.sections]
    # 批量查 source_name，补充到每个 section 中供前端溯源
    if sections_raw:
        source_ids = list({s["source_id"] for s in sections_raw if s["source_id"]})
        if source_ids:
            db = _ingest_pipeline().db
            name_map = await db.get_source_names_by_ids(source_ids)
            for s in sections_raw:
                s["source_name"] = name_map.get(s["source_id"], "")
    trace = None
    if result.trace:
        t = result.trace
        trace = {
            "query": t.query, "query_entities": t.query_entities,
            "expanded_query_entities": t.expanded_query_entities,
            "seed_event_ids": t.seed_event_ids, "expanded_event_ids": t.expanded_event_ids,
            "rerank_candidate_ids": t.rerank_candidate_ids,
            "reranked_ids": t.reranked_ids, "fallback": t.fallback,
        }
    return {"sections": sections_raw, "trace": trace, "total": len(sections_raw)}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="ai_sag API 服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = parser.parse_args()
    import uvicorn
    log_config = {
        "version": 1, "disable_existing_loggers": False,
        "loggers": {
            "uvicorn": {"level": "INFO", "propagate": False},
            "uvicorn.error": {"level": "INFO", "propagate": False},
            "uvicorn.access": {"level": "INFO", "propagate": False},
        },
    }
    uvicorn.run("ai_sag.api:app", host=args.host, port=args.port,
                reload=args.reload, log_config=log_config)


if __name__ == "__main__":
    main()