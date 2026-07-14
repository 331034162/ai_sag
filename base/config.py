"""配置：复用项目 .env，组件配置结构化为嵌套子 dataclass，便于灵活切换与扩展。

环境变量前缀：
- SAG_     复用项目原有配置（MySQL/Embedding/LLM）
- AISAG_   ai_sag 模块专用配置（后端选择、切分、向量库路径等）

配置结构：
    cfg.mysql.host / cfg.mysql.port ...
    cfg.embedding.backend / cfg.embedding.bge_model_path ...
    cfg.llm.backend / cfg.llm.model ...
    cfg.vector_store.backend / cfg.vector_store.chroma_path ...
    cfg.splitter.mode / cfg.splitter.chunk_size ...
    cfg.search.similarity_threshold / cfg.search.fusion ...
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

# ---------- 定位 & 加载 .env ----------
# 策略：多级回退，不依赖固定目录层级，适配任意 clone 路径和包改名场景。
#
# 回退顺序：
#   1. config.py 向上搜索：parent[0]=base/ → parent[1]=ai_sag/ → … → parent[-1]
#      （最多回溯 5 级，防止在根目录无休止搜索）
#   2. find_dotenv：从 CWD 往上自动搜索（Python-dotenv 内置）
#   3. 都没找到 → 给出明确提示，不静默失败

_THIS_FILE = Path(__file__).resolve()
_ENV_PATH = None

# 1) 从 config.py 所在目录逐级向上搜索 .env
for _ancestor in _THIS_FILE.parents:
    _candidate = _ancestor / ".env"
    if _candidate.exists():
        _ENV_PATH = _candidate
        break

# 2) 兜底：用 find_dotenv 在 CWD 往上搜
if _ENV_PATH is None:
    _ENV_PATH = find_dotenv(raise_error_if_not_found=False)
    if _ENV_PATH:
        _ENV_PATH = Path(_ENV_PATH)

if _ENV_PATH is not None:
    load_dotenv(_ENV_PATH)
else:
    # 完全找不到 .env 时也不要静默崩溃，给一个清晰的提示
    print(
        "[WARNING] 未找到 .env 文件。你可以在项目根目录创建 .env（参考 .env.example），"
        "或通过系统环境变量设置所需配置。"
    )

# 解析出项目根（_PKG_DIR）：优先用 .env 所在目录，否则回退到 config.py 上两级
if _ENV_PATH is not None:
    _PKG_DIR = _ENV_PATH.parent.resolve()
else:
    _PKG_DIR = _THIS_FILE.parents[1].resolve()


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


# ---------- 关键变量缺失检查 ----------
_MISSING = []
for _k, _desc in [("SAG_MYSQL_USER", "MySQL 用户名"), ("SAG_MYSQL_PASSWORD", "MySQL 密码"),
                   ("SAG_LLM_API_KEY", "LLM API Key")]:
    if not _env(_k):
        _MISSING.append(f"  - {_k}（{_desc}）")
if _MISSING:
    _hint = ""
    if _ENV_PATH is None:
        _hint = (
            "\n\n提示：项目根目录未找到 .env 文件。\n"
            "你可以复制 .env.example 为 .env 并填入你的配置：\n"
            "  cp .env.example .env"
        )
    elif not _ENV_PATH.exists():
        _hint = f"\n\n提示：指定的 .env 文件不存在：\n  {_ENV_PATH}"
    else:
        _hint = f"\n\n提示：.env 文件已找到（{_ENV_PATH}），但缺少上述变量，请补充。"
    raise RuntimeError(
        f"缺少必要的环境变量：\n" + "\n".join(_MISSING) + _hint
    )


@dataclass
class MysqlConfig:
    host: str = field(default_factory=lambda: _env("SAG_MYSQL_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(_env("SAG_MYSQL_PORT", "3306")))
    user: str = field(default_factory=lambda: _env("SAG_MYSQL_USER", "root"))
    password: str = field(default_factory=lambda: _env("SAG_MYSQL_PASSWORD", ""))
    database: str = field(default_factory=lambda: _env("SAG_MYSQL_DATABASE", "sag"))
    # 异步连接池大小（全链路异步改造）
    pool_size: int = field(default_factory=lambda: int(_env("AISAG_MYSQL_POOL_SIZE", "10")))
    # 超出 pool_size 后最多可创建的连接数
    max_overflow: int = field(default_factory=lambda: int(_env("AISAG_MYSQL_MAX_OVERFLOW", "5")))
    # 获取连接的超时时间（秒）
    pool_timeout: float = field(default_factory=lambda: float(_env("AISAG_MYSQL_POOL_TIMEOUT", "30")))
    # 连接回收时间（秒），防止 MySQL 8 小时断连
    pool_recycle: int = field(default_factory=lambda: int(_env("AISAG_MYSQL_POOL_RECYCLE", "3600")))


@dataclass
class EmbeddingConfig:
    backend: str = field(default_factory=lambda: _env("SAG_EMBEDDING_BACKEND", "bge").lower())
    bge_model_path: str = field(default_factory=lambda: _env("SAG_BGE_MODEL_PATH", ""))
    qwen3_model_path: str = field(default_factory=lambda: _env("SAG_EMBEDDING_MODEL_PATH", ""))
    device: str = field(default_factory=lambda: _env("SAG_EMBEDDING_DEVICE", "cpu"))


@dataclass
class LlmConfig:
    backend: str = field(default_factory=lambda: _env("AISAG_LLM_BACKEND", "deepseek").lower())
    base_url: str = field(default_factory=lambda: _env("SAG_LLM_BASE_URL", "https://api.deepseek.com"))
    api_key: str = field(default_factory=lambda: _env("SAG_LLM_API_KEY", ""))
    model: str = field(default_factory=lambda: _env("SAG_LLM_MODEL", "deepseek-chat"))
    timeout: float = field(default_factory=lambda: float(_env("SAG_LLM_TIMEOUT", "120")))
    max_retries: int = field(default_factory=lambda: int(_env("SAG_LLM_MAX_RETRIES", "3")))


@dataclass
class VectorStoreConfig:
    backend: str = field(default_factory=lambda: _env("AISAG_VECTOR_STORE_BACKEND", "chroma").lower())
    chroma_path: str = field(default_factory=lambda: _env(
        "AISAG_CHROMA_PATH", str(_PKG_DIR / ".chroma")))


@dataclass
class SplitterConfig:
    # semantic：按语义相似度切分，chunk 语义完整最适合 SAG 事件抽取（默认）
    # auto：按文档类型自动选（md→markdown，其余→sentence），速度快但语义可能被截断
    mode: str = field(default_factory=lambda: _env("AISAG_SPLITTER_MODE", "semantic").lower())
    chunk_size: int = field(default_factory=lambda: int(_env("AISAG_CHUNK_SIZE", "8192")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("AISAG_CHUNK_OVERLAP", "800")))
    language: str = field(default_factory=lambda: _env("AISAG_SPLITTER_LANGUAGE", "python"))
    # semantic 模式：语义断点分位阈值（0-100），值越小切得越碎。
    # 95=保守（仅差异最大的 5% 处断句），80=激进（20% 处断句，chunk 更小更均匀）
    breakpoint_percentile_threshold: int = field(
        default_factory=lambda: int(_env("AISAG_BREAKPOINT_PERCENTILE", "95")))


@dataclass
class LogConfig:
    """日志配置：参考 loguru 实践，支持控制台/文件双输出、异步写入、轮转压缩。"""
    level: str = field(default_factory=lambda: _env("AISAG_LOG_LEVEL", "INFO").upper())
    log_dir: str = field(default_factory=lambda: _env("AISAG_LOG_DIR", str(_PKG_DIR / "logs")))
    # 轮转：loguru 原生语法，如 "500 MB" 或 "00:00"（每天）
    rotation: str = field(default_factory=lambda: _env("AISAG_LOG_ROTATION", "500 MB"))
    # 保留：如 "30 days" 或 10（保留文件数）
    retention: str = field(default_factory=lambda: _env("AISAG_LOG_RETENTION", "30 days"))
    # 控制台是否启用彩色输出（容器环境建议关闭，避免 ANSI 代码）
    colorize: bool = field(default_factory=lambda: _env("AISAG_LOG_COLORIZE", "false").lower() == "true")


@dataclass
class SearchConfig:
    # 事件召回/粗排/基线 chunk 召回的相似度阈值（论文 3.3 Path B：0.4）
    similarity_threshold: float = field(
        default_factory=lambda: float(_env("AISAG_SIMILARITY_THRESHOLD", "0.4")))
    # 实体向量扩展阈值（Path A，独立于事件召回阈值）
    entity_expand_threshold: float = field(
        default_factory=lambda: float(_env("AISAG_ENTITY_EXPAND_THRESHOLD", "0.3")))
    max_hops: int = field(default_factory=lambda: int(_env("AISAG_MAX_HOPS", "2")))
    # 多跳扩展子策略：multi（固定跳数）/ hopllm（每跳相似度动态停止）
    sub_strategy: str = field(default_factory=lambda: _env("AISAG_SUB_STRATEGY", "hopllm").lower())
    # hopllm 动态停止：新跳事件内容相似度低于此值则终止扩展
    hop_relevance_threshold: float = field(
        default_factory=lambda: float(_env("AISAG_HOP_RELEVANCE_THRESHOLD", "0.15")))
    # hopllm 每跳重排后保留的种子事件数（对齐旧版 topK 剪枝）
    hop_seed_topk: int = field(default_factory=lambda: int(_env("AISAG_HOP_SEED_TOPK", "8")))
    # BFS 扩展每跳边界实体数量上限（论文 Section 4.4：entity frontier pruning budget=100）
    entity_frontier_budget: int = field(
        default_factory=lambda: int(_env("AISAG_ENTITY_FRONTIER_BUDGET", "100")))
    # 是否启用边界实体相关性筛选（方案1+3）：开启后 BFS 每跳对新增实体做综合评分截断，
    # 综合分 = α*IDF(度数倒数) + (1-α)*query相似度，优先保留低频且与query语义相关的实体，
    # 抑制"众邦银行"等高频枢纽实体桥接噪声事件。关闭则退回原随机截断。
    entity_frontier_filter: bool = field(
        default_factory=lambda: _env("AISAG_ENTITY_FRONTIER_FILTER", "true").lower() == "true")
    # 综合评分中 query 相似度的权重 α（0~1，越大越偏向语义相关性，越小越偏向低频优先）
    entity_frontier_query_weight: float = field(
        default_factory=lambda: float(_env("AISAG_ENTITY_FRONTIER_QUERY_WEIGHT", "0.6")))
    # 粗排相似度阈值（设为 0 则关闭，对齐旧版仅排序截断）
    coarse_threshold: float = field(
        default_factory=lambda: float(_env("AISAG_COARSE_THRESHOLD", "0")))
    max_events: int = field(default_factory=lambda: int(_env("AISAG_MAX_EVENTS", "100")))
    rerank_top_k: int = field(default_factory=lambda: int(_env("AISAG_RERANK_TOP_K", "5")))
    # 送入 LLM 重排的最大候选数，对齐论文图示 Top-100 设计（P1-4 修复）
    rerank_candidate_limit: int = field(
        default_factory=lambda: int(_env("AISAG_RERANK_CANDIDATE_LIMIT", "100")))
    max_sections: int = field(default_factory=lambda: int(_env("AISAG_MAX_SECTIONS", "5")))
    fusion: str = field(default_factory=lambda: _env("AISAG_FUSION", "concat").lower())

    # 种子事件向量召回策略：title（标题向量）/ summary（摘要向量）/ mixed（两者并发合并）
    seed_recall: str = field(default_factory=lambda: _env("AISAG_SEED_RECALL", "mixed").lower())

    # 查询重写时保留的最大对话轮数（每轮含 user+assistant）
    rewrite_max_rounds: int = field(
        default_factory=lambda: int(_env("AISAG_REWRITE_MAX_ROUNDS", "5")))


@dataclass
class IngestConfig:
    """入库流程配置。"""
    # 事件抽取是否并行：True 时并发抽取大幅提速，各 chunk 独立抽取（对齐论文设计）。
    extract_parallel: bool = field(
        default_factory=lambda: _env("AISAG_EXTRACT_PARALLEL", "false").lower() == "true")
    # 并行抽取时的最大并发 worker 数量
    extract_parallel_workers: int = field(
        default_factory=lambda: int(_env("AISAG_EXTRACT_PARALLEL_WORKERS", "4")))
    # 后台定时对账间隔秒数：定期清理"有 MySQL 无向量"的孤儿数据 + 硬删除软删事件。
    # 设为 0 则禁用定时对账（仅启动时对账一次）。
    reconcile_interval: int = field(
        default_factory=lambda: int(_env("AISAG_RECONCILE_INTERVAL", "300")))
    # 并发入库上限：限制同时入库的文档数，防止 LLM API rate limit / embedding OOM。
    # 设为 1 则串行入库，设为 3~5 可平衡吞吐与资源安全。
    concurrency: int = field(
        default_factory=lambda: int(_env("AISAG_INGEST_CONCURRENCY", "2")))
    # 事件抽取单 chunk 最大重试次数：瞬时故障（rate limit/网络抖动）自动重试，
    # 耗尽后抛 ExtractionError 终止入库，不会写入低质量 fallback 数据。
    extract_max_retries: int = field(
        default_factory=lambda: int(_env("AISAG_EXTRACT_MAX_RETRIES", "2")))
    # 事件标题（title）的字数上限：通过 system prompt 传递给 LLM，约束其输出长度。
    title_max_chars: int = field(
        default_factory=lambda: int(_env("AISAG_TITLE_MAX_CHARS", "100")))
    # 事件摘要（summary）的字数上限：通过 system prompt 传递给 LLM，约束其输出长度。
    summary_max_chars: int = field(
        default_factory=lambda: int(_env("AISAG_SUMMARY_MAX_CHARS", "500")))
    # 是否对事件摘要（summary）生成向量入库（默认开启，后续可用于种子事件召回）
    embed_summary: bool = field(
        default_factory=lambda: _env("AISAG_EMBED_SUMMARY", "true").lower() == "true")
    # 是否启用文档级文体识别（方案 A+B 前置步骤）：开启后每文档多一次 LLM 调用判断文体，
    # 驱动抽取时的边界判别规则和 role 词汇表。关闭则统一用 generic 文体，省一次调用。
    genre_detect: bool = field(
        default_factory=lambda: _env("AISAG_GENRE_DETECT", "true").lower() == "true")


@dataclass
class PdfDocParserConfig:
    """文档解析配置（loader 调用 doc_parser.* 时使用，PDF/Word/Excel 共用 OCR 与临时目录）。"""
    # OCR 后端引擎：rapidocr（默认，轻量）/ paddleocr（精度高但重）
    ocr_backend: str = field(default_factory=lambda: _env("AISAG_DOC_OCR_BACKEND", "rapidocr"))
    # 是否对 PDF/Word 中的图片做 OCR（开启可提取图片中的文字，但入库会变慢）
    ocr_images: bool = field(
        default_factory=lambda: _env("AISAG_DOC_OCR_IMAGES", "true").lower() == "true")
    # PDF 扫描件 markdown 生成模式：direct（默认，跳过 pymupdf4llm 二次版面分析）
    # / pymupdf4llm（用 pymupdf4llm 二次分析，表格列对齐更准但可能误判）
    pdf_markdown_mode: str = field(
        default_factory=lambda: _env("AISAG_PDF_MARKDOWN_MODE", "direct"))
    # 上传/解析时使用的临时目录（NamedTemporaryFile 的 dir 参数）。
    # 留空则用系统默认（Windows: %TEMP%，Linux: /tmp）；指定路径会自动创建。
    # 覆盖：api.py 上传入库的临时文件 + Excel 样式表降级副本。
    upload_tmp_dir: str = field(
        default_factory=lambda: _env("AISAG_UPLOAD_TMP_DIR", str(_PKG_DIR / "tmp")))


@dataclass
class Config:
    mysql: MysqlConfig = field(default_factory=MysqlConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    splitter: SplitterConfig = field(default_factory=SplitterConfig)
    log: LogConfig = field(default_factory=LogConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    doc_parser: PdfDocParserConfig = field(default_factory=PdfDocParserConfig)