"""日志模块：基于 loguru，支持控制台/文件双输出、异步写入、轮转压缩、trace_id 链路追踪。

参考工业级 loguru 实践，针对 ai_sag 单机本地场景做适配：
- 去掉多 Pod IP 分目录（ai_sag 非容器化集群部署）
- 用 loguru 原生 rotation/retention 语法，更稳健
- 保留 ContextVar trace_id，供 API 请求链路追踪

用法：
    from ai_sag.base.logger import get_logger, set_trace_id, generate_trace_id

    # API 中间件示例
    @app.middleware("http")
    async def trace_middleware(request, call_next):
        trace_id = request.headers.get("X-Trace-Id") or generate_trace_id()
        token = set_trace_id(trace_id)
        try:
            return await call_next(request)
        finally:
            token.reset()

    logger = get_logger()
    logger.info("文档入库完成 source_id={}", source_id)

环境变量（详见 base/config.py 的 LogConfig）：
    AISAG_LOG_LEVEL      INFO
    AISAG_LOG_DIR        ai_sag/logs
    AISAG_LOG_ROTATION   500 MB
    AISAG_LOG_RETENTION  30 days
    AISAG_LOG_COLORIZE   false
"""
from __future__ import annotations

import sys
import time
import uuid
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Optional

from loguru import logger

from .config import Config, LogConfig

# ✅ 上下文变量：线程/协程安全的请求追踪 ID
TRACE_ID_CTX: ContextVar[str] = ContextVar("ai_sag_trace_id", default="-")

# 带颜色的控制台格式
_LOG_FORMAT_COLOR = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<yellow>Trace-ID: {extra[trace_id]}</yellow> | "
    "<level>{message}</level>"
)

# 纯文本格式（文件 + 无颜色终端）
_LOG_FORMAT_PLAIN = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "Trace-ID: {extra[trace_id]} | "
    "{message}"
)

_initialized = False


def _trace_id_filter(record):
    """filter：动态从 ContextVar 注入 trace_id 到日志记录。"""
    record["extra"]["trace_id"] = TRACE_ID_CTX.get()
    return True


def init_logger(cfg: Optional[LogConfig] = None) -> "logger": # type: ignore
    """初始化日志系统。幂等：重复调用会先移除旧 handler 再重新配置。

    Args:
        cfg: 日志配置；为 None 时用 Config().log 读取环境变量默认值。
    """
    global _initialized
    cfg = cfg or Config().log

    # 清空默认 handler，避免重复输出
    logger.remove()
    # 给 logger 注入默认 trace_id extra，避免 filter 前格式化报错
    logger.configure(extra={"trace_id": "-"})

    log_path = Path(cfg.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # ✅ 控制台输出（异步写入，避免阻塞事件循环）
    console_format = _LOG_FORMAT_COLOR if cfg.colorize else _LOG_FORMAT_PLAIN
    logger.add(
        sys.stdout,
        format=console_format,
        level=cfg.level,
        colorize=cfg.colorize,
        enqueue=True,
        filter=_trace_id_filter,
    )

    # ✅ 文件输出（异步写入，纯文本，按大小轮转 + 自动压缩 + 过期清理）
    logger.add(
        log_path / "ai_sag_{time:YYYY-MM-DD}.log",
        format=_LOG_FORMAT_PLAIN,
        level="DEBUG",
        rotation=cfg.rotation,
        retention=cfg.retention,
        compression="gz",
        encoding="utf-8",
        enqueue=True,
        colorize=False,
        delay=True,
        backtrace=True,
        diagnose=False,
        filter=_trace_id_filter,
    )

    _initialized = True
    return logger


def get_logger():
    """获取已初始化的 logger 单例。首次调用时按默认配置自动初始化。"""
    if not _initialized:
        init_logger()
    return logger


def get_trace_id() -> str:
    """获取当前上下文的 trace_id。"""
    return TRACE_ID_CTX.get()


def set_trace_id(trace_id: str) -> Token:
    """设置当前上下文的 trace_id，返回 Token 用于 reset。

    示例：
        token = set_trace_id("abc")
        try:
            do_work()
        finally:
            reset_trace_id(token)
    """
    return TRACE_ID_CTX.set(trace_id)


def reset_trace_id(token: Token) -> None:
    """重置 trace_id 到 set_trace_id 之前的值。

    封装 ContextVar.reset(token)，避免调用方直接操作 TRACE_ID_CTX。
    """
    TRACE_ID_CTX.reset(token)


def generate_trace_id() -> str:
    """生成唯一 trace_id：时间戳 + 短 uuid，可读且全局唯一。"""
    return f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"


__all__ = [
    "init_logger",
    "get_logger",
    "get_trace_id",
    "set_trace_id",
    "reset_trace_id",
    "generate_trace_id",
]