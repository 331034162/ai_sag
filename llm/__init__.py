"""LLM 模块：默认支持 OpenAI 标准兼容大模型，提供 DeepSeek 适配。

用法：
    from ai_sag.llm import create_llm
    llm = create_llm(cfg)  # cfg.llm.backend 默认 "deepseek"
    resp = llm.complete("你好")
"""
from __future__ import annotations

from .deepseek import DeepSeekLLM
from .factory import create_llm

__all__ = ["create_llm", "DeepSeekLLM"]