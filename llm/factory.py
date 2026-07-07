"""LLM 工厂：默认只支持 OpenAI 标准兼容的大模型，按配置创建对应实现。

后端：
- deepseek  DeepSeek（OpenAI 兼容，绕过模型名校验），推荐用于 DeepSeek API
- openai    OpenAI 原生（gpt-4o 等，需模型名在白名单内）

所有后端返回 llama_index.core.llms.LLM，下游统一用 complete() / structured_predict()。
"""
from __future__ import annotations

from llama_index.core.llms import LLM

from ..base import Config


def create_llm(cfg: Config) -> LLM:
    backend = cfg.llm.backend.lower()
    if backend in ("deepseek", "openai_like", "openailike"):
        from .deepseek import DeepSeekLLM
        return DeepSeekLLM(
            model=cfg.llm.model,
            api_key=cfg.llm.api_key,
            api_base=cfg.llm.base_url,
            timeout=cfg.llm.timeout,
            max_retries=cfg.llm.max_retries,
        )
    if backend == "openai":
        from llama_index.llms.openai import OpenAI
        return OpenAI(
            model=cfg.llm.model,
            api_key=cfg.llm.api_key,
            api_base=cfg.llm.base_url or None,
            timeout=cfg.llm.timeout,
            max_retries=cfg.llm.max_retries,
        )
    raise ValueError(
        f"未知 LLM 后端: {backend}（支持: deepseek / openai）"
    )