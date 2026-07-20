"""LLM 模块：支持 OpenAI 标准及兼容接口的大模型。

多场景 LLM（配合 llm_profiles.yaml）：
    from ai_sag.llm import LlmFactory
    factory = LlmFactory(cfg)
    factory.get("ANSWER").achat(...)              # 答案生成
    factory.get("ENTITY_EXTRACT").astructured_predict(...)  # 实体抽取
"""
from __future__ import annotations

from .factory import LlmFactory
from .openai_like import OpenAILikeLLM

__all__ = ["LlmFactory", "OpenAILikeLLM"]