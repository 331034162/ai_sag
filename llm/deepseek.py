"""DeepSeek LLM 适配：继承 OpenAI，绕过模型名校验，适配 DeepSeek 的 deepseek-chat。

DeepSeek 兼容 OpenAI 接口，但 llama_index.llms.openai.OpenAI 会校验 model 名，
deepseek-chat 不在白名单会被拒。参照 deepseek_llm.py 的方案：构造时传 gpt-4 骗过校验，
再把真实 model 名赋给 self.model 并重写 metadata。
"""
from __future__ import annotations

from typing import Optional

from llama_index.core.llms import LLMMetadata
from llama_index.llms.openai import OpenAI


class DeepSeekLLM(OpenAI):
    """Custom OpenAI wrapper for DeepSeek that bypasses model name validation."""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        api_base: Optional[str] = "https://api.deepseek.com",
        additional_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        merged_kwargs = {"temperature": 0.0, **(additional_kwargs or {})}
        super().__init__(
            model="gpt-4",
            api_key=api_key,
            api_base=api_base,
            additional_kwargs=merged_kwargs,
            **kwargs,
        )
        self.model = model

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=128000,
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name=self.model,
        )