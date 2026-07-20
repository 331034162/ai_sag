"""OpenAI 兼容接口适配：基于 llama_index 的 OpenAI，绕过模型名校验。

适用场景：DeepSeek、阿里云百炼、vLLM 等 OpenAI 兼容后端，model 名不在 OpenAI 白名单。

llama_index.llms.openai.OpenAI 在 __init__ 和 achat 两个阶段都会校验 model 名，
非官方模型（如 deepseek-chat、qwen3.6-27b）会被拒。

策略：
- self.model 保持 "gpt-4"（骗过 llama_index 所有阶段的校验）
- 真实 model 名通过 additional_kwargs["model"] 注入，OpenAI SDK 会用该值覆盖请求体的 model 字段
- 重写 metadata.model_name 返回真实模型名，让下游 Agent / 日志能看到真实模型

参数分三层与 llamaindex / OpenAI SDK 对齐：
1) temperature / max_tokens / timeout / max_retries → 走 OpenAI.__init__ 标准参数
2) additional_kwargs（如 tool_choice / top_p） → 展开到请求体顶层（SDK 识别）
3) extra_body（如 thinking / enable_thinking） → 打包到 extra_body（SDK 不识别的厂商扩展字段）

注意：思考模式与 structured_predict（tool_choice=required）冲突时，应通过分场景配置解决：
    SAG_LLM_PROFILE_ENTITY_EXTRACT_LLM_NAME=qwen_fast   # 不开思考的模型
    SAG_LLM_PROFILE_RERANK_LLM_NAME=qwen_fast
    SAG_LLM_PROFILE_EVENT_EXTRACT_LLM_NAME=qwen_fast
    SAG_LLM_PROFILE_ANSWER_LLM_NAME=qwen_thinking       # 思考模型用于答案生成
或对结构化场景独立配置 extra_body：
    SAG_LLM_PROFILE_ENTITY_EXTRACT_EXTRA_BODY={"enable_thinking": false}
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from llama_index.core.llms import LLMMetadata
from llama_index.llms.openai import OpenAI


class OpenAILikeLLM(OpenAI):
    """OpenAI 兼容接口适配器：绕过模型名校验，支持任意 OpenAI 兼容后端。

    参数分层说明：
    - temperature / max_tokens：llamaindex OpenAI 标准参数，直接传给 super().__init__
    - additional_kwargs：合并到 super.additional_kwargs，由父类 _get_model_kwargs 展开到请求体顶层
      适合 OpenAI SDK 顶层识别的字段（tool_choice、top_p、seed 等）
    - extra_body：重写 _get_model_kwargs 注入到 all_kwargs["extra_body"]
      适合 SDK 不识别的厂商扩展字段（thinking、enable_thinking、reasoning_effort 等）
      对齐 OpenAI SDK 官方示例：client.chat.completions.create(extra_body={...})

    厂商扩展示例（通过 .env 或 llm_profiles.yaml 配置 extra_body）：
    - 阿里云百炼 qwen3.6-27b：{"enable_thinking": true}
    - DeepSeek deepseek-v4-pro：{"reasoning_effort": "high", "thinking": {"type": "enabled"}}

    思考模式与 structured_predict 冲突的处理见模块 docstring。
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        api_base: Optional[str] = "https://api.deepseek.com",
        additional_kwargs: Optional[dict] = None,
        extra_body: Optional[dict] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> None:
        # 真实 model 名通过 additional_kwargs 注入请求体顶层
        # （OpenAI SDK 在构造请求时会用该字段作为请求体的 model）
        # 注意：不要放到 super().__init__(model=...)，否则会被 llama_index 校验拒掉
        merged_additional = {
            "model": model,  # 真实模型名，注入请求体
            **(additional_kwargs or {}),
        }
        super().__init__(
            model="gpt-4",  # 骗过 llama_index 的 __init__ 和 achat 校验
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=merged_additional,
            **kwargs,
        )
        # 不再覆盖 self.model，保持 "gpt-4" 让所有阶段的校验都通过
        self._real_model = model  # 仅用于 metadata 暴露
        self._extra_body = dict(extra_body or {})

    def _get_model_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        """重写：把厂商扩展字段打包到 extra_body 传给 OpenAI SDK。

        OpenAI SDK 的 chat.completions.create(extra_body=...) 会把 extra_body 字段
        合并到 HTTP 请求体顶层，效果等同于官方示例：
            client.chat.completions.create(
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}},
            )
        统一走 extra_body 简化处理，避免 SDK 对非标顶层字段（如 thinking）报错。

        注意：思考模式与 structured_predict 冲突需通过分场景配置解决，
        不再在代码层做特殊处理（避免硬编码思考字段白名单）。
        """
        all_kwargs = super()._get_model_kwargs(**kwargs)
        if not self._extra_body:
            return all_kwargs
        existing_extra_body = all_kwargs.get("extra_body") or {}
        # 浅拷贝避免污染 self._extra_body；合并时 extra_body 覆盖同名字段
        all_kwargs["extra_body"] = {**existing_extra_body, **self._extra_body}
        return all_kwargs

    @property
    def metadata(self) -> LLMMetadata:
        # metadata.model_name 返回真实模型名，供下游 Agent / 日志识别
        return LLMMetadata(
            context_window=128000,
            num_output=self.max_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=True,
            model_name=self._real_model,
        )