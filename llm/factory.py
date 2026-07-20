"""LLM 工厂：按场景创建对应 LLM 实现。

多场景 LLM（配合 llm_profiles.yaml）：
    from llm import LlmFactory
    factory = LlmFactory(cfg)
    factory.get("ANSWER").achat(...)            # 答案生成
    factory.get("ENTITY_EXTRACT").astructured_predict(...)  # 实体抽取
    factory.get("GENRE_CLASSIFY").achat(...)    # 文档体裁分类

后端选择（按 profile.model 自动判断，无需也无法手动配置）：
- model 在 llamaindex 的 OpenAI 官方模型列表（ALL_AVAILABLE_MODELS）中
  → 走 openai 后端（gpt-4o / o1 / o3 等）
- 否则 → 走 openai_like 后端（兼容 DeepSeek、阿里云百炼、vLLM 等）

所有后端返回 llama_index.core.llms.LLM，下游统一用 complete() / structured_predict()。
"""
from __future__ import annotations

from llama_index.core.llms import LLM

from ..base import Config


# 场景列表（与 base/config.py 中 LLM_SCENES 保持一致）
SCENES = (
    "GENRE_CLASSIFY",      # 文档体裁分类（离线，普通 chat）
    "EVENT_EXTRACT",       # 事件抽取（离线，structured_predict）
    "QUERY_REWRITE",       # 查询重写（在线，普通 chat）
    "ENTITY_EXTRACT",      # 查询实体抽取（在线，structured_predict）
    "RERANK",              # LLM 精排（在线，structured_predict）
    "ANSWER",              # 答案生成（在线，普通 chat，建议开启思考）
)


def is_openai_official_model(model: str) -> bool:
    """判断 model 是否为 OpenAI 官方支持的模型（基于 llamaindex 的 ALL_AVAILABLE_MODELS）。

    llamaindex 的 OpenAI 类对模型名有白名单校验（用于推断 context_window /
    is_chat_model / is_function_calling_model 等元数据）。
    不在白名单内的模型（如 deepseek-chat、qwen3.6-27b 等）必须走 OpenAILike。
    """
    from llama_index.llms.openai.utils import ALL_AVAILABLE_MODELS
    # 处理微调模型名格式：ft:gpt-4o:xxx:yyy 或 ft:gpt-3.5-turbo:abc
    name = model
    if name.startswith("ft:"):
        name = name.split(":")[1]
    elif name.startswith("ft-"):
        name = name.split(":")[0].removeprefix("ft-")
    return name in ALL_AVAILABLE_MODELS


def _build_llm_from_scene(scene_cfg) -> LLM:
    """根据场景配置构建 LLM 实例。

    后端按 profile.model 自动判断：
    - OpenAI 官方模型（gpt-4o / o1 / o3 等）→ openai 后端
    - 其他模型（deepseek-chat / qwen3.6-27b / vLLM 等）→ openai_like 后端

    运行参数（additional_kwargs / extra_body）规则：
    - 场景独立配置覆盖一切；场景未独立配置（None）时用空字典兜底
    - additional_kwargs 中的 temperature/max_tokens 会被提取为 OpenAI 标准参数
      （避免与 self.temperature 重复处理），其余字段走 additional_kwargs
    """
    model = scene_cfg.profile.model
    use_openai_native = is_openai_official_model(model)

    # 场景运行参数（None → 空字典兜底）
    additional = dict(scene_cfg.additional_kwargs or {})
    extra = scene_cfg.extra_body or {}

    # 从 additional_kwargs 提取 temperature/max_tokens 作为 OpenAI 标准参数
    # （OpenAI 类内置处理这两个字段，避免被合并到请求体顶层时与 self.temperature 冲突）
    temperature = additional.pop("temperature", 0.0)
    max_tokens = additional.pop("max_tokens", None)

    if use_openai_native:
        from llama_index.llms.openai import OpenAI
        return OpenAI(
            model=model,
            api_key=scene_cfg.profile.api_key,
            api_base=scene_cfg.profile.base_url or None,
            timeout=scene_cfg.profile.timeout,
            max_retries=scene_cfg.profile.max_retries,
            temperature=temperature,
            max_tokens=max_tokens,
            additional_kwargs=additional or None,
        )

    from .openai_like import OpenAILikeLLM
    return OpenAILikeLLM(
        model=model,
        api_key=scene_cfg.profile.api_key,
        api_base=scene_cfg.profile.base_url,
        timeout=scene_cfg.profile.timeout,
        max_retries=scene_cfg.profile.max_retries,
        temperature=temperature,
        max_tokens=max_tokens,
        additional_kwargs=additional,
        extra_body=extra,
    )


class LlmFactory:
    """按场景创建并缓存 LLM 实例。

    用法：
        factory = LlmFactory(cfg)
        llm_answer = factory.get("ANSWER")          # 答案生成
        llm_extract = factory.get("ENTITY_EXTRACT")  # 实体抽取
        llm_genre = factory.get("GENRE_CLASSIFY")   # 文档体裁分类

    每个场景首次访问时创建 LLM 实例并缓存，后续直接返回缓存。
    未知场景会抛 ValueError，不再有 DEFAULT 兜底——请在调用方明确指定场景。
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._cache: dict[str, LLM] = {}

    def get(self, scene: str) -> LLM:
        if scene not in self._cache:
            scene_cfg = self._cfg.llm_scenes.get(scene)
            if scene_cfg is None:
                raise ValueError(
                    f"未知 LLM 场景: {scene}（支持: {', '.join(SCENES)}）"
                )
            self._cache[scene] = _build_llm_from_scene(scene_cfg)
        return self._cache[scene]