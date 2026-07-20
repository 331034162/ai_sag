"""配置：复用项目 .env，组件配置结构化为嵌套子 dataclass，便于灵活切换与扩展。

环境变量前缀：
- SAG_     复用项目原有配置（MySQL/Embedding/LLM 场景配置）
- AISAG_   ai_sag 模块专用配置（后端选择、切分、向量库路径等）

配置结构：
    cfg.mysql.host / cfg.mysql.port ...
    cfg.embedding.backend / cfg.embedding.model_path ...
    cfg.llm_scenes["ANSWER"].profile / .additional_kwargs / .extra_body
    cfg.vector_store.backend / cfg.vector_store.chroma_path ...
    cfg.splitter.mode / cfg.splitter.chunk_size ...
    cfg.search.similarity_threshold / cfg.search.fusion ...
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

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


def _parse_json_env(key: str, default):
    """解析 JSON 格式的环境变量（如 thinking_kwargs）。

    支持配置厂商特定的非标参数：
      AISAG_LLM_THINKING_KWARGS='{"enable_thinking": true}'
      AISAG_LLM_THINKING_KWARGS='{"reasoning_effort": "high", "thinking": {"type": "enabled"}}'

    空字符串或缺省时返回 default；解析失败给出明确错误，不静默回退。
    """
    import json
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"环境变量 {key} 的值不是合法 JSON：{raw!r}（{e}）"
        ) from e


# ---------- LLM Profile（yaml）加载 ----------
# 大模型连接身份配置从 llm_profiles.yaml 加载；运行参数（temperature / max_tokens /
# extra_body 等）在 .env 中按场景独立配置，实现"连接信息集中、参数按场景独立"。

_PROFILES_CACHE: dict | None = None


def _expand_env_vars(value):
    """递归把 yaml 里 ${VAR} 占位符替换为 os.environ[VAR]。

    api_key 等敏感信息支持 ${VAR} 形式从环境变量读取（也可直接写明文）。
    缺失环境变量时返回空字符串并给出警告（不抛错，兼容本地开发）。
    """
    if isinstance(value, str):
        return _ENV_VAR_RE.sub(_env_var_replacer, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


_ENV_VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _env_var_replacer(match: re.Match) -> str:
    var = match.group(1)
    val = os.environ.get(var)
    if val is None:
        print(f"[WARNING] 环境变量 {var} 未设置（在 llm_profiles.yaml 中被引用）")
        return ""
    return val


def _load_profiles_yaml() -> dict:
    """加载 llm_profiles.yaml（带缓存）。

    查找顺序：
      1. SAG_LLM_PROFILES_PATH 环境变量指定的路径
      2. _PKG_DIR / "llm_profiles.yaml"
      3. _PKG_DIR / "ai_sag" / "llm_profiles.yaml"（子包模式）

    返回 {profile_name: {base_url, api_key, model, timeout, max_retries}} 字典。
    """
    global _PROFILES_CACHE
    if _PROFILES_CACHE is not None:
        return _PROFILES_CACHE

    import yaml  # 延迟导入，未安装时也能加载其余配置
    candidates = []
    env_path = os.environ.get("SAG_LLM_PROFILES_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(_PKG_DIR / "llm_profiles.yaml")
    candidates.append(_THIS_FILE.parent.parent / "llm_profiles.yaml")

    yaml_path = next((p for p in candidates if p.exists()), None)
    if yaml_path is None:
        _PROFILES_CACHE = {}
        return _PROFILES_CACHE

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    # 展开 ${VAR} 占位符
    _PROFILES_CACHE = {name: _expand_env_vars(cfg) for name, cfg in raw.items()}
    return _PROFILES_CACHE


# 场景列表（与 llm/factory.py 中 SCENES 保持一致）
# 命名规则：所有场景配置统一前缀 SAG_LLM_PROFILE_<SCENE>_*
#   SAG_LLM_PROFILE_<SCENE>_LLM_NAME          选择 llm_profiles.yaml 中的 profile 名
#   SAG_LLM_PROFILE_<SCENE>_ADDITIONAL_KWARGS 场景独立 additional_kwargs（OpenAI SDK 顶层字段，
#                                            如 temperature / max_tokens / top_p / tool_choice）
#   SAG_LLM_PROFILE_<SCENE>_EXTRA_BODY       场景独立 extra_body（厂商扩展字段，
#                                            如 enable_thinking / reasoning_effort / thinking）
# 温度和 max_tokens 不再单独配置，统一走 ADDITIONAL_KWARGS（SDK 顶层识别）。
LLM_SCENES = (
    "GENRE_CLASSIFY",      # 文档体裁分类（离线）
    "EVENT_EXTRACT",       # 事件抽取（离线，structured_predict）
    "QUERY_REWRITE",       # 查询重写（在线）
    "ENTITY_EXTRACT",      # 查询实体抽取（在线，structured_predict）
    "RERANK",              # LLM 精排（在线，structured_predict）
    "ANSWER",              # 答案生成（在线，普通 chat）
)


# ---------- 关键变量缺失检查 ----------
_MISSING = []
for _k, _desc in [("SAG_MYSQL_USER", "MySQL 用户名"), ("SAG_MYSQL_PASSWORD", "MySQL 密码")]:
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
    """Embedding 配置。

    后端差异：
    - bge:   基于 llamaindex HuggingFaceEmbedding，CLS pooling（默认），
             无指令前缀（bge-small-zh-v1.5 上限 512）
    - qwen3: 基于 transformers 手写实现，last_token pooling（官方推荐），
             查询加 "Instruct: ...\\nQuery: " 前缀
             （Qwen3-Embedding-0.6B 支持到 8192）
    """
    # 后端：bge（默认，轻量稳定）| qwen3（精度高，支持长文本）
    backend: str = field(default_factory=lambda: _env("SAG_EMBEDDING_BACKEND", "bge").lower())
    # 本地模型路径（两种后端共用此变量；切后端时改路径即可）
    model_path: str = field(default_factory=lambda: _env("SAG_EMBEDDING_MODEL_PATH", ""))
    # 推理设备（默认 cpu；GPU 环境改 cuda。无 CUDA 时会自动降级到 cpu）
    device: str = field(default_factory=lambda: _env("SAG_EMBEDDING_DEVICE", "cpu"))
    # tokenizer 最大长度（bge 后端会自动压缩到模型上限 512；qwen3 默认 8192）
    max_length: int = field(default_factory=lambda: int(_env("SAG_EMBEDDING_MAX_LENGTH", "8192")))
    # 是否对向量做 L2 归一化（默认 true，适配余弦相似度）
    normalize: bool = field(default_factory=lambda: _env("SAG_EMBEDDING_NORMALIZE", "true").lower() == "true")
    # 批处理大小（默认 32，按批送入模型；0=不切批一次性送入，可能 OOM）
    batch_size: int = field(default_factory=lambda: int(_env("SAG_EMBEDDING_BATCH_SIZE", "32")))


# ---------- 多场景 LLM 配置（基于 llm_profiles.yaml）----------
#
# 在 llm_profiles.yaml 中定义多个 profile（仅 base_url / api_key / model / timeout /
# max_retries），在 .env 中按 6 个场景（GENRE_CLASSIFY / EVENT_EXTRACT / QUERY_REWRITE /
# ENTITY_EXTRACT / RERANK / ANSWER）显式选择 profile 并独立配置运行参数。
#
# 命名规则（所有场景配置统一前缀 SAG_LLM_PROFILE_<SCENE>_*）：
#   SAG_LLM_PROFILE_<SCENE>_LLM_NAME          选择 yaml 中的 profile 名（必填）
#   SAG_LLM_PROFILE_<SCENE>_ADDITIONAL_KWARGS 场景独立 additional_kwargs
#                                              （SDK 顶层字段：temperature / max_tokens / top_p / tool_choice）
#   SAG_LLM_PROFILE_<SCENE>_EXTRA_BODY       场景独立 extra_body
#                                              （厂商扩展：enable_thinking / reasoning_effort / thinking）
#
# 后端由 factory 按 profile.model 自动判断（OpenAI 官方模型走 openai，其他走 openai_like），
# 无需也无法在 .env 中显式配置。
# 每个场景必须显式配置 _LLM_NAME，不存在 DEFAULT 回退机制。
# 配置示例见 .env.example


@dataclass
class LlmProfile:
    """模型连接身份：对应 llm_profiles.yaml 中的一个条目。"""
    name: str
    base_url: str
    api_key: str
    model: str
    timeout: float = 120.0
    max_retries: int = 3

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "LlmProfile":
        return cls(
            name=name,
            base_url=str(data.get("base_url", "")),
            api_key=str(data.get("api_key", "")),
            model=str(data.get("model", "")),
            timeout=float(data.get("timeout", 120)),
            max_retries=int(data.get("max_retries", 3)),
        )


@dataclass
class LlmSceneConfig:
    """场景级 LLM 配置：profile（连接身份） + 运行参数。

    运行参数通过 additional_kwargs / extra_body 两个字典承载：
    - additional_kwargs: OpenAI SDK 顶层字段（temperature / max_tokens / top_p / tool_choice 等）
    - extra_body: 厂商扩展字段（enable_thinking / reasoning_effort / thinking 等）

    后端由 factory 按 profile.model 自动判断：
    - model 在 llamaindex 的 OpenAI 官方模型列表内 → openai 后端
    - 否则 → openai_like 后端（兼容 DeepSeek、阿里云百炼、vLLM 等）
    additional_kwargs / extra_body 为 None 时 factory 会用空字典兜底。
    """
    scene: str
    profile: LlmProfile
    additional_kwargs: Optional[Dict[str, Any]] = None  # None=未配置（factory 用空字典）
    extra_body: Optional[Dict[str, Any]] = None          # None=未配置（factory 用空字典）


def _resolve_profile(scene: str, profiles: dict) -> tuple[str, LlmProfile]:
    """解析场景使用的 profile。

    环境变量：SAG_LLM_PROFILE_<SCENE>_LLM_NAME（必填）
    未配置或配置错误时告警并取第一个 profile 兜底（避免直接报错影响存量环境）。
    """
    name = os.environ.get(f"SAG_LLM_PROFILE_{scene}_LLM_NAME", "").strip()
    if name and name in profiles:
        return name, LlmProfile.from_dict(name, profiles[name])
    # 未配置或配置错误：告警并取第一个 profile 兜底
    if profiles:
        first_name = next(iter(profiles))
        if name:
            print(
                f"[WARNING] 场景 {scene} 配置的 profile '{name}' 在 llm_profiles.yaml 中未找到，"
                f"回退到第一个 profile '{first_name}'"
            )
        else:
            print(
                f"[WARNING] 场景 {scene} 未配置 SAG_LLM_PROFILE_{scene}_LLM_NAME，"
                f"回退到第一个 profile '{first_name}'（建议在 .env 中显式配置）"
            )
        return first_name, LlmProfile.from_dict(first_name, profiles[first_name])
    # 完全没有 yaml：明确报错
    raise RuntimeError(
        f"未找到 llm_profiles.yaml，场景 {scene} 无法解析 profile。"
        f"请创建 llm_profiles.yaml（参考 llm_profiles.yaml.example）"
    )


def _load_llm_scenes() -> Dict[str, LlmSceneConfig]:
    """加载所有场景的 LLM 配置。

    命名规则（统一前缀 SAG_LLM_PROFILE_<SCENE>_*）：
      SAG_LLM_PROFILE_<SCENE>_LLM_NAME          选 profile（必填）
      SAG_LLM_PROFILE_<SCENE>_ADDITIONAL_KWARGS additional_kwargs（temperature/max_tokens 等）
      SAG_LLM_PROFILE_<SCENE>_EXTRA_BODY       extra_body（厂商扩展字段）
    后端由 factory 按 profile.model 自动判断，无需也无法在 .env 中配置。
    每个场景必须显式配置 _LLM_NAME，不存在 DEFAULT 回退。
    """
    profiles = _load_profiles_yaml()

    scenes: Dict[str, LlmSceneConfig] = {}
    for scene in LLM_SCENES:
        # 1. 解析 profile（必填，缺失时 _resolve_profile 兜底或报错）
        profile_name, profile = _resolve_profile(scene, profiles)
        # 2. 解析场景运行参数（场景未配 → None，factory 用空字典兜底）
        additional = _parse_json_env(f"SAG_LLM_PROFILE_{scene}_ADDITIONAL_KWARGS", None)
        extra = _parse_json_env(f"SAG_LLM_PROFILE_{scene}_EXTRA_BODY", None)
        scenes[scene] = LlmSceneConfig(
            scene=scene,
            profile=profile,
            additional_kwargs=dict(additional) if additional is not None else None,
            extra_body=dict(extra) if extra is not None else None,
        )
    return scenes


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
    # 表格专用 chunk_size：0 表示复用 chunk_size。
    # 表格每行以"列名: 值"呈现，单行字符数较多，可独立调参平衡实体抽取精度与 LLM 调用成本。
    table_chunk_size: int = field(default_factory=lambda: int(_env("AISAG_TABLE_CHUNK_SIZE", "0")))
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
class CleanerConfig:
    """文本清洗配置：控制 TextCleaner 各步骤开关，适配特殊文档场景。

    默认值与 TextCleaner 原默认行为一致，无需改动即可正常工作。
    特殊场景（如代码片段多的技术文档）可通过环境变量关闭某项清洗。
    """
    # 去 HTML/XML 标签（只剥离合法标签，不误伤 a<b、List<String> 等文本）
    strip_html: bool = field(default_factory=lambda: _env("AISAG_CLEANER_STRIP_HTML", "false").lower() == "true")
    # 合并硬断行（非段落边界的单换行）；md/xlsx/xls/csv 自动跳过
    merge_hard_breaks: bool = field(default_factory=lambda: _env("AISAG_CLEANER_MERGE_HARD_BREAKS", "true").lower() == "true")
    # 压缩多余空行（连续 3+ 换行压为 2 个）
    collapse_blank_lines: bool = field(default_factory=lambda: _env("AISAG_CLEANER_COLLAPSE_BLANK_LINES", "true").lower() == "true")
    # 规整空白（全角空格、连续空格、行首尾空白）
    normalize_whitespace: bool = field(default_factory=lambda: _env("AISAG_CLEANER_NORMALIZE_WHITESPACE", "true").lower() == "true")
    # 列表项保护：合并硬断行时识别 -/*/1. 等列表标记，不把列表项压成一行
    # 关闭后退化为旧行为（所有单换行都合并），不推荐
    protect_list_items: bool = field(default_factory=lambda: _env("AISAG_CLEANER_PROTECT_LIST_ITEMS", "true").lower() == "true")


@dataclass
class SearchConfig:
    # 事件召回/粗排/基线 chunk 召回的相似度阈值（论文 3.3 Path B：0.4）
    similarity_threshold: float = field(
        default_factory=lambda: float(_env("AISAG_SIMILARITY_THRESHOLD", "0.4")))
    # 实体向量扩展阈值（Path A，独立于事件召回阈值）
    entity_expand_threshold: float = field(
        default_factory=lambda: float(_env("AISAG_ENTITY_EXPAND_THRESHOLD", "0.5")))
    # 实体向量扩展 top_k（每个查询实体名召回的近邻数）
    # 默认 10：配合 seed_entity_min_batch=15，单实体 10 个近邻 + 精确匹配即可触发离群检测；
    # 多实体场景去重后更容易达到 min_batch，让度数过滤统计量更稳
    entity_expand_topk: int = field(
        default_factory=lambda: int(_env("AISAG_ENTITY_EXPAND_TOPK", "10")))
    # 实体向量扩展开关（默认开启）：
    #   true  - 用 LLM 抽取的实体名 embedding 去向量库找近邻，补充同义词/相关实体
    #   false - 仅用精确匹配（SQL 按名字查），不从向量库扩展实体
    # 开启动机：弥补实体名同义词/别名差异（如"大模型"vs"AI大模型应用"），提高召回率；
    # 若数据噪声多或泛化实体污染严重，可设为 false 退化为精确匹配，更保守。
    entity_expand_enabled: bool = field(
        default_factory=lambda: _env("AISAG_ENTITY_EXPAND_ENABLED", "true").lower() == "true")
    max_hops: int = field(default_factory=lambda: int(_env("AISAG_MAX_HOPS", "2")))
    # 多跳扩展子策略：multi（固定跳数）/ hopllm（每跳相似度动态停止）
    sub_strategy: str = field(default_factory=lambda: _env("AISAG_SUB_STRATEGY", "hopllm").lower())
    # hopllm 动态停止：新跳事件 summary 相似度低于此值则终止扩展
    # （改用 summary 打分后分数普遍高于 content，0.15 形同虚设，上调至 0.3）
    hop_relevance_threshold: float = field(
        default_factory=lambda: float(_env("AISAG_HOP_RELEVANCE_THRESHOLD", "0.3")))
    # BFS 扩展事件软过滤阈值（默认 0 禁用）：每跳扩展事件的 summary 分数低于此值则不 track
    # 作用：减少粗排/精排候选规模，剔除 BFS 扩展噪声
    # 与 hop_relevance_threshold 的区别：
    #   - soft_threshold 控制单事件是否进候选池（< 阈值的不 track）
    #   - hop_relevance_threshold 控制是否继续扩展（best_score < 阈值则停止 BFS）
    # 关系：soft_threshold <= hop_relevance_threshold（软过滤更宽松，停止信号更严格）
    # 被剔除的事件会加入 tracked_events 避免重新召回，但无深度记录不进候选池
    hop_event_soft_threshold: float = field(
        default_factory=lambda: float(_env("AISAG_HOP_EVENT_SOFT_THRESHOLD", "0.0")))
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
    # 边界实体度数硬过滤方法选择（可配置切换）：
    #   "percentile" - 分位数法（默认）：剔除度数 > batch 内 P{percentile} 的实体，自适应 batch 分布
    #   "mad"        - 绝对中位差法：剔除度数 > median + k*MAD/0.6745 的实体，对长尾分布鲁棒
    #   "tukey"      - Tukey 篱笆法（箱线图）：剔除度数 > Q3 + k*IQR 的实体，经典统计方法
    #   "otsu"       - Otsu 大津法：数据驱动自动找最优二分阈值，无需手动设分位数
    #   "none"       - 关闭度数硬过滤，仅用绝对上限 + 综合评分（最早的行为）
    entity_degree_method: str = field(
        default_factory=lambda: _env("AISAG_ENTITY_DEGREE_METHOD", "otsu").lower())
    # ---- 以下 4 个参数控制 BFS 边界实体的度数离群检测 ----
    # 背景：BFS 每跳收集的边界实体中，有些实体关联了大量事件（高「度数」），
    # 这类「枢纽实体」（如"2024年""已发布"等常见词）会桥接大量无关事件，污染检索。
    # 过滤分两层：① 绝对上限兜底 → ② 统计离群检测（方法由 entity_degree_method 选择）。
    #
    # 绝对度数硬上限：度数 > 此值的实体直接剔除，所有方法共用兜底。
    # 例如设为 50，则关联超过 50 个事件的实体无条件丢弃。设为 0 关闭。
    entity_degree_abs_max: int = field(
        default_factory=lambda: int(_env("AISAG_ENTITY_DEGREE_ABS_MAX", "50")))
    # 分位数法分位点（仅 entity_degree_method="percentile" 时生效）。
    # 计算当前 batch 内实体度数的 P{percentile} 分位数，度数超过该值的实体剔除。
    # 例如 P95 默认剔除 batch 内度数最高的 5% 枢纽实体。
    # 取值范围 0~100，设为 100 则不过滤（P100 = 最大值）。
    entity_degree_percentile: float = field(
        default_factory=lambda: float(_env("AISAG_ENTITY_DEGREE_PERCENTILE", "95")))
    # 离群倍数 k（entity_degree_method="mad" 或 "tukey" 时生效）。
    # k 越大 → 阈值越高 → 过滤越少（越保守）；k 越小 → 过滤越多（越激进）。
    #
    # MAD  法：阈值 = median + k × MAD / 0.6745    （MAD/0.6745 ≈ σ 鲁棒估计）
    # Tukey法：阈值 = Q3 + k × IQR                  （经典箱线图 k=1.5，默认 3.0 更保守）
    #
    # MAD 法 k 与置信度对照（正态假设，单侧 Φ(k)）：
    #   k=1.0 → 84.1%      k=2.0 → 97.7%      k=3.0 → 99.73%
    #   k=4.0 → 99.997%    k=5.0 → 99.99997%  （k≥5 近似关闭 MAD 过滤）
    #
    # 注意：此参数对 percentile / otsu / none 方法无影响。
    entity_degree_outlier_k: float = field(
        default_factory=lambda: float(_env("AISAG_ENTITY_DEGREE_OUTLIER_K", "3.0")))
    # 度数离群检测的最小 batch 大小。
    # 当边界实体数 < 此值时不执行统计检测（小样本统计量不稳定，容易误杀），
    # 此时仅靠绝对上限兜底。设为 1 则永不跳过。BFS 边界过滤和种子实体过滤共用此值。
    entity_degree_min_batch: int = field(
        default_factory=lambda: int(_env("AISAG_ENTITY_DEGREE_MIN_BATCH", "10")))
    # 种子实体候选总量防御性上限：精确匹配+向量扩展合并去重后的硬上限
    # 实际候选数 = LLM抽实体数 × (1精确 + 10向量扩展) ≈ 30-50，几乎不触发，纯防御目的
    seed_entity_budget: int = field(
        default_factory=lambda: int(_env("AISAG_SEED_ENTITY_BUDGET", "200")))
    # 粗排相似度阈值（设为 0 则关闭，对齐旧版仅排序截断）
    coarse_threshold: float = field(
        default_factory=lambda: float(_env("AISAG_COARSE_THRESHOLD", "0")))
    max_events: int = field(default_factory=lambda: int(_env("AISAG_MAX_EVENTS", "100")))
    rerank_top_k: int = field(default_factory=lambda: int(_env("AISAG_RERANK_TOP_K", "5")))
    # 送入 LLM 重排的最大候选数，对齐论文图示 Top-100 设计（P1-4 修复）
    rerank_candidate_limit: int = field(
        default_factory=lambda: int(_env("AISAG_RERANK_CANDIDATE_LIMIT", "100")))
    # 精排候选格式中实体关联权重的强弱信号阈值（基于 aisag_event_entities.weight）
    # weight >= strong 为强信号（核心关联），weak <= weight < strong 为中信号，
    # weight < weak 为弱信号（背景引用）。weight 用于实体角色重要性判别，非主判据。
    rerank_weight_strong: float = field(
        default_factory=lambda: float(_env("AISAG_RERANK_WEIGHT_STRONG", "0.7")))
    rerank_weight_weak: float = field(
        default_factory=lambda: float(_env("AISAG_RERANK_WEIGHT_WEAK", "0.4")))
    # 事件召回轮次衰减基数：精排调整后相关度 = 向量相关度 × decay^event_depth
    # 事件深度语义：种子事件(实体召回+向量召回) depth=0；BFS hop=0 新事件 depth=1；hop=1 depth=2 ...
    # decay=0.6 → 种子1.0 / BFS第1轮0.6 / BFS第2轮0.36，第N轮召回可信度递减。
    # 直接作用在事件分数上，避免实体维度打分的方向错位与向量扩展污染。
    rerank_event_decay: float = field(
        default_factory=lambda: float(_env("AISAG_RERANK_EVENT_DECAY", "0.6")))
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
    # 是否清理 Excel 入库 documents 表的 V6 角色前缀（#TITLE#/#FORM#/#SIGNING#/#GROUP_HEADER#/#DATA#）。
    # 开启：写 documents 表前剥离行首前缀，保证 document 与 chunk 口径一致（推荐）。
    # 关闭：documents 表保留原始 V6 CSV（含前缀），用于调试或特殊场景。
    # 仅对 xlsx/xls 且 content 确实含前缀的文档生效，其他文件类型不受影响。
    strip_excel_role_prefix: bool = field(
        default_factory=lambda: _env("AISAG_STRIP_EXCEL_ROLE_PREFIX", "true").lower() == "true")


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
    # 多场景 LLM 配置：按 6 个场景（GENRE_CLASSIFY / EVENT_EXTRACT /
    # QUERY_REWRITE / ENTITY_EXTRACT / RERANK / ANSWER）独立配置 profile 和运行参数。
    # 每个场景必须显式配置 _LLM_NAME，不存在 DEFAULT 回退机制。
    # 后端由 factory 按 profile.model 自动判断，无需也无法手动配置。
    llm_scenes: Dict[str, "LlmSceneConfig"] = field(default_factory=_load_llm_scenes)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    splitter: SplitterConfig = field(default_factory=SplitterConfig)
    cleaner: CleanerConfig = field(default_factory=CleanerConfig)
    log: LogConfig = field(default_factory=LogConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    doc_parser: PdfDocParserConfig = field(default_factory=PdfDocParserConfig)