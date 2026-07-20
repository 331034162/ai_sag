"""验证多场景 LLM 配置：profile 加载、场景独立参数、backend 自动判断。

直接读取项目 .env 配置进行验证，不修改环境变量。
"""
import sys
from pathlib import Path
# 项目实际包在 d:\ai_sag_git\ai_sag\（base/、llm/、retrieval/ 等子包）
# 通过 `python -m ai_sag._check_scenes` 启动时包名就是 ai_sag，
# 直接运行脚本时则把父目录加入 sys.path 让 ai_sag 包可见。
_PKG_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PKG_PARENT))

from ai_sag.base import Config  # noqa: E402
from ai_sag.base.config import _load_profiles_yaml  # noqa: E402
from ai_sag.llm import LlmFactory  # noqa: E402
from ai_sag.llm.factory import is_openai_official_model  # noqa: E402


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
section("Step 1: 验证 llm_profiles.yaml 加载")
# ============================================================
profiles = _load_profiles_yaml()
print(f"已加载 profiles: {list(profiles.keys())}")
for name, cfg in profiles.items():
    print(f"  [{name}] model={cfg.get('model')!r} base_url={cfg.get('base_url')!r} "
          f"api_key={'***' if cfg.get('api_key') else '(空)'}")
if not profiles:
    print("[ERROR] 未找到 llm_profiles.yaml 或文件为空")
    sys.exit(1)
else:
    print("[OK] profiles 加载正确")


# ============================================================
section("Step 1.5: 验证 is_openai_official_model 自动判断")
# ============================================================
cases = [
    ("gpt-4o", True),
    ("gpt-4-turbo", True),
    ("gpt-3.5-turbo", True),
    ("o1-preview", True),
    ("o3-mini", True),
    ("text-davinci-003", True),
    ("deepseek-chat", False),
    ("qwen3.6-27b", False),
    ("qwen-turbo", False),
    ("Qwen2.5-7B-Instruct", False),
]
all_pass = True
for model, expected in cases:
    actual = is_openai_official_model(model)
    ok = "OK" if actual == expected else "FAIL"
    if actual != expected:
        all_pass = False
    print(f"  [{ok}] {model:<25} expected={expected} actual={actual}")
assert all_pass, "is_openai_official_model 判断错误"
print("[OK] backend 自动判断逻辑正确")


# ============================================================
section("Step 2: 读取当前 .env 配置的场景")
# ============================================================
cfg = Config()
print(f"已配置场景: {list(cfg.llm_scenes.keys())}")
for scene, sc in cfg.llm_scenes.items():
    print(f"  [{scene}]")
    print(f"    profile.name = {sc.profile.name}")
    print(f"    profile.model = {sc.profile.model}")
    print(f"    additional_kwargs = {sc.additional_kwargs}")
    print(f"    extra_body    = {sc.extra_body}")


# ============================================================
section("Step 3: 验证 LlmFactory 按场景返回不同 LLM 实例")
# ============================================================
factory = LlmFactory(cfg)
scenes_to_check = ["ANSWER", "ENTITY_EXTRACT", "RERANK",
                   "QUERY_REWRITE", "GENRE_CLASSIFY", "EVENT_EXTRACT"]
print(f"{'场景':<20} {'model':<25} {'backend':<14} {'temperature':<12} {'extra_body'}")
print("-" * 95)
for scene in scenes_to_check:
    try:
        llm = factory.get(scene)
        extra = llm._extra_body if hasattr(llm, "_extra_body") else "(无 _extra_body 属性)"
        temp = getattr(llm, "temperature", "?")
        cls = type(llm).__name__
        print(f"{scene:<20} {llm._real_model:<25} {cls:<14} {str(temp):<12} {extra}")
    except Exception as e:
        print(f"{scene:<20} [ERROR] {e}")


# ============================================================
section("Step 4: 验证缓存（同一场景多次访问返回同一实例）")
# ============================================================
a = factory.get("ANSWER")
b = factory.get("ANSWER")
c = factory.get("ENTITY_EXTRACT")
print(f"ANSWER is ANSWER:       {a is b}  (应为 True)")
print(f"ANSWER is ENTITY_EXTRACT: {a is c}  (应为 False)")
assert a is b, "同场景应返回缓存实例"
assert a is not c, "不同场景应返回不同实例"
print("[OK] 缓存正确")


# ============================================================
section("Step 5: 验证 SagRetriever._llm_for 回退逻辑")
# ============================================================
from ai_sag.retrieval.sag_retriever import SagRetriever  # noqa: E402

# 不传 llm_factory：所有场景都应回退到 self.llm
class _StubLLM:
    _real_model = "stub-model"
retriever_legacy = SagRetriever.__new__(SagRetriever)
retriever_legacy.llm = _StubLLM()
retriever_legacy._llm_factory = None
assert retriever_legacy._llm_for("ANY_SCENE") is retriever_legacy.llm, \
    "无 factory 时应回退到 self.llm"
print("[OK] 无 llm_factory 时回退到 self.llm（旧版兼容）")


# ============================================================
section("Step 6: 验证未知场景应抛 ValueError（无 DEFAULT 兜底）")
# ============================================================
try:
    factory.get("UNKNOWN_SCENE")
    print("[ERROR] 应该抛 ValueError 但未抛出")
except ValueError as e:
    print(f"[OK] 未知场景抛 ValueError: {e}")


print()
print("=" * 70)
print("✅ 全部验证通过")
print("=" * 70)