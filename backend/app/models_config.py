"""Per-agent model configuration.

Why per-agent: different agents have genuinely different requirements. A vision
task needs a multimodal model, while text planning can use a smaller model.

Three design rules:

  1. **Keys never live in code.** They are stored in `data/providers.json` with
     owner-only permissions and never returned to the browser in full.
  2. **Empty means inherit.** A profile left unset falls back to the global
     default, so a user can configure one key and get a working system.
  3. **Profiles are additive.** Adding a provider does not disturb existing
     assignments; deleting a provider clears the agents pointing at it.

The file is small and rewritten atomically, because a half-written settings file
would leave the app unable to start.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import BASE_DIR, PROVIDERS_PATH

#: Agent ids that can be bound to a model — the agents that actually run.
#:
#: This must match `app.doc.agents_six.register_six()`.
AGENT_IDS = [
    "director",
    "script_adaptor",
    "storyboard",
    "consistency",
    "video_gen",
    "quality_review",
]

#: Labels for error messages and the settings table, so a misconfiguration names
#: the agent a user knows.
AGENT_IDS_LABEL = {
    "director": "导演",
    "script_adaptor": "内容改编",
    "storyboard": "剧本分镜",
    "consistency": "一致性",
    "video_gen": "视频生成",
    "quality_review": "质量审查",
}

BINDABLE_AGENT_IDS = AGENT_IDS

#: Agents that read images, so the settings page can flag a missing vision model.
#: `quality_review` compares a rendered shot against the reference portrait; the
#: director inspects the source material before deciding how to split it.
VISION_AGENTS = ["director", "quality_review"]


def profile_usable(profile: ProviderProfile) -> bool:
    """Whether a profile has enough to make a call.

    A key is required unless the endpoint is local (Ollama and similar gateways
    legitimately ignore it). `base_url` and `model` are always required — a
    missing model produces a confusing 404 from the provider, not a helpful one.
    """
    if not profile.base_url.strip() or not profile.model.strip():
        return False
    if profile.api_key.strip():
        return True
    return "127.0.0.1" in profile.base_url or "localhost" in profile.base_url


def text_capable(profile: ProviderProfile | None) -> bool:
    """Whether this endpoint can serve a chat completion at all.

    A generation gateway (`kind="gateway"`, e.g. a ComfyUI job queue) is not an
    OpenAI-compatible endpoint: it has no `/chat/completions`, and a text agent
    pinned to it dies with "无法连接 LLM 服务（http://…:8190）" — which reads like a
    network fault even when the box is up and the real problem is that the agent
    was pointed at a drawing service.

    This is why the check is on `kind` rather than on "is it local": a gateway on
    localhost is still not a text model, and a hosted endpoint that happens to use
    a port is still fine.
    """
    if profile is None:
        return False
    if (profile.kind or "").strip().lower() == "gateway":
        return False
    return bool(profile.base_url.strip())

#: Presets shown in the UI. Values are defaults only — every field is editable,
#: because providers rename models and users may use a proxy or an internal gateway.
PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "deepseek": {
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "vision_model": "",
        "note": "中文强、便宜、适合文案与结构化输出；无视觉能力",
    },
    "kimi": {
        "label": "Kimi（月之暗面）",
        "base_url": "https://api.moonshot.cn/v1",
        "model": "kimi-k2-turbo-preview",
        "vision_model": "",
        "note": "长上下文，适合长文案与分镜；部分型号支持视觉",
    },
    "qwen": {
        "label": "通义千问（阿里百炼）",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
        "vision_model": "qwen-vl-max",
        "note": "视觉能力强，适合商品图分析",
    },
    "zhipu": {
        "label": "智谱 GLM",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-plus",
        "vision_model": "glm-4v-plus",
        "note": "国内可用，视觉型号可选",
    },
    "doubao": {
        "label": "豆包（火山方舟）",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-pro-32k",
        "vision_model": "doubao-vision-pro-32k",
        "note": "速度快，视觉与文本都有",
    },
    "openai": {
        "label": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "vision_model": "gpt-4o",
        "note": "通用能力强，价格较高",
    },
    "ollama": {
        "label": "本地 Ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen2.5:7b-instruct",
        "vision_model": "",
        "note": "完全离线，免 Key；7B 以上模型才写得动结构化脚本",
    },
    "custom": {
        "label": "自定义（任意 OpenAI 兼容端点）",
        "base_url": "",
        "model": "",
        "vision_model": "",
        "note": "填自己的地址与模型名；代理、自建、内网网关都走这里",
    },
}


@dataclass
class ProviderProfile:
    """One configured provider (an endpoint + key + model)."""

    id: str
    preset: str = "custom"
    label: str = ""
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    vision_model: str = ""
    #: A text-to-image model on this provider (e.g. cogview-4-250304). Empty means
    #: this provider cannot serve the image stage.
    image_model: str = ""
    #: An image-to-video model on this provider (e.g. cogvideox-3). Empty means
    #: this provider cannot serve the video stage.
    video_model: str = ""
    #: What kind of endpoint this is: "" (an OpenAI-compatible provider) or
    #: "gateway" (a local ComfyUI gateway, which is a job queue rather than a
    #: model API and has a different shape).
    #:
    #: Explicit rather than inferred from the URL: guessing "has a port in it, so
    #: it must be local" would misclassify a hosted vendor that happens to use a
    #: unusual address, and the two need different clients.
    kind: str = ""
    timeout: float = 60.0
    temperature: float = 0.4

    def public(self) -> dict[str, Any]:
        """Safe for the browser: the key is masked, never sent in full."""
        return {
            "id": self.id,
            "preset": self.preset,
            "label": self.label,
            "base_url": self.base_url,
            "model": self.model,
            "vision_model": self.vision_model,
            "image_model": self.image_model,
            "video_model": self.video_model,
            "kind": self.kind,
            "timeout": self.timeout,
            "temperature": self.temperature,
            # `has_key` used to mean "the field is not empty", which made a
            # placeholder like `t-key` render as a green "已配置" chip while every
            # run failed with "尚未配置 LLM". Two separate facts are reported now:
            # whether a key is present, and whether it even looks like a key.
            "has_key": bool(self.api_key),
            "key_verdict": key_verdict(self.api_key),
            "key_hint": _mask_key(self.api_key),
        }


#: Verdicts for a stored key. `missing` and `placeholder` both mean "no usable
#: key", and are kept apart because the fix differs: one needs a key pasted in, the
#: other needs the leftover scratch value replaced.
KEY_OK = "ok"
KEY_MISSING = "missing"
KEY_PLACEHOLDER = "placeholder"
KEY_SUSPICIOUS = "suspicious"


def key_verdict(key: str) -> str:
    """Whether a stored key looks usable, without calling the provider.

    This is a shape check, not authentication: it cannot prove a key works (only a
    request can), but it can prove one is a placeholder. That distinction matters
    because `bool(api_key)` reported `t-key` as configured, so the UI showed a green
    chip while every run failed with "尚未配置 LLM" — the user had no way to tell
    that the value in the box was scratch data left behind by a test.

    Thresholds are deliberately loose. Real keys vary wildly (Zhipu 32 hex chars,
    OpenAI `sk-…` with 48+, SiliconFlow `sk-…`, Relay keys of 20-24), so anything
    long and non-trivial is given the benefit of the doubt, and only obviously
    artificial values are rejected. A false "suspicious" would be worse than a
    false "ok": the second is caught by the test button, the first sends the user
    hunting for a problem that is not there.
    """
    raw = (key or "").strip()
    if not raw:
        return KEY_MISSING

    lowered = raw.lower()

    # Values that are self-evidently scratch data.
    if lowered in {"t-key", "m-key", "key", "test", "dummy", "placeholder", "none", "null"}:
        return KEY_PLACEHOLDER
    if lowered.startswith(("your-key", "your_key", "xxx", "todo", "changeme", "placeholder")):
        return KEY_PLACEHOLDER
    # An untouched template such as `sk-xxxx` or `sk-****`.
    if set(raw.replace("sk-", "")) <= set("x*.-_ "):
        return KEY_PLACEHOLDER

    # Too short to be any provider's key. The shortest real one seen in the wild is
    # about 20 characters; 12 leaves room for an internal gateway that uses a short
    # token without accepting obvious placeholders.
    if len(raw) < 12:
        return KEY_SUSPICIOUS

    return KEY_OK


def _mask_key(key: str) -> str:
    """Show enough to recognise a key, never enough to use one."""
    if not key:
        return ""
    if len(key) <= 8:
        return "•" * len(key)
    return f"{key[:4]}…{key[-4:]}"


@dataclass
class ModelSettings:
    """Everything the settings page edits."""

    providers: list[ProviderProfile] = field(default_factory=list)
    #: agent id → provider id. Missing or empty means "use the default provider".
    assignments: dict[str, str] = field(default_factory=dict)
    #: provider id used when an agent has no explicit assignment.
    default_provider: str = ""
    #: agent id → ordered provider ids tried when the primary call fails. This is
    #: the "keep running even if a key expires" path; empty means no fallback
    #: beyond the operator's shared model.
    fallbacks: dict[str, list[str]] = field(default_factory=dict)
    #: Whether model calls stream their reasoning back as it is produced.
    #:
    #: On by default because a stage that thinks for a minute should show progress.
    #: Off means one request that returns when the model is done — fewer wasted
    #: calls when a reply would have been truncated mid-stream, at the cost of a
    #: silent wait. Stored with the provider config because it is a property of how
    #: this deployment calls its models, not of any single provider.
    stream: bool = True

    def provider_for(self, agent_id: str) -> ProviderProfile | None:
        """Resolve which provider an agent should use.

        Resolution order: explicit assignment → default provider → the only usable
        text provider.

        The last step used to fire only when exactly one provider was configured.
        That made a perfectly obvious setup fail: a user with a Zhipu key (text
        model `glm-4.5-air`) and a SiliconFlow key (image model only, no text model)
        had two providers, so the fallback refused to pick between them — every
        run reported "尚未配置大模型" while the settings page showed a green
        "已配置" chip and a filled-in model name. Filling the dialog correctly did
        not help, because nothing in it was wrong.

        "Usable" means: a key, and a text model to call. A media-only provider
        cannot serve a text agent no matter how the user feels about it, so it is
        excluded rather than guessed at. If several remain, the choice is genuinely
        ambiguous and the user must use the assignment table — guessing there would
        silently bill the wrong account.

        An assignment that points at a generation gateway is treated as "no
        assignment". The settings table lets a user pin a *media* model to an
        agent, and an agent like `consistency` needs both a text model and an image
        model; the single `providerId:model` field can only hold one of them. When
        the media one wins (the user picked "本地生成网关 · comfyui（生图）"), every
        chat call used to be sent to the gateway's address and failed with
        "无法连接 LLM 服务（http://<gateway-host>:8190）" the moment that box was off —
        pointing at the network while the real cause was the binding. Falling back
        to the default text provider keeps the run alive; the media capability is
        resolved separately, so the image/video model is still found.
        """
        raw = self.assignments.get(agent_id) or ""
        parts = raw.split(":", 1) if raw else []
        pid = parts[0] if parts else ""
        pinned = parts[1] if len(parts) == 2 else ""
        assigned = self._by_id(pid)

        # One assignment field is shared by the agent's text and media
        # capabilities.  A value such as ``media:qwen-image`` means "use this
        # provider when drawing", not "send the prompt-writing chat request to
        # qwen-image's gateway".  Compare the pinned model with the provider's
        # capability-specific models before accepting the provider for chat.
        # If a deployment deliberately uses the same model name for text and a
        # media capability, the text model wins and the explicit assignment is
        # still honoured.
        non_text_models: set[str] = set()
        if assigned is not None:
            non_text_models = {
                str(assigned.vision_model or "").strip(),
                str(assigned.image_model or "").strip(),
                str(assigned.video_model or "").strip(),
            }
            non_text_models.discard("")
        pins_non_text_model = bool(
            pinned
            and pinned in non_text_models
            and pinned != str(assigned.model or "").strip()
        ) if assigned is not None else False

        if not pins_non_text_model and text_capable(assigned):
            return assigned
        # No assignment, or one that cannot speak chat: try the default next.
        fallback_id = (self.default_provider or "").split(":", 1)[0]
        default = self._by_id(fallback_id)
        if text_capable(default):
            return default
        candidates = [
            p for p in self.providers if profile_usable(p) and p.model and text_capable(p)
        ]
        if len(candidates) == 1:
            return candidates[0]
        # One provider with no text model still counts: a media-only install
        # doing prompt work is a legitimate setup, and refusing it would break
        # exactly the case that always worked. It must still be USABLE though —
        # a keyless or addressless provider cannot be called at all, and
        # selecting it only moves the failure further downstream.
        if (
            not candidates
            and len(self.providers) == 1
            and profile_usable(self.providers[0])
            and text_capable(self.providers[0])
        ):
            return self.providers[0]
        return None

    def _by_id(self, provider_id: str) -> ProviderProfile | None:
        if not provider_id:
            return None
        return next((p for p in self.providers if p.id == provider_id), None)

    def pinned_model(self, agent_id: str, capability: str) -> str:
        """The model explicitly chosen for this agent+capability, if any.

        Storage format is `providerId:model`; the model is the part after the
        first colon (model ids may themselves contain colons, so split once).
        """
        raw = self.assignments.get(agent_id) or ""
        if ":" not in raw:
            return ""
        return raw.split(":", 1)[1]




# ============================================================================
# Persistence
# ============================================================================


def _empty() -> ModelSettings:
    return ModelSettings(providers=[], assignments={}, default_provider="", fallbacks={})


def _as_bool(value: Any, *, default: bool) -> bool:
    """Read a settings flag that may arrive as a bool or a hand-edited string."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


#: 自动升级超时的天花板。没有顶的话，一次配错（比如 1 秒）会被翻倍成
#: 天文数字；而且超过半小时的单次调用在实践中一定是挂死了，不是慢。
TIMEOUT_CEILING = 1800.0


def escalate_timeout(
    model_settings: Any,
    agent_id: str,
    *,
    factor: float = 2.0,
    ceiling: float = TIMEOUT_CEILING,
) -> tuple[float, float] | None:
    """超时重试前，把该岗位所用供应商的超时上限翻倍。

    为什么在重试时升而不是让用户一开始就配大：超时是**确定性**失败——原样
    重发、同一个上限，几乎必然再挂一次，重试形同白等。这一步让每次重试
    「至少比上次多等一倍」，重试才有意义。对只有一个慢模型的用户尤其重要：
    他们没有第二个模型可换，自动升超时是唯一能自救的路。

    只改**本次运行**内存里的对象，不写盘：这是运行期的自救，不是配置的修改。
    返回 (旧值, 新值)；无可升（找不到供应商 / 已到顶）返回 None。
    """
    if model_settings is None:
        return None
    try:
        profile = model_settings.provider_for(agent_id)
    except Exception:  # noqa: BLE001 - 升级失败就按原上限重试，不该因此中断
        return None
    if profile is None:
        return None
    old = float(getattr(profile, "timeout", 0) or 0)
    new = min(old * factor, ceiling)
    if new <= old + 1:  # 已到顶（或翻不动了）
        return None
    try:
        profile.timeout = new
    except Exception:  # noqa: BLE001 - frozen dataclass 等意外，按原上限重试
        return None
    return (old, new)


def load() -> ModelSettings:
    if not PROVIDERS_PATH.exists():
        return _empty()
    try:
        raw = json.loads(PROVIDERS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A corrupted settings file must not prevent the app from starting; the
        # user can re-enter keys rather than being locked out entirely.
        return _empty()

    providers = []
    for item in raw.get("providers") or []:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        providers.append(
            ProviderProfile(
                id=str(item.get("id")),
                preset=str(item.get("preset") or "custom"),
                label=str(item.get("label") or ""),
                base_url=str(item.get("base_url") or ""),
                api_key=str(item.get("api_key") or ""),
                model=str(item.get("model") or ""),
                vision_model=str(item.get("vision_model") or ""),
                # Media capabilities must survive the round trip. Without these
                # two lines the settings page could save `image_model` and the
                # loader silently dropped it, so the generator never saw a media
                # provider and always fell back to the text one.
                image_model=str(item.get("image_model") or ""),
                video_model=str(item.get("video_model") or ""),
                # The endpoint kind has to survive the round trip for the same
                # reason: a user marks a profile "gateway" in the settings dialog,
                # and without this line the loader drops the marker — the gateway
                # is then never selected, and the run silently uses the hosted
                # client against an address that does not speak that protocol.
                kind=str(item.get("kind") or ""),
                timeout=float(item.get("timeout") or 60.0),
                temperature=float(item.get("temperature") or 0.4),
            )
        )
    assignments = {
        str(k): str(v)
        for k, v in (raw.get("assignments") or {}).items()
        if isinstance(v, str) and v
    }
    fallbacks = {
        str(agent): [str(pid) for pid in ids if isinstance(pid, str) and pid]
        for agent, ids in (raw.get("fallbacks") or {}).items()
        if isinstance(ids, list)
    }
    return ModelSettings(
        providers=providers,
        assignments=assignments,
        default_provider=str(raw.get("default_provider") or ""),
        fallbacks=fallbacks,
        # Absent means "never chosen", which is the streaming default. A hand-edited
        # file may carry a string, so both forms are accepted rather than trusting
        # `bool("false") is True`.
        stream=_as_bool(raw.get("stream"), default=True),
    )


def _is_test_process() -> bool:
    """Whether this interpreter is running a script under `tests/`.

    Only used to protect the real settings file. The failure it prevents is silent
    and expensive: a test that isolates `STUDIO_DATA_DIR` but forgets
    `STUDIO_CONFIG_DIR` saves its fixture profiles over the user's
    `backend/config/providers.json`, so their key is replaced by `t-key` and the
    next run reports "Key 是占位值" — a symptom far away from the cause.
    """
    path = getattr(sys.modules.get("__main__"), "__file__", "") or ""
    return "tests" in Path(path).parts


def save(settings: ModelSettings) -> None:
    """Atomic write with owner-only permissions.

    Temp file + replace: a crash mid-write leaves the previous file intact rather
    than a truncated one, which would lose every key at once.
    """
    if _is_test_process():
        target = PROVIDERS_PATH.resolve()
        real_config = (BASE_DIR / "config").resolve()
        if target == real_config / "providers.json" or real_config in target.parents:
            raise RuntimeError(
                "测试进程试图写入真实配置 backend/config/providers.json。"
                "请在测试开头设置 STUDIO_CONFIG_DIR 与 STUDIO_PROVIDERS_PATH "
                "指向临时目录——否则用户的 Key 会被测试夹具覆盖。"
            )
    PROVIDERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "providers": [asdict(p) for p in settings.providers],
        "assignments": settings.assignments,
        "default_provider": settings.default_provider,
        "fallbacks": settings.fallbacks,
        "stream": bool(settings.stream),
    }
    fd, tmp_path = tempfile.mkstemp(dir=str(PROVIDERS_PATH.parent), prefix=".providers-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        try:
            os.chmod(tmp_path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass  # Windows ignores POSIX modes; the file is still outside the repo
        os.replace(tmp_path, PROVIDERS_PATH)
    finally:
        if Path(tmp_path).exists():
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def provider_to_llm_config(profile: ProviderProfile, *, needs_vision: bool = False):
    """Bridge a profile to the LLM client's config object."""
    from .config import LLMConfig

    return LLMConfig(
        base_url=profile.base_url,
        api_key=profile.api_key,
        model=profile.model,
        vision_model=(profile.vision_model or profile.model) if needs_vision else (profile.vision_model or ""),
        timeout_seconds=profile.timeout,
        temperature=profile.temperature,
    )


#: Agents that need image understanding; they must be bound to a vision-capable
#: model. Defined once, near `AGENT_IDS` — a duplicate at the end of the module
#: shadowed the first definition and silently restored the obsolete `intake`.
