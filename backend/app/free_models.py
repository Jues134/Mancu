"""Free model catalogue, the shared "our free model" endpoint, and fallback chains.

Three related features, kept in one module because they share one concern: making
the system runnable when a user has no paid key.

  1. **Catalogue** — models that cost nothing (keyless endpoints, free tiers,
     local Ollama). Each entry declares whether it truly needs no key, because a
     "free" endpoint that silently demands one is worse than no preset at all.
  2. **Shared model** — a deployment operator configures ONE provider via
     environment variables and every user of that instance can select it without
     holding a key. It is rate-limited per run to protect the shared quota.
  3. **Fallback chain** — when an agent's primary model fails (auth error, quota,
     timeout), retry down an ordered list instead of failing the whole run.

Design constraint that shapes everything here: a catalogue entry is only useful if
the endpoint is actually reachable. The `probe` helper lets the UI show live
reachability instead of letting a user pick a dead endpoint and hit a wall later.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


# ============================================================================
# Catalogue
# ============================================================================


@dataclass
class FreeModel:
    """One selectable zero-cost option."""

    id: str
    label: str
    provider_preset: str
    base_url: str
    model: str
    vision_model: str = ""
    #: True only when the endpoint genuinely works without an API key.
    keyless: bool = False
    #: Where the user gets a key / free quota, for options that need one.
    signup_url: str = ""
    note: str = ""
    #: Reachability result from the last probe (see probe_all).
    reachable: bool | None = None


#: Ordered by "most likely to just work" for the users this project targets.
#:
#: Only entries whose endpoint answered from a real machine during development are
#: marked keyless; anything requiring signup is listed with its signup URL rather
#: than pretending to be free.
FREE_MODELS: list[FreeModel] = [
    FreeModel(
        id="ollama-local",
        label="本地 Ollama（完全离线）",
        provider_preset="ollama",
        base_url="http://127.0.0.1:11434/v1",
        model="qwen2.5:7b-instruct",
        keyless=True,
        note="免 Key、不联网、数据不出本机。需要先安装 Ollama 并 pull 模型；"
        "7B 以下的模型写不出稳定的结构化输出。",
    ),
    FreeModel(
        id="zhipu-free",
        label="智谱 GLM（新用户免费额度）",
        provider_preset="zhipu",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model="glm-4-flash",
        vision_model="glm-4v-flash",
        keyless=False,
        signup_url="https://open.bigmodel.cn/usercenter/apikeys",
        note="注册即赠免费额度，glm-4-flash 长期免费；国内直连速度好。",
    ),
    FreeModel(
        id="siliconflow-free",
        label="硅基流动（新用户赠送额度）",
        provider_preset="custom",
        base_url="https://api.siliconflow.cn/v1",
        model="Qwen/Qwen2.5-7B-Instruct",
        keyless=False,
        signup_url="https://cloud.siliconflow.cn/account/ak",
        note="注册赠送额度，国产模型多；适合当备用端点。",
    ),
    FreeModel(
        id="pollinations",
        label="Pollinations（每日免费额度）",
        provider_preset="custom",
        base_url="https://gen.pollinations.ai/v1",
        model="openai-fast",
        keyless=False,
        signup_url="https://enter.pollinations.ai/keys",
        note="旧的免 Key 地址已停用，现在必须注册领取 Key。"
        "该域名在部分网络环境下不可达，建议先用「测试连通性」确认。",
    ),
    FreeModel(
        id="openrouter-free",
        label="OpenRouter 免费模型",
        provider_preset="custom",
        base_url="https://openrouter.ai/api/v1",
        model="meta-llama/llama-3.1-8b-instruct:free",
        keyless=False,
        signup_url="https://openrouter.ai/keys",
        note="有若干 :free 后缀的免费模型，需注册 Key；限流较严。"
        "该域名在部分网络环境下不可达。",
    ),
    FreeModel(
        id="deepseek-trial",
        label="DeepSeek（新用户试用额度）",
        provider_preset="deepseek",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        keyless=False,
        signup_url="https://platform.deepseek.com/api_keys",
        note="国内最常用、中文强、单价极低（约 ¥0.001/千字），"
        "比赛成本可忽略。若追求稳定，建议这项而不是纯免费端点。",
    ),
]


def catalogue() -> list[dict[str, Any]]:
    return [
        {
            "id": m.id,
            "label": m.label,
            "provider_preset": m.provider_preset,
            "base_url": m.base_url,
            "model": m.model,
            "vision_model": m.vision_model,
            "keyless": m.keyless,
            "signup_url": m.signup_url,
            "note": m.note,
            "reachable": m.reachable,
        }
        for m in FREE_MODELS
    ]


def find(model_id: str) -> FreeModel | None:
    return next((m for m in FREE_MODELS if m.id == model_id), None)


# ============================================================================
# Shared "our free model"
# ============================================================================


@dataclass
class SharedModel:
    """A provider the deployment operator pays for, offered to every user.

    This is what "our free model" means in practice: the instance owner sets it up
    once, and a visitor can run the whole pipeline without a personal key. The
    trade-off is that the owner's key is exposed to abuse, so this module also
    enforces a per-run call budget.
    """

    enabled: bool = False
    label: str = ""
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    vision_model: str = ""
    #: Hard ceiling on shared-model calls per run, so one demo cannot drain it.
    calls_per_run: int = 20
    #: Per-request timeout, kept short so a slow shared pool fails fast.
    timeout: float = 45.0

    @property
    def configured(self) -> bool:
        return bool(self.base_url.strip() and self.model.strip() and self.api_key.strip())

    def public(self) -> dict[str, Any]:
        """Never leak the operator's key to the browser."""
        return {
            "enabled": self.enabled and self.configured,
            "label": self.label or "官方共享模型",
            "model": self.model,
            "base_url": self.base_url,
            "calls_per_run": self.calls_per_run,
            # Surfaced so the UI can be honest that this is a limited demo channel.
            "hint": "由部署方提供，按任务限量使用；超出后需自行配置 Key",
        }


def load_shared() -> SharedModel:
    """Read from environment: this is infrastructure config, not user data."""
    return SharedModel(
        enabled=os.environ.get("STUDIO_SHARED_MODEL_ENABLED", "").lower() in {"1", "true", "yes"},
        label=os.environ.get("STUDIO_SHARED_MODEL_LABEL", ""),
        base_url=os.environ.get("STUDIO_SHARED_MODEL_BASE_URL", "").strip(),
        api_key=os.environ.get("STUDIO_SHARED_MODEL_API_KEY", "").strip(),
        model=os.environ.get("STUDIO_SHARED_MODEL_MODEL", "").strip(),
        vision_model=os.environ.get("STUDIO_SHARED_MODEL_VISION_MODEL", "").strip(),
        calls_per_run=int(os.environ.get("STUDIO_SHARED_MODEL_CALLS_PER_RUN", "20") or 20),
    )


# ============================================================================
# Fallback chains
# ============================================================================


@dataclass
class FallbackChain:
    """Ordered provider ids tried in turn when the primary fails.

    Only transient/quota/auth failures trigger a fallback — a content-policy
    rejection or an invalid prompt would fail identically on every model, so
    retrying down the chain there would just burn calls.
    """

    provider_ids: list[str] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        return {"provider_ids": self.provider_ids}


#: Errors worth trying the next model for.
#: Deliberately excludes content-policy and malformed-output failures.
FALLBACK_ELIGIBLE = (
    "auth", "quota", "rate", "timeout", "connection", "server", "unknown_model",
)


#: Error fragments that mean "stop the run", not "try another model".
#: The pipeline checks this to halt instead of re-dispatching.
TERMINAL_MARKERS = (
    # Every wording observed from a real provider in this project. Covers both the
    # raw body ("余额不足或无可用资源包,请充值") and this app's diagnosis
    # ("账户余额或额度不足"), because a rule matching one but not the other looks
    # implemented while failing in practice.
    "余额不足",
    "余额或额度不足",
    "账户余额",
    "额度不足",
    "欠费",
    "请充值",
    "无可用资源包",
    "insufficient balance",
    "insufficient_quota",
    "payment required",
    "no credit",
)


def is_terminal(error_text: str) -> bool:
    """Whether this error ends the run rather than triggering a retry.

    Separate from `should_fallback` because the consequences differ: a fallback
    tries one more model and moves on, while a terminal error must stop
    dispatching entirely and tell the user what to fix.
    """
    text = (error_text or "").lower()
    return any(marker in text for marker in TERMINAL_MARKERS)


#: error_type → 换一个模型有没有用。
#:
#: 这张表比下面那串关键词可靠：关键词要跟中文文案一起改，改一次文案就漏一类。
#: `LLMError.error_type` 是分类的唯一来源，这里只负责把它翻译成「换不换」。
#: 值为 None 表示这张表里没有，退回按文案判断（老调用点还会传裸字符串）。
_FALLBACK_BY_TYPE: dict[str, bool] = {
    # 换模型解决不了：同一段文字、同一个审核标准，换一家大概还是同一个结果，
    # 而每换一次都要重新生成整段，代价是几分钟和一次计费。
    "moderation_in": False,
    "moderation_out": False,
    "invalid_request": False,
    "balance": False,
    "not_configured": False,
    # 换模型能解决：这是链路或配额问题，不是这段提示词的问题。
    "auth": True,
    "not_found": True,
    "rate_limit": True,
    "timeout": True,
    "transport": True,
    "unreachable": True,
    "http_error": True,
    "bad_response": True,
    # 输出格式/长度问题：换个模型确实可能就好，但更该先调大预算，所以这里
    # 判 False —— 由 `chat_json` 自己的加倍预算重试负责，不必换供应商。
    "truncated": False,
    "empty_reply": False,
    "unparseable": False,
}


def should_fallback(error_text: str, error_type: str = "") -> bool:
    """Classify an LLM error as "try another model" or "fix the request".

    Keeping this a small pure function makes the policy testable and keeps it
    from silently drifting into retrying non-recoverable failures (which would
    multiply spend on a prompt that can never succeed).

    `error_type` 优先于文案：前者是调用方给出的分类，后者是猜的。两个都传的时候
    以分类为准，因为文案会改，分类不会。
    """
    if error_type and error_type in _FALLBACK_BY_TYPE:
        return _FALLBACK_BY_TYPE[error_type]

    text = (error_text or "").lower()

    # Balance exhaustion is terminal. Tried first, before the general quota
    # branch, because "余额不足" would otherwise match the recoverable rule.
    #
    # Why terminal: another model cannot conjure credit. Falling through would
    # either bill a different account the user did not choose, or fail the same
    # way after several minutes — and the surfaced error would describe the last
    # attempt rather than the real cause.
    # One marker list, shared with `is_terminal`. Two hand-maintained tuples for
    # the same policy drifted apart: this one matched "余额不足" but not
    # "余额或额度不足", so the diagnosis produced by this app's own error layer
    # was classified as recoverable.
    if is_terminal(text):
        return False

    if any(k in text for k in ("content policy", "moderation", "safety", "敏感", "审核拒绝")):
        return False
    if any(k in text for k in ("不是合法 json", "无法解析", "invalid character")):
        return False
    if any(k in text for k in ("401", "403", "unauthorized", "forbidden", "api key", "鉴权")):
        return True
    if any(k in text for k in ("429", "rate limit", "quota", "余额", "限流", "额度")):
        return True
    if any(k in text for k in ("timeout", "timed out", "超时", "connection", "无法连接",
                               "network", "中途被中断", "连接被重置", "connection reset")):
        return True
    if any(k in text for k in ("500", "502", "503", "504", "server error")):
        return True
    if any(k in text for k in ("404", "model not found", "不存在", "unknown model")):
        return True
    return False


# ============================================================================
# Reachability probe
# ============================================================================


async def probe(url: str, timeout: float = 8.0) -> bool:
    """Can this endpoint be reached at all?

    A catalogue entry a user cannot reach is worse than one that is absent: they
    pick it, the run fails, and the failure looks like a bug in the app. This
    probe lets the UI mark entries as unreachable up front.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Any HTTP answer — even 401 — proves the endpoint is reachable; only a
            # transport failure means it is not.
            resp = await client.get(url, headers={"User-Agent": "studio-probe/1.0"})
            return resp.status_code < 500
    except Exception:  # noqa: BLE001 - probe never raises
        return False


async def probe_all(timeout: float = 8.0) -> list[dict[str, Any]]:
    """Probe every catalogue entry concurrently and cache the result."""

    async def one(m: FreeModel) -> None:
        m.reachable = await probe(m.base_url.rstrip("/") + "/models", timeout=timeout)

    await asyncio.gather(*(one(m) for m in FREE_MODELS))
    return catalogue()


def shared_budget_remaining(calls_used: int) -> int:
    shared = load_shared()
    return max(0, shared.calls_per_run - calls_used)
