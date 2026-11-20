"""Channel aggregation proxy — the "free model relay" layer.

This is the piece the reference plugin calls 白嫖模型接入 / 反代. Its job is to
present ONE OpenAI-compatible endpoint to the agents while, behind it, rotating
across many providers and many accounts per provider that offer free quota.

Why this exists as a separate layer rather than N entries in the provider list:
  * Free tiers are per-ACCOUNT, so quota is the scarce resource. Aggregating
    accounts behind one logical channel is what makes "free" scale to a real run.
  * Free tiers expire, hit 429, or demand a daily check-in. The relay is the only
    place that can observe that and switch accounts without the agent noticing.
  * Keys must not reach the browser. The relay holds credentials server-side and
    the UI only sees masked hints and counts.

Three concepts:
  channel  — one provider offering free quota (e.g. Huawei CodeArts, GLM)
  account  — one credential on that channel (several per channel is normal)
  route    — how a request picks a channel/account (goal + model mapping)

The proxy is deliberately a *forwarding* layer: it does not rewrite prompts or
parse responses beyond what is needed to detect failures and count usage. Any
cleverness there would corrupt the agent's structured output.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .config import CONFIG_DIR

#: Relay channels carry API keys, so this file lives with the rest of the
#: machine configuration rather than in the user's data directory.
CHANNELS_PATH = CONFIG_DIR / "channels.json"


# ============================================================================
# Channel presets — what each free provider actually offers
# ============================================================================


@dataclass
class ChannelPreset:
    """A known free-quota provider, with the facts a user needs to judge it."""

    id: str
    label: str
    base_url: str
    default_model: str
    #: Free quota described honestly: "daily tokens" vs "signup bonus" differ.
    quota_kind: str
    quota_note: str
    #: Where the user obtains a credential.
    signup_url: str
    #: Whether the provider needs a daily check-in to keep quota alive.
    daily_checkin: bool
    docs_url: str = ""


#: Ordered by likelihood of being usable on a mainland-China connection.
#: Quotas are the providers' publicly documented free tiers, not guesses; where a
#: number changes often the note says so instead of hard-coding a stale figure.
CHANNEL_PRESETS: list[ChannelPreset] = [
    ChannelPreset(
        id="huawei-codearts",
        label="华为云 CodeArts",
        base_url="https://maas-api.cn-east-3.myhuaweicloud.com/v1",
        default_model="deepseek-v3",
        quota_kind="每日免费额度",
        quota_note="实名后每日赠送免费 Token，DeepSeek/GLM/openPangu 多模型；额度按天刷新",
        signup_url="https://console.huaweicloud.com/codeartsmodelarts/",
        daily_checkin=False,
        docs_url="https://support.huaweicloud.com/productdesc-codeartsmodelarts/",
    ),
    ChannelPreset(
        id="zhipu",
        label="智谱 GLM",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-4-flash",
        quota_kind="长期免费档",
        quota_note="glm-4-flash 长期免费，限并发与速率；注册即用，无需签到",
        signup_url="https://open.bigmodel.cn/usercenter/apikeys",
        daily_checkin=False,
    ),
    ChannelPreset(
        id="siliconflow",
        label="硅基流动",
        base_url="https://api.siliconflow.cn/v1",
        default_model="Qwen/Qwen2.5-7B-Instruct",
        quota_kind="注册赠送额度",
        quota_note="注册赠送一定额度（约 ¥14 等值），用完为止；多账号可叠加",
        signup_url="https://cloud.siliconflow.cn/account/ak",
        daily_checkin=False,
    ),
    ChannelPreset(
        id="aliyun-bailian",
        label="阿里百炼",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-turbo",
        quota_kind="新用户免费额度",
        quota_note="qwen-turbo 等模型有免费调用额度；开通后按月/按量计算",
        signup_url="https://bailian.console.aliyun.com/",
        daily_checkin=False,
    ),
    ChannelPreset(
        id="modelscope",
        label="魔搭 ModelScope",
        base_url="https://api-inference.modelscope.cn/v1",
        default_model="Qwen/Qwen2.5-7B-Instruct",
        quota_kind="每日免费调用",
        quota_note="阿里达摩院社区，绑定阿里云账号后每日免费调用次数；限流较严",
        signup_url="https://modelscope.cn/my/myaccesstoken",
        daily_checkin=False,
    ),
    ChannelPreset(
        id="openrouter",
        label="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        default_model="meta-llama/llama-3.1-8b-instruct:free",
        quota_kind="免费模型档",
        quota_note="带 :free 后缀的模型免费但限流严格；需注册领取 Key",
        signup_url="https://openrouter.ai/keys",
        daily_checkin=False,
    ),
    ChannelPreset(
        id="custom",
        label="自定义渠道",
        base_url="",
        default_model="",
        quota_kind="自定义",
        quota_note="任意 OpenAI 兼容端点，自建代理或公司内网网关都可以",
        signup_url="",
        daily_checkin=False,
    ),
]


def preset_by_id(pid: str) -> ChannelPreset | None:
    return next((p for p in CHANNEL_PRESETS if p.id == pid), None)


# ============================================================================
# Data model
# ============================================================================


@dataclass
class ChannelAccount:
    """One credential on a channel. Quota is tracked per account, not per channel."""

    id: str
    label: str = ""
    api_key: str = ""
    #: Set when the provider issues a separate token for a different endpoint.
    base_url: str = ""
    model: str = ""
    enabled: bool = True
    #: Local bookkeeping so the UI can show what is left without extra API calls.
    quota_used: int = 0
    quota_total: int = 0
    quota_unit: str = "tokens"
    last_used_at: float = 0.0
    last_error: str = ""
    #: Set when the provider rate-limits this account; the relay skips it until then.
    cooldown_until: float = 0.0

    @property
    def usable(self) -> bool:
        if not self.enabled:
            return False
        if self.cooldown_until and self.cooldown_until > time.time():
            return False
        if self.quota_total and self.quota_used >= self.quota_total:
            return False
        return True

    def public(self) -> dict[str, Any]:
        """Masked projection for the browser."""
        return {
            "id": self.id,
            "label": self.label,
            "enabled": self.enabled,
            "has_key": bool(self.api_key),
            "key_hint": _mask(self.api_key),
            "model": self.model,
            "quota_used": self.quota_used,
            "quota_total": self.quota_total,
            "quota_unit": self.quota_unit,
            "usable": self.usable,
            "cooldown_seconds": max(0, int(self.cooldown_until - time.time())),
            "last_error": self.last_error[:160],
        }


@dataclass
class Channel:
    """One provider plus its accounts."""

    id: str
    preset: str
    label: str
    base_url: str
    default_model: str = ""
    enabled: bool = True
    #: Daily check-in support: the relay records when it last ran, and reports
    #: whether it succeeded per account.
    daily_checkin: bool = False
    last_checkin_at: float = 0.0
    last_checkin_result: str = ""
    accounts: list[ChannelAccount] = field(default_factory=list)

    def usable_accounts(self) -> list[ChannelAccount]:
        return [a for a in self.accounts if a.usable]

    def public(self, preset: ChannelPreset | None) -> dict[str, Any]:
        usable = self.usable_accounts()
        return {
            "id": self.id,
            "preset": self.preset,
            "label": self.label,
            "base_url": self.base_url,
            "default_model": self.default_model,
            "enabled": self.enabled,
            "daily_checkin": self.daily_checkin,
            "last_checkin_at": self.last_checkin_at,
            "last_checkin_result": self.last_checkin_result,
            "account_count": len(self.accounts),
            "usable_account_count": len(usable),
            "accounts": [a.public() for a in self.accounts],
            "quota_note": preset.quota_note if preset else "",
            "signup_url": preset.signup_url if preset else "",
        }


@dataclass
class RelaySettings:
    """The relay as a whole."""

    channels: list[Channel] = field(default_factory=list)
    #: When true, agents can address the relay as a provider (id "relay").
    enabled: bool = True
    #: Accounts tried per request before giving up.
    max_attempts: int = 4
    #: Rotation: sticky keeps one account per channel for cache friendliness;
    #: round_robin spreads load so no single account burns its quota first.
    rotation: str = "round_robin"
    #: In-memory cursor for round-robin, persisted so restarts do not restart at 0.
    cursor: int = 0


def _mask(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "•" * len(key)
    return f"{key[:4]}…{key[-4:]}"


# ============================================================================
# Persistence
# ============================================================================


def _empty() -> RelaySettings:
    return RelaySettings()


def load() -> RelaySettings:
    if not CHANNELS_PATH.exists():
        return _empty()
    try:
        raw = json.loads(CHANNELS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # Same rule as provider settings: a corrupt file must not block startup.
        return _empty()

    channels: list[Channel] = []
    for item in raw.get("channels") or []:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        accounts = []
        for acc in item.get("accounts") or []:
            if not isinstance(acc, dict) or not acc.get("id"):
                continue
            accounts.append(
                ChannelAccount(
                    id=str(acc.get("id")),
                    label=str(acc.get("label") or ""),
                    api_key=str(acc.get("api_key") or ""),
                    base_url=str(acc.get("base_url") or ""),
                    model=str(acc.get("model") or ""),
                    enabled=bool(acc.get("enabled", True)),
                    quota_used=int(acc.get("quota_used") or 0),
                    quota_total=int(acc.get("quota_total") or 0),
                    quota_unit=str(acc.get("quota_unit") or "tokens"),
                    last_used_at=float(acc.get("last_used_at") or 0),
                    last_error=str(acc.get("last_error") or ""),
                    cooldown_until=float(acc.get("cooldown_until") or 0),
                )
            )
        channels.append(
            Channel(
                id=str(item.get("id")),
                preset=str(item.get("preset") or "custom"),
                label=str(item.get("label") or ""),
                base_url=str(item.get("base_url") or ""),
                default_model=str(item.get("default_model") or ""),
                enabled=bool(item.get("enabled", True)),
                daily_checkin=bool(item.get("daily_checkin", False)),
                last_checkin_at=float(item.get("last_checkin_at") or 0),
                last_checkin_result=str(item.get("last_checkin_result") or ""),
                accounts=accounts,
            )
        )
    return RelaySettings(
        channels=channels,
        enabled=bool(raw.get("enabled", True)),
        max_attempts=int(raw.get("max_attempts") or 4),
        rotation=str(raw.get("rotation") or "round_robin"),
        cursor=int(raw.get("cursor") or 0),
    )


def save(settings: RelaySettings) -> None:
    """Atomic write, owner-only: this file holds several credentials at once."""
    CHANNELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "enabled": settings.enabled,
        "max_attempts": settings.max_attempts,
        "rotation": settings.rotation,
        "cursor": settings.cursor,
        "channels": [asdict(c) for c in settings.channels],
    }
    fd, tmp = tempfile.mkstemp(dir=str(CHANNELS_PATH.parent), prefix=".channels-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        try:
            os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        os.replace(tmp, CHANNELS_PATH)
    finally:
        if Path(tmp).exists():
            try:
                os.unlink(tmp)
            except OSError:
                pass
