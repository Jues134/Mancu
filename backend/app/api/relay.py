"""Relay API: channel management plus the OpenAI-compatible proxy endpoint.

The proxy endpoint (`/relay/v1/chat/completions`) is what agents point at. It is
intentionally a *pass-through*: the request body is forwarded with only the model
substituted, and the upstream response is returned untouched. Rewriting either
side would break the OpenAI contract that every client here depends on.

Channel management endpoints never return raw credentials — only masked hints —
because this file holds several keys at once.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .. import relay, relay_proxy
from ..relay import CHANNEL_PRESETS, Channel, ChannelAccount, RelaySettings

router = APIRouter()


class AccountInput(BaseModel):
    id: str = ""
    label: str = ""
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    enabled: bool = True
    quota_total: int = 0
    quota_used: int = 0


class ChannelInput(BaseModel):
    id: str = ""
    preset: str = "custom"
    label: str = ""
    base_url: str = ""
    default_model: str = ""
    enabled: bool = True
    daily_checkin: bool = False
    accounts: list[AccountInput] = []


class RelayPayload(BaseModel):
    enabled: bool = True
    rotation: str = "round_robin"
    max_attempts: int = 4
    channels: list[ChannelInput] = []


@router.get("/api/relay/presets")
async def get_presets() -> dict:
    return {
        "presets": [
            {
                "id": p.id,
                "label": p.label,
                "base_url": p.base_url,
                "default_model": p.default_model,
                "quota_kind": p.quota_kind,
                "quota_note": p.quota_note,
                "signup_url": p.signup_url,
                "daily_checkin": p.daily_checkin,
                "docs_url": p.docs_url,
            }
            for p in CHANNEL_PRESETS
        ]
    }


@router.get("/api/relay")
async def get_relay() -> dict:
    cfg = relay.load()
    return {
        "enabled": cfg.enabled,
        "rotation": cfg.rotation,
        "max_attempts": cfg.max_attempts,
        "summary": relay_proxy.summary(cfg),
        "channels": [
            c.public(relay.preset_by_id(c.preset)) for c in cfg.channels
        ],
        # Agents address this path as a provider; surfaced so the UI can show it.
        "endpoint": "/relay/v1",
    }


@router.put("/api/relay")
async def put_relay(payload: RelayPayload) -> dict:
    existing = relay.load()
    old_channels = {c.id: c for c in existing.channels}
    old_accounts = {
        (c.id, a.id): a
        for c in existing.channels
        for a in c.accounts
    }

    channels: list[Channel] = []
    for raw in payload.channels:
        cid = (raw.id or "").strip()
        if not cid:
            raise HTTPException(400, "渠道缺少 id")
        if not raw.base_url.strip():
            raise HTTPException(400, f"渠道「{raw.label or cid}」缺少接口地址")

        accounts: list[ChannelAccount] = []
        for acc in raw.accounts:
            aid = (acc.id or "").strip()
            if not aid:
                raise HTTPException(400, f"渠道「{raw.label or cid}」有账号缺少 id")
            # Empty key keeps the stored one: the UI never round-trips secrets.
            key = acc.api_key.strip() or old_accounts.get((cid, aid), ChannelAccount(id=aid)).api_key
            accounts.append(
                ChannelAccount(
                    id=aid,
                    label=acc.label,
                    api_key=key,
                    base_url=acc.base_url.strip(),
                    model=acc.model.strip(),
                    enabled=acc.enabled,
                    quota_total=acc.quota_total,
                    quota_used=acc.quota_used,
                    quota_unit=old_accounts.get((cid, aid), ChannelAccount(id=aid)).quota_unit,
                    last_used_at=old_accounts.get((cid, aid), ChannelAccount(id=aid)).last_used_at,
                )
            )

        prior = old_channels.get(cid)
        preset = relay.preset_by_id(raw.preset)
        channels.append(
            Channel(
                id=cid,
                preset=raw.preset,
                # Label resolution: explicit label → preset label → id. Written as
                # nested ifs because mixing `or` with a conditional expression
                # silently binds to the wrong operand.
                label=raw.label or (preset.label if preset else cid),
                base_url=raw.base_url.strip(),
                default_model=raw.default_model.strip(),
                enabled=raw.enabled,
                daily_checkin=raw.daily_checkin,
                last_checkin_at=prior.last_checkin_at if prior else 0.0,
                last_checkin_result=prior.last_checkin_result if prior else "",
                accounts=accounts,
            )
        )

    cfg = RelaySettings(
        channels=channels,
        enabled=payload.enabled,
        rotation=payload.rotation if payload.rotation in {"round_robin", "sticky"} else "round_robin",
        max_attempts=max(1, min(10, payload.max_attempts)),
        cursor=0,
    )
    relay.save(cfg)
    return {
        "enabled": cfg.enabled,
        "summary": relay_proxy.summary(cfg),
        "channels": [c.public(relay.preset_by_id(c.preset)) for c in cfg.channels],
    }


@router.post("/api/relay/checkin")
async def run_checkin() -> dict:
    """Probe every check-in-enabled account and report the truth per account.

    This does NOT claim free credits were earned: most providers have no official
    check-in API, and scraping their web UI would be fragile and against terms.
    What it does is verify credentials still work and refresh cooldown state.
    """
    cfg = relay.load()
    if not any(c.daily_checkin and c.enabled for c in cfg.channels):
        return {
            "results": [],
            "checked_at": time.time(),
            "note": "没有渠道开启签到。签到仅用于验证凭据与刷新额度状态，不会伪造积分。",
        }
    result = await relay_proxy.checkin(cfg)
    return result


@router.post("/api/relay/test/{channel_id}/{account_id}")
async def test_account(channel_id: str, account_id: str) -> dict:
    """One minimal request proving a specific account works."""
    cfg = relay.load()
    channel = next((c for c in cfg.channels if c.id == channel_id), None)
    if channel is None:
        raise HTTPException(404, "渠道不存在")
    account = next((a for a in channel.accounts if a.id == account_id), None)
    if account is None:
        raise HTTPException(404, "账号不存在")

    import httpx

    base = (account.base_url or channel.base_url).rstrip("/")
    model = account.model or channel.default_model
    if not base or not model:
        return {"ok": False, "error": "缺少接口地址或模型"}
    if not account.api_key:
        return {"ok": False, "error": "缺少 API Key"}

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            resp = await client.post(
                f"{base}/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 8,
                },
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {account.api_key}"},
            )
        if resp.status_code == 200:
            body = resp.json()
            reply = ""
            try:
                reply = body["choices"][0]["message"]["content"][:40]
            except (KeyError, IndexError, TypeError):
                reply = ""
            return {"ok": True, "model": model, "reply": reply}
        return {"ok": False, "error": f"HTTP {resp.status_code}", "detail": resp.text[:200]}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"连接失败：{exc}"}


@router.get("/api/relay/health")
async def relay_health() -> dict:
    return relay_proxy.summary()


# ============================================================================
# The proxy endpoint itself
# ============================================================================


@router.post("/relay/v1/chat/completions")
async def relay_chat_completions(payload: dict[str, Any]) -> JSONResponse:
    """OpenAI-compatible pass-through across the channel pool.

    Agents bind to this as a provider (`base_url = http://127.0.0.1:<port>/relay/v1`)
    with any placeholder key. Rotation, quota tracking and failure handling all
    happen inside; the caller sees one stable endpoint.
    """
    result = await relay_proxy.forward(payload)
    if result.ok:
        return JSONResponse(result.payload)
    return JSONResponse(
        status_code=502,
        content={
            "error": result.payload.get("error", "反代调用失败"),
            "attempts": [a.__dict__ for a in result.attempts],
        },
    )


@router.get("/relay/v1/models")
async def relay_models() -> dict:
    """Models reachable through the relay, so model pickers can list them."""
    cfg = relay.load()
    seen: dict[str, dict[str, Any]] = {}
    for channel in cfg.channels:
        if not channel.enabled:
            continue
        for account in channel.accounts:
            if not account.enabled:
                continue
            model = account.model or channel.default_model
            if not model or model in seen:
                continue
            seen[model] = {
                "id": model,
                "object": "model",
                "owned_by": channel.id,
                "channel": channel.label,
            }
    return {"object": "list", "data": list(seen.values())}
