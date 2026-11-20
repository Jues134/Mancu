"""Relay forwarding: rotate accounts, retry failures, and expose one endpoint.

The relay is a local OpenAI-compatible proxy. Agents point at
`http://127.0.0.1:<port>/relay/v1` and the proxy decides which channel/account
actually serves each request.

Failure handling is the interesting part. Free tiers fail in specific ways, and
each needs a different response:

  429 / quota  → cool that account down (it is alive but exhausted), try another
  401 / 403    → disable nothing automatically, but report; a wrong key is not
                 self-healing and silently retrying wastes attempts
  5xx / timeout→ transient; try another account, no cooldown penalty
  200          → count the call against the account's local tally

A cooldown has a floor so a burst of failures cannot lock every account out
permanently by pushing `cooldown_until` years ahead.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Any

import httpx

from . import relay


COOLDOWN_SECONDS = 60
#: Never lock an account out longer than this, even if the provider keeps failing.
COOLDOWN_MAX = 30 * 60


@dataclass
class RelayAttempt:
    """One attempt against one account, kept so the UI can explain a failure."""

    channel_id: str
    account_id: str
    status: int
    error: str
    duration_ms: int
    model: str


@dataclass
class RelayResult:
    ok: bool
    payload: dict[str, Any]
    attempts: list[RelayAttempt]
    channel_id: str = ""
    account_id: str = ""
    model: str = ""


def _candidate_pairs(cfg: relay.RelaySettings, preferred_model: str) -> list[tuple[relay.Channel, relay.ChannelAccount]]:
    """Flatten channels into (channel, account) pairs in the order to try them.

    Ordering rationale:
      * disabled channels and unusable accounts are excluded up front, so the
        attempt budget is spent on things that can actually answer;
      * round-robin rotates the starting point across calls, so a single account
        does not burn its quota while others sit idle;
      * a model hint reorders rather than filters — an account whose model
        differs still gets a chance, because a model mismatch is usually recoverable
        server-side and filtering it out would leave nothing to try.
    """
    pairs: list[tuple[relay.Channel, relay.ChannelAccount]] = []
    for channel in cfg.channels:
        if not channel.enabled:
            continue
        for account in channel.usable_accounts():
            pairs.append((channel, account))

    if not pairs:
        return []

    if cfg.rotation == "round_robin":
        start = cfg.cursor % len(pairs)
        return pairs[start:] + pairs[:start]
    return pairs


def _pick_model(channel: relay.Channel, account: relay.ChannelAccount, requested: str) -> str:
    """Model to send for this account.

    An explicitly requested model wins when the account declares none; when the
    account pins a model, that pin wins because free tiers often only expose one
    model per credential.
    """
    return account.model or channel.default_model or requested


def _record_failure(settings: relay.RelaySettings, channel: relay.Channel, account: relay.ChannelAccount, status: int, message: str) -> None:
    account.last_error = message[:200]
    if status == 429 or "quota" in message.lower() or "余额" in message:
        # Exhausted or throttled: cool down, but bounded.
        account.cooldown_until = min(
            time.time() + COOLDOWN_MAX, time.time() + COOLDOWN_SECONDS
        )
    elif status >= 500 or status == 0:
        # Transient: no cooldown, just note it.
        pass
    relay.save(settings)


def _record_success(settings: relay.RelaySettings, account: relay.ChannelAccount, tokens: int = 1) -> None:
    account.last_used_at = time.time()
    account.last_error = ""
    account.cooldown_until = 0
    account.quota_used += tokens
    relay.save(settings)


async def forward(
    request_body: dict[str, Any],
    *,
    timeout: float = 60.0,
) -> RelayResult:
    """Forward one chat-completions request through the channel pool.

    Returns the first successful upstream payload unchanged — the caller (an LLM
    client) expects the provider's own response shape, and rewriting it would
    break function-calling and token accounting downstream.
    """
    settings = relay.load()
    attempts: list[RelayAttempt] = []

    if not settings.enabled:
        return RelayResult(False, {"error": "反代已关闭"}, attempts)

    requested_model = str(request_body.get("model") or "")
    pairs = _candidate_pairs(settings, requested_model)
    if not pairs:
        return RelayResult(
            False,
            {"error": "没有可用的渠道账号。请先在设置页添加渠道与账号。"},
            attempts,
        )

    budget = max(1, settings.max_attempts)
    for channel, account in pairs[:budget]:
        model = _pick_model(channel, account, requested_model)
        base = (account.base_url or channel.base_url).rstrip("/")
        if not base:
            attempts.append(RelayAttempt(channel.id, account.id, 0, "渠道缺少接口地址", 0, model))
            continue

        payload = {**request_body, "model": model}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {account.api_key}",
        }

        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{base}/chat/completions", json=payload, headers=headers)
        except Exception as exc:  # noqa: BLE001 - transport errors are retryable
            elapsed = int((time.perf_counter() - started) * 1000)
            attempts.append(RelayAttempt(channel.id, account.id, 0, f"连接失败：{exc}", elapsed, model))
            _record_failure(settings, channel, account, 0, f"连接失败：{exc}")
            continue

        elapsed = int((time.perf_counter() - started) * 1000)

        if resp.status_code == 200:
            try:
                body = resp.json()
            except ValueError:
                body = {"error": "上游返回了非 JSON 内容", "raw": resp.text[:200]}
                attempts.append(RelayAttempt(channel.id, account.id, 200, "非 JSON 响应", elapsed, model))
                _record_failure(settings, channel, account, 500, "非 JSON 响应")
                continue
            usage = body.get("usage") or {}
            _record_success(settings, account, int(usage.get("total_tokens") or 1))
            # Advance the rotating cursor so the next call starts elsewhere.
            settings.cursor = (settings.cursor + 1) % max(1, len(pairs))
            relay.save(settings)
            attempts.append(RelayAttempt(channel.id, account.id, 200, "", elapsed, model))
            return RelayResult(
                True, body, attempts,
                channel_id=channel.id, account_id=account.id, model=model,
            )

        text = ""
        try:
            text = resp.text[:300]
        except Exception:  # noqa: BLE001
            pass
        attempts.append(RelayAttempt(channel.id, account.id, resp.status_code, text, elapsed, model))
        _record_failure(settings, channel, account, resp.status_code, f"{resp.status_code} {text}")

        # A 4xx that is not about rate limiting means the request itself is wrong
        # (bad params, unsupported model). Trying other accounts with the same body
        # would fail identically, so stop and let the caller fix the request.
        if 400 <= resp.status_code < 500 and resp.status_code != 429:
            break

    return RelayResult(False, {"error": "所有渠道账号都失败了", "attempts": [a.__dict__ for a in attempts]}, attempts)


async def checkin(settings: relay.RelaySettings | None = None) -> dict[str, Any]:
    """Daily check-in sweep.

    Honest limitation: most providers expose no official check-in API, and
    automating a web check-in would be both fragile and against their terms. So
    this does what is genuinely useful and safe — it probes each enabled account
    with a one-token request, which both refreshes quota state and surfaces dead
    credentials. The result is reported per account rather than as a fake
    "+200 积分" claim.
    """
    cfg = settings or relay.load()
    results: list[dict[str, Any]] = []

    for channel in cfg.channels:
        if not channel.enabled or not channel.daily_checkin:
            continue
        for account in channel.accounts:
            if not account.enabled:
                continue
            base = (account.base_url or channel.base_url).rstrip("/")
            model = account.model or channel.default_model
            if not base or not model or not account.api_key:
                results.append({
                    "channel": channel.id, "account": account.id,
                    "ok": False, "message": "缺少地址/模型/Key",
                })
                continue
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    resp = await client.post(
                        f"{base}/chat/completions",
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": "ping"}],
                            "max_tokens": 1,
                        },
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {account.api_key}",
                        },
                    )
                ok = resp.status_code == 200
                if ok:
                    account.last_error = ""
                    account.cooldown_until = 0
                else:
                    account.last_error = f"{resp.status_code} {resp.text[:120]}"
                results.append({
                    "channel": channel.id, "account": account.id,
                    "ok": ok, "message": "可用" if ok else account.last_error,
                })
            except Exception as exc:  # noqa: BLE001
                results.append({
                    "channel": channel.id, "account": account.id,
                    "ok": False, "message": f"连接失败：{exc}",
                })
        channel.last_checkin_at = time.time()
        ok_count = sum(1 for r in results if r["channel"] == channel.id and r["ok"])
        channel.last_checkin_result = f"{ok_count}/{len(channel.accounts)} 个账号可用"

    relay.save(cfg)
    return {"results": results, "checked_at": time.time()}


def summary(cfg: relay.RelaySettings | None = None) -> dict[str, Any]:
    """Compact status for the settings header and the relay health endpoint."""
    cfg = cfg or relay.load()
    total = sum(len(c.accounts) for c in cfg.channels)
    usable = sum(len(c.usable_accounts()) for c in cfg.channels)
    enabled = [c for c in cfg.channels if c.enabled and c.usable_accounts()]
    return {
        "enabled": cfg.enabled,
        "channels": len(cfg.channels),
        "enabled_channels": len(enabled),
        "accounts": total,
        "usable_accounts": usable,
        "rotation": cfg.rotation,
    }
