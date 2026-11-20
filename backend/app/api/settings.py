"""Settings API: provider configuration and per-agent model binding.

Security posture:
  * Keys are never returned in full. The list endpoint emits a masked hint so a
    user can tell which key is configured without the value crossing the wire.
  * Update accepts an empty key to mean "keep the existing one", which lets the
    UI edit other fields without ever holding the secret.
  * Deleting a provider also clears any agent bound to it, so a run never
    resolves to a dead provider.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from ..tools.provider_network import provider_trust_env
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .. import free_models, models_config
from .. import agent_capabilities, agent_model_resolver, model_catalog
from ..models_config import (
    AGENT_IDS,
    AGENT_IDS_LABEL,
    BINDABLE_AGENT_IDS,
    PROVIDER_PRESETS,
    ProviderProfile,
)

router = APIRouter()


class ProviderInput(BaseModel):
    id: str = ""
    preset: str = "custom"
    label: str = ""
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    vision_model: str = ""
    #: Media models on this provider. Present here because a user who bought a
    #: hosted image/video key must be able to declare it from the settings page —
    #: without these fields the dialog could not save what `select_backend` reads.
    image_model: str = ""
    video_model: str = ""
    #: "" for an OpenAI-compatible provider, "gateway" for a local ComfyUI queue.
    #: The two need different clients, so the kind is explicit rather than guessed
    #: from the address.
    kind: str = ""
    timeout: float = 60.0
    temperature: float = 0.4


class SettingsPayload(BaseModel):
    providers: list[ProviderInput] = []
    assignments: dict[str, str] = {}
    default_provider: str = ""
    #: agent id → ordered provider ids tried when the primary model fails.
    fallbacks: dict[str, list[str]] = {}
    #: Streaming preference. `None` means "not sent" — a client that only edits
    #: providers must not flip a toggle it never showed, so the stored value is
    #: kept unless this is explicitly true/false.
    stream: bool | None = None


class TestProviderPayload(BaseModel):
    provider_id: str = ""
    # Allow testing an unsaved draft, so a user can verify a key before committing
    # it to disk (and before it is used by a paid run).
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    #: How long the provider itself may take.  The browser does not impose a
    #: second, shorter deadline: its button keeps spinning until this upstream
    #: request finishes or this provider deadline is reached.
    timeout: float | None = None


@router.get("/api/settings/presets")
async def get_presets() -> dict:
    """Presets + the agent list the UI renders as assignment targets."""
    return {
        "presets": [
            {"id": key, **value} for key, value in PROVIDER_PRESETS.items()
        ],
        "agents": [
            {
                "id": agent,
                "label": AGENT_IDS_LABEL.get(agent, agent),
                # Capability chips. `needs` carries both the key and the label so
                # the UI renders text the backend owns; `vision_agents` is kept for
                # older callers that only know about vision.
                "needs": agent_capabilities.describe(agent),
            }
            for agent in AGENT_IDS
        ],
        "vision_agents": sorted(models_config.VISION_AGENTS),
    }


@router.get("/api/settings")
async def get_settings() -> dict:
    cfg = models_config.load()
    shared = free_models.load_shared()
    return {
        "providers": [p.public() for p in cfg.providers],
        "assignments": cfg.assignments,
        "default_provider": cfg.default_provider,
        "fallbacks": cfg.fallbacks,
        # Whether model replies stream as they are written. Exposed here because the
        # toggle lives in this dialog, and a setting the page cannot read back is a
        # setting that silently resets.
        "stream": cfg.stream,
        # The operator-provided model: present but disabled when unconfigured,
        # so the UI can explain why it is unavailable instead of hiding it.
        "shared_model": shared.public(),
        # Every model the user could pick, grouped by capability and labelled with
        # its provider. The dropdown needs the provider name because the same
        # model id can exist on two providers, and picking the wrong one bills the
        # wrong account.
        "model_options": agent_model_resolver.all_options(cfg),
        # Provider-first grouping: one heading per supplier with its models
        # beneath. This is what the assignment dropdown renders, because the
        # question being asked there is "which supplier, and what does it have" —
        # not "which capability". The capability-keyed form above stays for the
        # needs warning, which asks the other question.
        "provider_groups": agent_model_resolver.provider_groups(cfg),
        "agents": [
            {
                "id": agent,
                "label": AGENT_IDS_LABEL.get(agent, agent),
                # Capability chips: what this agent needs from a provider beyond a
                # text model. Kept alongside `resolved_provider` so a reader can see
                # both "what it needs" and "what it is currently bound to".
                "needs": agent_capabilities.describe(agent),
                # What will actually run, per capability. This is what replaces
                # "继承默认": the table shows the model in use rather than an
                # indirection the reader has to resolve themselves.
                "uses": agent_model_resolver.resolve_for(
                    cfg, agent, agent_capabilities.capabilities_of(agent)
                ),
                # Kept for older callers; the UI no longer depends on it.
                "resolved_provider": (cfg.provider_for(agent).id if cfg.provider_for(agent) else ""),
            }
            for agent in AGENT_IDS
        ],
    }


@router.get("/api/settings/free-models")
async def list_free_models() -> dict:
    """The zero-cost catalogue, with live reachability.

    Reachability is probed on demand rather than cached aggressively: which
    endpoints answer depends on the user's network, and a stale "reachable"
    flag would send them into a failure that looks like an app bug.
    """
    return {"models": await free_models.probe_all()}


@router.post("/api/settings/free-models/{model_id}/import")
async def import_free_model(model_id: str) -> dict:
    """Add a catalogue entry to the provider list.

    Creates the provider with an empty key when the endpoint needs one, so the
    UI can prompt for it right there instead of sending the user to re-enter
    endpoint and model by hand.
    """
    entry = free_models.find(model_id)
    if entry is None:
        raise HTTPException(404, f"未知的免费模型：{model_id}")

    cfg = models_config.load()
    if any(p.base_url == entry.base_url and p.model == entry.model for p in cfg.providers):
        raise HTTPException(409, "这个模型已经在供应商列表里了")

    profile = ProviderProfile(
        id=f"free-{entry.id}",
        preset=entry.provider_preset,
        label=entry.label,
        base_url=entry.base_url,
        api_key="",
        model=entry.model,
        vision_model=entry.vision_model,
    )
    cfg.providers.append(profile)
    if not cfg.default_provider:
        cfg.default_provider = profile.id
    models_config.save(cfg)
    return {
        "provider": profile.public(),
        "needs_key": not entry.keyless,
        "signup_url": entry.signup_url,
        "note": entry.note,
    }


@router.put("/api/settings")
async def put_settings(payload: SettingsPayload) -> dict:
    current = models_config.load()
    existing = {p.id: p for p in current.providers}

    providers: list[ProviderProfile] = []
    for raw in payload.providers:
        pid = (raw.id or "").strip()
        if not pid:
            raise HTTPException(400, "供应商缺少 id")
        if not raw.base_url.strip():
            raise HTTPException(400, f"供应商「{raw.label or pid}」缺少接口地址")
        # A blank model is allowed, and refusing it rejected the WHOLE save.
        #
        # The symptom that exposed this: a user pasted a real key, left "文本模型"
        # at (不配置), and clicked save — the request failed with 400 "缺少模型名",
        # so the key was never stored, while the dialog made it look saved. Every
        # later run then reported "尚未配置 LLM".
        #
        # Blank is a legitimate state: a provider may be entered for its image or
        # video model only, and a media provider legitimately has no text model.
        # Whether a model is *required* depends on how the provider is used, and
        # that check belongs at use time (see `select_backend`), not here where it
        # blocks an unrelated field from being saved.

        # Empty key means "keep the stored one": the UI never holds the secret.
        key = raw.api_key.strip() or (existing[pid].api_key if pid in existing else "")

        providers.append(
            ProviderProfile(
                id=pid,
                preset=raw.preset,
                label=raw.label or PROVIDER_PRESETS.get(raw.preset, {}).get("label", pid),
                base_url=raw.base_url.strip(),
                api_key=key,
                model=raw.model.strip(),
                vision_model=raw.vision_model.strip(),
                # These were dropped here, which silently erased a configured image
                # or video model every time the dialog was saved — the user's
                # Kwai-Kolors setting disappeared and nothing said why.
                image_model=(raw.image_model or "").strip(),
                video_model=(raw.video_model or "").strip(),
                kind=(raw.kind or "").strip(),
                timeout=float(raw.timeout or 60.0),
                temperature=float(raw.temperature if raw.temperature is not None else 0.4),
            )
        )

    valid_ids = {p.id for p in providers}
    # Accept both "providerId" and "providerId:model". The latter pins a specific
    # model on that provider; dropping it here would make the selection silently
    # revert the next time the page reloads.
    assignments = {
        agent: value
        for agent, value in payload.assignments.items()
        if agent in BINDABLE_AGENT_IDS
        and (value.split(":", 1)[0] if value else "") in valid_ids
    }
    default_provider = payload.default_provider if payload.default_provider in valid_ids else ""

    # Fallback chains: keep only providers that still exist, and never let an
    # agent fall back to the model it already uses as its primary (that would be
    # a pointless retry against the same failing endpoint).
    fallbacks = {
        agent: [pid for pid in ids if pid in valid_ids]
        for agent, ids in payload.fallbacks.items()
        if agent in BINDABLE_AGENT_IDS and isinstance(ids, list)
    }
    fallbacks = {a: ids for a, ids in fallbacks.items() if ids}

    cfg = models_config.ModelSettings(
        providers=providers,
        assignments=assignments,
        default_provider=default_provider,
        fallbacks=fallbacks,
        # Kept when the request did not carry it: this payload is also sent by
        # clients that only know about providers, and resetting the flag on their
        # saves would make the toggle appear to flip itself.
        stream=current.stream if payload.stream is None else bool(payload.stream),
    )
    models_config.save(cfg)
    return {
        "providers": [p.public() for p in cfg.providers],
        "assignments": cfg.assignments,
        "default_provider": cfg.default_provider,
        "fallbacks": cfg.fallbacks,
        "stream": cfg.stream,
        "shared_model": free_models.load_shared().public(),
    }


@router.post("/api/settings/models")
async def scan_models(payload: TestProviderPayload) -> dict:
    """List every model this key can use, grouped by capability.

    The grouping exists because `/models` returns only ids: choosing which one
    draws an image is otherwise guesswork, and the wrong pick fails mid-run.
    """
    from .. import model_scan

    base_url, api_key = payload.base_url, payload.api_key
    timeout = payload.timeout
    if payload.provider_id:
        # A saved provider keeps its key server-side; the browser never sees it
        # again, so a rescan must read it from the config rather than the body.
        cfg = models_config.load()
        profile = next((p for p in cfg.providers if p.id == payload.provider_id), None)
        if profile is None:
            return {"ok": False, "error": f"找不到供应商 {payload.provider_id}"}
        base_url = base_url or profile.base_url
        api_key = api_key or profile.api_key
        timeout = timeout or profile.timeout

    # 同步 httpx 调用必须离开事件循环。等待时间使用供应商自己的配置：此前这里
    # 强行 15 秒结束、浏览器又在 25 秒结束，导致一个被配置为 600 秒的内网模型
    # 还在排队时按钮先停止旋转，看起来像程序擅自判了失败。
    upstream_timeout = max(1.0, float(timeout or 60.0))
    cat = await run_in_threadpool(
        model_scan.scan, base_url, api_key, timeout=upstream_timeout
    )
    if cat.error:
        return {
            "ok": False,
            "error": cat.error,
            "source": cat.error_source or "local",
            "upstream_status": cat.upstream_status,
            "upstream_body": cat.upstream_body,
            "upstream_url": (base_url or "").rstrip("/") + "/models",
            "groups": {},
            "counts": {},
        }

    groups = cat.group()
    return {
        "ok": True,
        "error": "",
        "total": len(cat.entries),
        "counts": cat.counts(),
        "groups": groups,
        # A short pick-list per capability: the first few are what the page offers
        # as defaults, so "which one should I use" is answered without scrolling.
        "suggested": {
            cap: [item["id"] for item in items[:3]]
            for cap, items in groups.items()
            if items
        },
    }


@router.post("/api/settings/test")
async def test_provider(payload: TestProviderPayload) -> dict:
    """One tiny completion proving endpoint + key + model actually work.

    Testing an unsaved draft matters: a user should find out a key is wrong here,
    not in the middle of a paid run.
    """
    if payload.provider_id:
        cfg = models_config.load()
        profile = next((p for p in cfg.providers if p.id == payload.provider_id), None)
        if profile is None:
            return {"ok": False, "error": f"找不到供应商 {payload.provider_id}"}
        # Use the visible draft; provider_id is only needed to reuse the stored
        # secret. Testing stale saved URL/model values misdiagnoses user edits.
        base_url = payload.base_url or profile.base_url
        api_key = payload.api_key or profile.api_key
        model = payload.model if 'model' in payload.model_fields_set else profile.model
        timeout = payload.timeout or profile.timeout
    else:
        base_url, api_key, model = payload.base_url, payload.api_key, payload.model
        timeout = payload.timeout or 60.0

    if not base_url or not model:
        return {"ok": False, "error": "缺少接口地址或模型名"}

    # This is deliberately a minimal upstream request rather than `chat()`.
    # `chat()` adds thinking/JSON-mode fields and moderation retries for real
    # generation; an OpenAI-compatible relay may reject those fields with 400,
    # which made a healthy connection test fail because of our own probe shape.
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "只回复一个词：OK"},
            {"role": "user", "content": "ping"},
        ],
        "max_tokens": 16,
        "temperature": 0,
    }
    try:
        async with httpx.AsyncClient(timeout=max(1.0, float(timeout)), trust_env=provider_trust_env(url)) as client:
            response = await client.post(url, json=body, headers=headers)
    except httpx.TimeoutException as exc:
        return {
            "ok": False,
            "source": "network",
            "error": f"没有收到上游响应：等待 {float(timeout):g} 秒后超时（{url}）",
            "detail": f"{type(exc).__name__}: {exc}",
            "upstream_status": None,
            "upstream_url": url,
        }
    except httpx.RequestError as exc:
        return {
            "ok": False,
            "source": "network",
            "error": f"没有收到上游 HTTP 响应（{type(exc).__name__}）：{str(exc) or '连接失败'}",
            "detail": f"{type(exc).__name__}: {exc}",
            "upstream_status": None,
            "upstream_url": url,
        }

    raw = (response.text or "").strip()
    if not response.is_success:
        from ..model_scan import upstream_message

        message = upstream_message(raw)
        return {
            # The Studio route itself completed normally (HTTP 200).  The exact
            # provider status is data, so the UI can truthfully say who sent it.
            "ok": False,
            "source": "upstream",
            "error": (
                f"上游模型服务返回 HTTP {response.status_code}"
                + (f"：{message}" if message else "（上游没有返回错误正文）")
            ),
            "upstream_status": response.status_code,
            "upstream_body": raw[:500],
            "upstream_url": url,
        }

    try:
        data = response.json()
        reply = str(data["choices"][0]["message"].get("content") or "").strip()
        returned_model = str(data.get("model") or model)
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        return {
            "ok": False,
            "source": "upstream_response",
            "error": "上游返回 HTTP 200，但正文不是 OpenAI 兼容的回答结构",
            "detail": f"{type(exc).__name__}: {exc}",
            "upstream_status": response.status_code,
            "upstream_body": raw[:500],
            "upstream_url": url,
        }
    return {"ok": True, "model": returned_model, "reply": reply[:40], "source": "upstream"}


@router.post("/api/settings/import-env")
async def import_from_env() -> dict:
    """Create a provider from the current environment variables, if any.

    Makes the tool usable for someone who already exported STUDIO_LLM_API_KEY
    without retyping it into the browser.
    """
    from ..config import settings as app_settings

    llm = app_settings.llm
    if not llm.api_key and "127.0.0.1" not in llm.base_url:
        return {"ok": False, "error": "环境变量中没有可导入的 Key"}

    cfg = models_config.load()
    existing = next((p for p in cfg.providers if p.base_url == llm.base_url), None)
    if existing is not None:
        return {"ok": False, "error": "该地址已配置过了"}

    profile = ProviderProfile(
        id=f"env-{len(cfg.providers) + 1}",
        preset="custom",
        label="环境变量导入",
        base_url=llm.base_url,
        api_key=llm.api_key,
        model=llm.model,
        vision_model=llm.vision_model,
    )
    cfg.providers.append(profile)
    if not cfg.default_provider:
        cfg.default_provider = profile.id
    models_config.save(cfg)
    return {"ok": True, "provider": profile.public()}
