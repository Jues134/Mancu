"""Fetch and classify a provider's model catalogue.

`scan(base_url, api_key)` performs the one network call the settings page needs:
`GET {base_url}/models`, then hands the ids to `model_catalog` for grouping.

Why one function rather than a route-local implementation:

  · The route needs it, and a test needs it without HTTP — a single callable
    keeps one implementation instead of a route and a test that drift apart.
  · Failure handling is one place: a key that cannot list its models must produce
    a readable reason, not an empty group that looks like "no models exist".
"""

from __future__ import annotations

import json

import httpx

from . import model_catalog
from .tools.provider_network import provider_trust_env


def upstream_message(body: str) -> str:
    """Keep the provider's own words, but unwrap the common JSON envelope."""
    text = (body or "").strip()
    if not text:
        return ""
    try:
        data = json.loads(text)
        error = data.get("error") if isinstance(data, dict) else None
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])[:500]
        if isinstance(data, dict):
            for key in ("message", "detail", "msg"):
                if data.get(key):
                    return str(data[key])[:500]
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass
    return text[:500]


def scan(
    base_url: str,
    api_key: str,
    *,
    timeout: float = 30.0,
) -> model_catalog.Catalogue:
    """List and classify every model this key can use."""
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return model_catalog.Catalogue(error="缺少接口地址", error_source="local")
    if not api_key:
        return model_catalog.Catalogue(error="缺少 API Key", error_source="local")

    url = f"{base}/models"
    try:
        resp = httpx.get(
            url, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout,
            trust_env=provider_trust_env(url),
        )
    except httpx.RequestError as exc:
        return model_catalog.Catalogue(
            error=f"没有收到上游 HTTP 响应（{type(exc).__name__}）：{str(exc)[:160]}",
            error_source="network",
        )

    if resp.status_code != 200:
        body = (resp.text or "").strip()[:500]
        message = upstream_message(body)
        return model_catalog.Catalogue(
            error=(
                f"上游模型服务返回 HTTP {resp.status_code}"
                + (f"：{message}" if message else "（上游没有返回错误正文）")
            ),
            error_source="upstream",
            upstream_status=resp.status_code,
            upstream_body=body,
        )

    try:
        body = resp.json()
    except Exception:  # noqa: BLE001
        return model_catalog.Catalogue(
            error="上游模型列表返回的不是 JSON",
            error_source="upstream_response",
            upstream_status=resp.status_code,
            upstream_body=(resp.text or "")[:500],
        )

    # OpenAI-compatible providers use `data`; a few use `models`.
    items = body.get("data") or body.get("models") or []
    if not isinstance(items, list) or not items:
        return model_catalog.Catalogue(
            error=f"上游模型列表为空：{str(body)[:160]}",
            error_source="upstream_response",
            upstream_status=resp.status_code,
            upstream_body=str(body)[:500],
        )

    ids: list[str] = []
    owned: dict[str, str] = {}
    for item in items:
        if isinstance(item, dict):
            mid = str(item.get("id") or item.get("model") or "").strip()
            if mid:
                ids.append(mid)
                owned[mid] = str(item.get("owned_by") or "")
        elif isinstance(item, str):
            ids.append(item.strip())

    cat = model_catalog.build(ids, owned)
    if not cat.entries:
        return model_catalog.Catalogue(
            error="上游返回中没有可用的模型 id",
            error_source="upstream_response",
            upstream_status=resp.status_code,
            upstream_body=str(body)[:500],
        )
    return cat
