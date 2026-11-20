"""Persistent user-facing enable/disable switches for MCP services."""

from __future__ import annotations

import json
from pathlib import Path

from ..config import CONFIG_DIR


_PATH = Path(CONFIG_DIR) / "mcp_tools.json"
_DEFAULTS = {"knowledge": True, "feishu": True}


def load() -> dict[str, bool]:
    data: dict = {}
    if _PATH.is_file():
        try:
            value = json.loads(_PATH.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                data = value
        except (OSError, ValueError):
            data = {}
    return {
        key: data[key] if isinstance(data.get(key), bool) else default
        for key, default in _DEFAULTS.items()
    }


def enabled(service: str) -> bool:
    if service not in _DEFAULTS:
        return False
    return load()[service]


def set_enabled(service: str, value: bool) -> dict[str, bool]:
    if service not in _DEFAULTS:
        raise KeyError(f"未知 MCP 服务：{service}")
    state = load()
    state[service] = bool(value)
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = _PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(_PATH)
    return state
