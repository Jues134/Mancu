"""Registration for the active six-agent document pipeline."""

from __future__ import annotations

SIX = "six"
DEFAULT = SIX


def selected() -> str:
    """Return the only supported roster."""
    return SIX


def ensure_registered() -> str:
    """Register the active roster once and return its identifier."""
    from ..platform.registry import registry
    from .agents_six import register_six, registered_ids

    wanted = set(registered_ids())
    have = {agent["id"] for agent in registry.catalog()}
    if not wanted <= have:
        register_six()
    return SIX
