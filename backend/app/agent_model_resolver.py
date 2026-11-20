"""Resolve what each agent is actually going to use, per capability.

Requirement: the assignment table must show the model in use — "导演 → 智谱 ·
glm-5", "角色设定 → 硅基流动 · Tongyi-MAI/Z-Image-Turbo" — rather than
"继承默认". And every option in the dropdown must name its provider.

Why this needs a resolver rather than reading `assignments`:

`assignments` only records *which provider* an agent is pinned to. The model
depends on which capability is being asked about:

  · a text-thinking agent uses `provider.model`
  · consistency and video generation use a text model AND a media model — the media model
    comes from whichever provider declares `image_model` / `video_model`, which
    is often a *different* provider

So a single "resolved provider" per agent cannot express the truth. This returns
one entry per capability the agent needs, each naming provider, model and how it
was chosen.
"""

from __future__ import annotations

from typing import Any

from . import model_catalog
from .models_config import ModelSettings, ProviderProfile


#: Which provider field holds the model for each capability.
FIELD_OF = {
    model_catalog.CAP_TEXT: "model",
    model_catalog.CAP_VISION: "vision_model",
    model_catalog.CAP_IMAGE: "image_model",
    model_catalog.CAP_VIDEO: "video_model",
}

CAP_LABELS = {
    model_catalog.CAP_TEXT: "文本",
    model_catalog.CAP_VISION: "视觉",
    model_catalog.CAP_IMAGE: "生图",
    model_catalog.CAP_VIDEO: "生视频",
}


def _model_of(profile: ProviderProfile, capability: str) -> str:
    return str(getattr(profile, FIELD_OF.get(capability, "model"), "") or "")


def resolve_for(
    cfg: ModelSettings, agent_id: str, capabilities: list[str]
) -> list[dict[str, Any]]:
    """Resolve the model serving each capability this agent needs.

    Returns one dict per capability:
      {capability, label, provider_id, provider_label, model, source}

    `source` says how it was chosen, so the UI can distinguish a deliberate
    assignment from a fallback the user never made:
      assigned   — the agent is pinned to this provider
      default    — inherited from the default provider
      capability — chosen because it is the only provider offering this media model
      missing    — nothing configured; the agent cannot do this
    """
    text_profile = cfg.provider_for(agent_id)
    # The single `providerId:model` field may name a *media* model (that is how a
    # generation gateway gets pinned to an agent). It must not be shown as the
    # text model of whichever provider `provider_for` fell back to — that rendered
    # "智谱 · comfyui" beside a run that actually called glm-5.
    assigned_id = (cfg.assignments.get(agent_id) or "").split(":", 1)[0]
    out: list[dict[str, Any]] = []

    for cap in capabilities:
        if cap == model_catalog.CAP_TEXT:
            profile = text_profile
            source = (
                "assigned" if cfg.assignments.get(agent_id)
                else "default" if cfg.default_provider
                else "only" if len(cfg.providers) == 1
                else "missing"
            )
            # An explicit pin wins over the provider's default model, else the
            # dropdown selection would have no effect. Only when the pin names the
            # provider actually in use.
            pinned = cfg.pinned_model(agent_id, cap) if assigned_id == (profile.id if profile else "") else ""
            model = pinned or (_model_of(profile, cap) if profile else "")
            if pinned:
                source = "pinned"
        else:
            # A media capability is served by whichever provider declares the
            # matching model. Pinning the agent to a text provider must not
            # override that: character needs Kolors even when it "thinks" with
            # glm-5.
            pinned = cfg.pinned_model(agent_id, cap)
            candidates = [
                p for p in cfg.providers if _model_of(p, cap)
            ]
            # Prefer the assigned/default provider when it happens to offer the
            # capability; otherwise take the first that does.
            preferred = next(
                (p for p in candidates if p is text_profile), None
            )
            profile = preferred or (candidates[0] if candidates else None)
            source = (
                "assigned" if preferred is not None
                else "capability" if profile is not None
                else "missing"
            )
            # Same rule as text: a pin only names the model when it names *this*
            # provider, otherwise "gateway:comfyui" would be printed as the video
            # model of whichever provider happened to be picked.
            if assigned_id != (profile.id if profile else ""):
                pinned = ""
            model = pinned or (_model_of(profile, cap) if profile else "")
            if pinned:
                source = "pinned"

        out.append(
            {
                "capability": cap,
                "label": CAP_LABELS.get(cap, cap),
                "provider_id": profile.id if profile else "",
                "provider_label": (profile.label or profile.id) if profile else "",
                "model": model,
                "source": source,
            }
        )

    # Every agent thinks with a text model, so a text entry is always produced —
    # even when the agent's declared capabilities are only media (character needs
    # an image model AND a text model to describe the person). Returning media
    # only made the table show "导演 · 视觉" with no model for its reasoning.
    if not any(u["capability"] == model_catalog.CAP_TEXT for u in out):
        profile = text_profile
        pinned = cfg.pinned_model(agent_id, model_catalog.CAP_TEXT)
        out.insert(
            0,
            {
                "capability": model_catalog.CAP_TEXT,
                "label": CAP_LABELS[model_catalog.CAP_TEXT],
                "provider_id": profile.id if profile else "",
                "provider_label": (profile.label or profile.id) if profile else "",
                "model": pinned or (_model_of(profile, model_catalog.CAP_TEXT) if profile else ""),
                "source": (
                    "pinned" if pinned
                    else "assigned" if cfg.assignments.get(agent_id)
                    else "default"
                ),
            },
        )
    return out


def provider_groups(cfg: ModelSettings) -> list[dict[str, Any]]:
    """Every provider with the models it declares, for a provider-first dropdown.

    Shape:
      [
        {
          "provider_id": "zhipu",
          "provider_label": "智谱 GLM",
          "models": [
            {"value": "zhipu:glm-5", "model": "glm-5",
             "capability": "text", "label": "文本"},
            …
          ],
        },
        …
      ]

    Order follows the configured provider list, so the first provider the user
    added appears first — the same order the supplier cards use above, which keeps
    the two sections consistent.
    """
    groups: list[dict[str, Any]] = []
    for profile in cfg.providers:
        models: list[dict[str, str]] = []
        seen: set[str] = set()
        for cap in CAP_LABELS:
            model = _model_of(profile, cap)
            if not model:
                continue
            # A single model id could be declared for two capabilities; list it
            # once with the first capability it matched rather than duplicating.
            key = f"{cap}:{model}"
            if key in seen:
                continue
            seen.add(key)
            models.append(
                {
                    # Pure `providerId:model`, which the resolver already parses.
                    # A `|capability` suffix corrupted the model name on
                    # resolution — the capability travels separately in `label`.
                    "value": f"{profile.id}:{model}",
                    "model": model,
                    "capability": cap,
                    "label": CAP_LABELS.get(cap, cap),
                }
            )
        if models:
            groups.append(
                {
                    "provider_id": profile.id,
                    "provider_label": profile.label or profile.id,
                    "models": models,
                }
            )
    return groups


def all_options(cfg: ModelSettings) -> dict[str, list[dict[str, str]]]:
    """Every model the user could pick, grouped by capability, with its provider.

    The dropdown needs the provider name on each entry — the same model id can
    exist on two providers, and picking the wrong one sends the request to the
    wrong account.
    """
    out: dict[str, list[dict[str, str]]] = {cap: [] for cap in CAP_LABELS}
    for profile in cfg.providers:
        for cap in CAP_LABELS:
            model = _model_of(profile, cap)
            if not model:
                continue
            out.setdefault(cap, []).append(
                {
                    "value": f"{profile.id}:{model}",
                    "provider_id": profile.id,
                    "provider_label": profile.label or profile.id,
                    "model": model,
                    "display": f"{profile.label or profile.id} · {model}",
                }
            )
    return out
