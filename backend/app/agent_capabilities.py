"""Declare which capability each agent needs, and expose it to the settings page.

The requirement: the chip next to an agent should say what that agent needs —
"需视觉模型" for the director, but "需生图模型" for consistency and "需生视频模型"
for video_gen.

Why a declaration table rather than a flag per agent:

  · It is a single readable statement of the pipeline's capability needs, so a
    reader can see at a glance which stages cost image/video calls.
  · The settings page derives its chips from it, so adding a capability to an
    agent does not require editing both the backend payload and the frontend.
  · It separates *what the agent needs* (media capability) from *which text model
    it thinks with* (the assignment dropdown). Those are different axes, and the
    UI previously conflated them: every agent showed only "需视觉".

Grouping below matches the six-agent roster in `app.doc.agents_six`:

  director        text model for analysis; inspects source material images
  script_adaptor  text only
  storyboard      text only (writes prompts)
  consistency     text model to describe assets + image model to render the
                  four-view character sheets and scene/prop references
  video_gen       text model to adapt a shot prompt + video model to render it
  quality_review  text model; needs vision to judge a rendered frame

"""

from __future__ import annotations

#: Capability keys the settings page renders as chips.
CAP_VISION = "vision"
CAP_IMAGE = "image"
CAP_VIDEO = "video"

#: Chinese labels, so the chip text lives next to the decision rather than in the
#: frontend, where it would drift.
CAP_LABELS = {
    CAP_VISION: "需视觉模型",
    CAP_IMAGE: "需生图模型",
    CAP_VIDEO: "需生视频模型",
}

#: agent id → the capabilities that agent requires.
AGENT_CAPABILITIES: dict[str, list[str]] = {
    "director": [CAP_VISION],
    "script_adaptor": [],
    "storyboard": [],
    "consistency": [CAP_IMAGE],
    "video_gen": [CAP_VIDEO],
    "quality_review": [CAP_VISION],
}


def capabilities_of(agent_id: str) -> list[str]:
    return AGENT_CAPABILITIES.get(agent_id, [])


def describe(agent_id: str) -> list[dict[str, str]]:
    """Chips for one agent: [{"key": "image", "label": "需生图模型"}, …]."""
    return [
        {"key": cap, "label": CAP_LABELS.get(cap, cap)}
        for cap in capabilities_of(agent_id)
    ]
