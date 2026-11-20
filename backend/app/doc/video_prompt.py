"""Rewrite shot prompts into MiniMax-H3's structured form.

Why a separate rewrite step instead of writing the final prompt in the storyboard:

The storyboard is written before the target video model is known. Its job is to
describe *what the audience sees* — a static composition. MiniMax-H3 needs
something different: duration, aspect ratio, a subject action, camera commands
from a fixed vocabulary, and a sound hint. Conflating the two would mean either the
storyboard speaks in model-specific syntax (unusable if the model changes) or the
video model receives a static description and produces a nearly frozen clip.

So the pipeline keeps both:

  · `visual_description` — the storyboard's own wording, for humans to read
  · `video_prompt`       — the rewritten form, for the model to consume

Everything below comes from measured documentation, not invention:

  · Structure
      "15s, 16:9. Live-action footage of … Core story: … Tone: … " — every
      official H3 example opens with duration and aspect ratio, then a core story,
      then tone/edit notes.
  · Camera commands
      The 15 documented brackets — `[Push in]`, `[Pan left]`, `[Truck left]`,
      `[Pedestal up]`, `[Tilt up]`, `[Zoom in]`, `[Shake]`, `[Tracking shot]`,
      `[Static shot]`, and their opposites. Multiple commands may share one
      bracket (`[Pan left,Pedestal up]`, max 3); sequential ones read
      `…[Push in], then…[Pull out]`.
  · Limits
      4–15 seconds, integer; prompt ≤ 7000 characters. A 5-second default sits
      inside the range.

The rewrite uses one model call per segment rather than per shot: twelve shots
would otherwise mean twelve round trips, and the segment's shots must share a
tone, which a single call naturally keeps consistent.
"""

from __future__ import annotations

from typing import Any

#: The documented camera vocabulary, mapped from this project's `camera_move`
#: values to the bracket syntax MiniMax scores as "more accurate".
CAMERA_BRACKETS: dict[str, str] = {
    "static": "[Static shot]",
    "push_in": "[Push in]",
    "pull_out": "[Pull out]",
    "pan_left": "[Pan left]",
    "pan_right": "[Pan right]",
    "truck_left": "[Truck left]",
    "truck_right": "[Truck right]",
    "pedestal_up": "[Pedestal up]",
    "pedestal_down": "[Pedestal down]",
    "tilt_up": "[Tilt up]",
    "tilt_down": "[Tilt down]",
    "zoom_in": "[Zoom in]",
    "zoom_out": "[Zoom out]",
    "shake": "[Shake]",
    "tracking": "[Tracking shot]",
    # Aliases the storyboard model tends to write instead of the canonical key.
    "handheld": "[Shake]",
    "follow": "[Tracking shot]",
    "dolly_in": "[Push in]",
    "dolly_out": "[Pull out]",
    "pan": "[Pan left]",
    "tilt": "[Tilt up]",
    "zoom": "[Zoom in]",
}

#: Supported H3 durations, in seconds. Outside this range the API rejects the
#: request, so a shot duration is clamped rather than sent and failed.
MIN_SECONDS = 4
MAX_SECONDS = 15

#: H3's hard limit on prompt length.
MAX_PROMPT_CHARS = 7000


def bracket_for(camera_move: str) -> str:
    """Map this project's camera value to H3's bracket command."""
    key = (camera_move or "").strip().lower().replace(" ", "_")
    return CAMERA_BRACKETS.get(key, "[Static shot]")


def clamp_seconds(seconds: Any) -> int:
    """H3 accepts integers in 4–15 only."""
    try:
        value = int(round(float(seconds or 5)))
    except (TypeError, ValueError):
        value = 5
    return max(MIN_SECONDS, min(MAX_SECONDS, value))


def build_rewrite_prompt(
    *,
    shots: list[dict[str, Any]],
    era: str = "",
    style: str = "",
    aspect: str = "16:9",
    banned: list[str] | None = None,
) -> tuple[str, str]:
    """Compose the instruction that asks a model to rewrite the shot prompts.

    Returns (system, user). The system message encodes the target structure and
    the camera vocabulary so the model does not invent its own syntax; the user
    message lists the shots with everything the rewrite needs.
    """
    # The system prompt lives in app/doc/prompts/visual.txt, next to the other
    # agent prompts. It carries one dynamic value - the aspect ratio - which is
    # substituted by hand rather than with str.format: the file also contains
    # literal JSON braces, and format() would treat those as fields and raise.
    from .prompts import load_prompt

    system = load_prompt("visual").replace("{aspect}", aspect)

    lines: list[str] = []
    if era:
        lines.append(f"时代背景：{era}")
    if style:
        lines.append(f"全片画风：{style}")
    if banned:
        lines.append("禁止出现：" + "、".join(banned[:8]))
    lines.append("")
    lines.append("需要改写的镜头：")
    for shot in shots:
        lines.append(
            f"- id={shot.get('id')}｜景别={shot.get('shot_type') or 'medium'}"
            f"｜时长={clamp_seconds(shot.get('seconds'))}s"
            f"｜运镜={bracket_for(shot.get('camera_move'))}"
        )
        lines.append(f"  画面：{shot.get('visual_description') or ''}")
        if shot.get("narration"):
            lines.append(f"  旁白：{str(shot.get('narration'))[:80]}")
        if shot.get("characters"):
            lines.append(f"  人物：{'、'.join(shot['characters'])}")

    user = "\n".join(lines)
    return system, user


def fallback_prompt(shot: dict[str, Any], *, aspect: str = "16:9") -> str:
    """Assemble a compliant prompt without a model call.

    Used when the rewrite fails or no text model is configured. It follows the
    same structure so downstream code sees one shape either way — a film that
    silently drops the structure would look like a model failure later.
    """
    seconds = clamp_seconds(shot.get("seconds"))
    camera = bracket_for(shot.get("camera_move"))
    parts = [
        f"{seconds}s, {aspect}.",
        str(shot.get("visual_description") or "").strip(),
    ]
    if shot.get("shot_type"):
        parts.append(f"景别：{shot['shot_type']}。")
    parts.append(f"镜头：{camera}")
    prompt = " ".join(p for p in parts if p and p != "。")
    return prompt[:MAX_PROMPT_CHARS]


def validate(prompt: str) -> str:
    """Trim and length-check one rewritten prompt."""
    text = (prompt or "").strip()
    if len(text) > MAX_PROMPT_CHARS:
        # Cut at the limit but keep the trailing bracket command, which is the
        # part that controls the camera — losing it silently changes the shot.
        tail = text[-80:]
        text = text[: MAX_PROMPT_CHARS - 90] + "…" + tail
    return text
