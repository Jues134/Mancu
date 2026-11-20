"""Frame chaining between shots: the previous clip's tail becomes the next head.

Why this module exists: each shot is rendered as an independent video, so two
adjacent shots share nothing — different courtyard, different light, different
face. The storyboard now declares HOW shots connect (`link_type` +
`first_frame_strategy`); this module turns that declaration into a concrete file.

The mechanism is deliberately backend-agnostic. The previous clip is a local mp4;
its last frame is extracted ONCE with ffmpeg into a PNG, and that PNG is handed
over through the one parameter every backend already understands:
`render_video(reference_image=...)`.

  · local gateway (fl2va)    → the PNG is uploaded and sent as `first_frame`,
    which anchors the shot to continue from exactly that picture.
  · hosted i2v (即梦 SD2.5 等) → the same PNG is uploaded to the provider and
    sent as `image_url`, whose semantics on that family of endpoints is also
    "the first frame of the shot".

No backend-specific branch lives here, on purpose: adding a third provider means
implementing `render_video`, not re-learning this chain.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

#: The five ways the storyboard says a shot relates to its predecessor.
CONTINUOUS_ACTION = "CONTINUOUS_ACTION"
SAME_SCENE_CUT = "SAME_SCENE_CUT"
SCENE_TRANSITION = "SCENE_TRANSITION"
TIME_JUMP = "TIME_JUMP"
MONTAGE = "MONTAGE"

#: Shots that may inherit the previous clip's tail frame. A scene change must
#: NOT: the tail frame would drag the old location into the new shot.
CHAINABLE = {CONTINUOUS_ACTION, SAME_SCENE_CUT}

#: What the strategy names mean, for records and the UI.
KIND_LABELS = {
    "prev_last_frame": "上一镜尾帧",
    "scene_reference": "场景参考图",
    "character_sheet": "人物设定稿",
    "none": "无首帧",
}


def _is_valid_image(path: Path) -> bool:
    """A PNG ffmpeg can write is not always a picture.

    A corrupt source can produce a non-empty file that no decoder accepts, and
    handing that to the next render would anchor the shot to garbage. PIL is the
    judge because it is the same decoder the consistency stage already trusts.
    """
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.width > 0 and img.height > 0
    except Exception:  # noqa: BLE001 - unreadable is unreadable, whatever broke
        return False


def extract_last_frame(video: str | Path, out_png: Path) -> str:
    """Write the last frame of `video` to `out_png`; "" when it cannot be done.

    `-sseof` seeks from the end, then one frame is grabbed. For a 4-second clip
    seeking back 0.3s lands on (or one frame before) the final picture — close
    enough that the next shot continues where the previous one ended, which is
    the requirement; bit-exactness is not.

    Failing here is not fatal to the run: the caller falls back to the scene
    reference rather than losing the shot.
    """
    from .media_tools import find_ffmpeg

    video = Path(video)
    if not video.is_file():
        return ""
    try:
        ff = find_ffmpeg()
    except Exception:  # noqa: BLE001 - no ffmpeg is a degradation, not a crash
        return ""
    out_png.parent.mkdir(parents=True, exist_ok=True)

    def run(args: list[str]) -> bool:
        try:
            proc = subprocess.run([ff, "-y", *args], capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return proc.returncode == 0 and _is_valid_image(out_png)

    # A very short clip can seek past its only frame; the second attempt starts
    # further back and lets ffmpeg overwrite with the last frame it finds.
    ok = run(["-sseof", "-0.3", "-i", str(video),
              "-frames:v", "1", "-q:v", "2", str(out_png)]) or \
        run(["-sseof", "-1.2", "-i", str(video),
             "-update", "1", "-q:v", "2", str(out_png)])
    return str(out_png) if ok else ""


def first_frame_for(
    shot: dict[str, Any],
    *,
    prev_clip: str = "",
    scene_image: str = "",
    character_image: str = "",
    out_dir: Path | None = None,
    supports_first_frame: bool = True,
) -> tuple[str, str]:
    """Decide this shot's first frame. Returns (local_path, kind).

    Priority, and why in this order:

      1. The storyboard's own strategy, when it asks for the previous tail —
         but only while the link is chainable and the previous clip exists.
      2. The scene reference image, which every non-chainable link and every
         first shot falls to: it fixes the place, so the shot is at least in
         the right room.
      3. The character sheet — the historical fallback, better than nothing.
      4. Nothing, when neither the storyboard nor the assets offer a picture.

    A backend that declares no first-frame support gets "" rather than a file it
    would silently drop: an honest "no anchor" beats a chain the model ignores.
    """
    if not supports_first_frame:
        return "", "none"

    strategy = str(shot.get("first_frame_strategy") or "").strip()
    link = str(shot.get("link_type") or "").strip()

    if strategy == "none":
        return "", "none"

    wants_chain = strategy == "prev_last_frame" or (
        not strategy and link in CHAINABLE
    )
    if wants_chain and link in CHAINABLE and prev_clip and out_dir is not None:
        shot_id = str(shot.get("shot_id") or "shot")
        png = out_dir / f"{shot_id}-first.png"
        frame = extract_last_frame(prev_clip, png)
        if frame:
            return frame, "prev_last_frame"
        # extraction failed → fall through to the scene reference

    if scene_image and Path(scene_image).is_file():
        return str(scene_image), "scene_reference"
    if character_image and Path(character_image).is_file():
        return str(character_image), "character_sheet"
    return "", "none"


def kind_label(kind: str) -> str:
    """Chinese name for a first-frame kind, for traces and the conversation."""
    return KIND_LABELS.get(kind, kind)

def effective_link(declared: str, scene_id: str = "", prev_scene: str = "") -> str:
    """The link this render should honour, given the shot's actual scene.

    The storyboard's `link_type` is a declaration, and a model that forgets to
    emit one gets the chainable default. That default is right for most shots and
    catastrophic for the one that cuts to a new location: it would anchor the new
    scene to the old room's last frame. A different `scene_id` is observed fact,
    so it overrides a claim of continuity.

    A declared SCENE_TRANSITION / TIME_JUMP / MONTAGE is never downgraded: those
    are the storyboard's deliberate decisions even when the scene id repeats (a
    time jump inside one room, say).
    """
    link = str(declared or "")
    if scene_id and prev_scene and scene_id != prev_scene:
        if link in ("", CONTINUOUS_ACTION, SAME_SCENE_CUT):
            return SCENE_TRANSITION
    return link or CONTINUOUS_ACTION
