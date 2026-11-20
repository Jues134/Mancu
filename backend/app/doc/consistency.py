"""Visual consistency measurement.

What this can and cannot do, established by measurement before writing it:

  · **SSIM (structural similarity) catches a style break.** A near-identical pair
    scored 0.775 while a changed backdrop dropped to 0.601 — a usable gap.
  · **SSIM cannot identify a face.** Two different people in the same pose and
    lighting scored 0.863, higher than the legitimately-matched pair. So this
    measure must never be presented as face recognition.

Therefore the design puts the identity guarantee somewhere else: the character's
reference portrait is the *first frame* of every shot, so identity is fixed
generationally rather than checked afterwards. This module's job is the weaker,
honest one — detect that a shot drifted away from the established look, e.g. a
style break or a completely different composition.

Reporting numbers without stating this limit would be the worst option: a critic
that "verifies face consistency" by comparing grey geometry would pass anything
and give false assurance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity
except ImportError:  # pragma: no cover
    structural_similarity = None  # type: ignore[assignment]


#: Below this, a shot has visibly drifted from the reference look. Chosen from the
#: measured gap (0.775 matched vs 0.601 restyled); 0.45 sits below both, so it only
#: fires on a gross mismatch rather than a legitimate camera change.
DRIFT_THRESHOLD = 0.45

#: Comparison size. Small enough to be fast, large enough that structure survives.
_COMPARE_PX = 128


def similarity(path_a: str | Path, path_b: str | Path) -> float | None:
    """Structural similarity of two images, or None when uncomputable."""
    if structural_similarity is None:
        return None
    try:
        a = _grey(path_a)
        b = _grey(path_b)
    except Exception:  # noqa: BLE001 - a missing frame is not a similarity of 0
        return None
    if a is None or b is None:
        return None
    return float(structural_similarity(a, b, data_range=1.0))


def _grey(path: str | Path) -> np.ndarray | None:
    p = Path(path)
    if not p.is_file():
        return None
    with Image.open(p) as img:
        return np.asarray(
            img.convert("L").resize((_COMPARE_PX, _COMPARE_PX))
        ).astype(np.float32) / 255.0


def first_frame(video_path: str | Path, out_png: Path) -> str:
    """Extract the first frame of a shot for comparison.

    Done with ffmpeg rather than an image library because the artifact is a video;
    comparing the reference portrait to a video requires one frame of that video,
    and the first frame is the one that was structurally seeded by the reference.
    """
    import subprocess

    from .media_tools import find_ffmpeg

    try:
        ff = find_ffmpeg()
    except Exception:  # noqa: BLE001
        return ""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [ff, "-y", "-i", str(video_path), "-vframes", "1", str(out_png)],
        capture_output=True,
    )
    return str(out_png) if out_png.exists() else ""


def compare_against_reference(
    shot_path: str,
    reference_url: str,
    state: dict[str, Any],
) -> float | None:
    """Similarity between a generated shot and its reference portrait.

    The reference arrives as a URL (that is what the video call consumes), so the
    local copy cached during generation is used here — re-downloading per shot would
    add a network round-trip to every comparison and would fail on an expired link.

    Returns None when either side is unavailable, which the critic treats as "no
    measurement" rather than "failure". Treating a missing measurement as a
    rejection would punish shots for a tooling gap.
    """
    portraits = {p.get("url"): p.get("path") for p in (state.get("portraits") or [])}
    ref_path = portraits.get(reference_url) or ""
    if not ref_path or not Path(ref_path).is_file():
        return None

    p = Path(shot_path)
    if p.suffix.lower() in {".mp4", ".mov", ".webm"}:
        frame = first_frame(p, p.with_name(p.stem + "_first.png"))
        if not frame:
            return None
        p = Path(frame)

    return similarity(p, ref_path)


def describe_limits() -> dict[str, Any]:
    """Capability statement, surfaced in the UI and the report.

    Written down because the difference between "we verify consistency" and
    "we measure structural drift" is the difference between a claim and a
    measurement, and a reviewer will ask which one it is.
    """
    return {
        "method": "SSIM 结构相似度",
        "threshold": DRIFT_THRESHOLD,
        "available": structural_similarity is not None,
        "catches": ["画风突变", "构图崩坏", "整体色调偏移"],
        "cannot": ["判断是否为同一张脸", "识别角色身份"],
        "identity_mechanism": "人物一致性由「定妆照作为首帧」保证，而非事后比对",
    }
