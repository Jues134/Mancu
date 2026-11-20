"""Internal narration and final-assembly steps for the active pipeline."""

from __future__ import annotations

import re
from typing import Any

from ..agents.base import AgentContext
from ..platform.registry import AgentDefinition
from . import media_mode
from . import schemas as s


def _select(
    mapping: dict[str, Any], segments: list[dict[str, Any]], index: int
) -> dict[str, Any]:
    if not mapping:
        return {}
    if 0 <= index < len(segments):
        segment_id = str(segments[index].get("id") or "")
        if segment_id and segment_id in mapping:
            return mapping[segment_id]
    return next(iter(mapping.values()))


def _assets_dir(ctx: AgentContext, kind: str):
    from ..config import DATA_DIR

    return DATA_DIR / "assets" / ctx.run_id / kind


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", name)[:40] or "item"


def _generator(ctx: AgentContext, capability: str = "") -> Any:
    """Resolve the configured media backend without bypassing spend switches."""
    needed = {
        "image": "generate_images",
        "video": "generate_videos",
    }.get(capability)

    if media_mode.prompt_only():
        ctx.state["__media_skip"] = "只生成提示词（提示词模式）"
        return None
    if needed and not ctx.state.get(needed, False):
        ctx.state["__media_skip"] = "只生成提示词（未勾选生成开关）"
        return None

    # Test harnesses and resumed in-memory runs may inject the old single backend.
    legacy = ctx.state.get("__generator")
    if legacy is not None:
        return legacy

    cache = ctx.scratch.setdefault("media_generators", {})
    cached = cache.get(capability or "any")
    if cached is not None:
        return cached

    from .media_backend import select_backend

    agent_id = {"image": "consistency", "video": "video_gen"}.get(capability, "")
    selection = select_backend(
        ctx.model_settings, capability=capability, agent_id=agent_id
    )
    ctx.state["__backend_selection"] = selection.to_dict()
    if not selection.usable:
        ctx.state["__media_skip"] = selection.reason or "未配置可用的生成模型"
        return None

    generator = selection.backend
    capabilities = selection.capabilities or {}
    if capability == "image" and not capabilities.get(
        "image", bool(getattr(generator, "image_model", ""))
    ):
        ctx.state["__media_skip"] = f"{selection.name} 未配置可用的生图模型"
        return None
    if capability == "video" and not capabilities.get(
        "video", bool(getattr(generator, "video_model", ""))
    ):
        ctx.state["__media_skip"] = (
            f"{selection.name} 未配置可用的生视频模型；"
            "可在设置中配置视频模型，或改用本地 ComfyUI 网关"
        )
        return None

    cache[capability or "any"] = generator
    ctx.state.pop("__media_skip", None)
    return generator


async def synthesize_narration(
    ctx: AgentContext,
    definition: AgentDefinition,
    options: dict[str, Any],
) -> dict[str, Any]:
    """Create per-shot narration clips when a local voice is required."""
    del definition
    boards = ctx.state.get("storyboards") or {}
    segments = ctx.state.get("segments") or []
    index = int(options.get("segment_index") or 0)
    board = _select(boards, segments, index)
    shots = board.get("shots") or []

    from .tts import synthesize

    out_dir = _assets_dir(ctx, "voice")
    clips: list[dict[str, Any]] = []
    for shot in shots[:12]:
        text = str(shot.get("narration") or "").strip()
        if not text:
            continue
        clip = s.VoiceClip(shot_id=str(shot.get("id")), text=text[:200])
        try:
            result = await synthesize(
                text, out_dir / f"{_safe_name(str(shot.get('id')))}.mp3"
            )
            clip.path = result.path
            clip.seconds = result.seconds
            clip.voice = result.voice
            clip.ok = result.ok
            if not result.ok:
                clip.error = result.error
        except Exception as exc:  # noqa: BLE001 - narration is optional
            clip.error = str(exc)[:160]
        clips.append(clip.model_dump())

    voiced = [clip for clip in clips if clip.get("ok")]
    return {
        "voice_clips": clips,
        "voiced_ok": len(voiced),
        "voice_summary": (
            f"配音 {len(voiced)}/{len(clips)} 段"
            + ("" if voiced else "（TTS 不可用，将输出无声字幕版）")
        ),
    }


async def assemble_film(
    ctx: AgentContext,
    definition: AgentDefinition,
    options: dict[str, Any],
) -> dict[str, Any]:
    """Assemble rendered shots, subtitles and narration into the final film."""
    del definition, options
    from .compose_film import build_film

    boards = ctx.state.get("storyboards") or {}
    board = _select(boards, ctx.state.get("segments") or [], 0)
    shots = board.get("shots") or []
    assets = ctx.state.get("shot_assets") or []
    clips = ctx.state.get("voice_clips") or []

    if media_mode.prompt_only():
        seconds = sum(float(shot.get("seconds") or 0) for shot in shots)
        cut = s.FinalCut(
            path="",
            shot_count=len(shots),
            seconds=seconds,
            has_audio=bool(clips),
            note=f"提示词模式：计划合成 {len(shots)} 个镜头 / {seconds:.0f} 秒（未渲染）",
        )
        return {
            "final_cut": cut.model_dump(),
            "video_path": "",
            "has_video": False,
            "compositor_summary": cut.note,
        }

    from ..config import DATA_DIR

    out = DATA_DIR / "videos" / f"{ctx.run_id}.mp4"
    cut = await build_film(
        shots=shots,
        assets=assets,
        voice_clips=clips,
        out_path=out,
    )
    return {
        "final_cut": cut.model_dump(),
        "video_path": cut.path,
        "has_video": bool(cut.path),
        "compositor_summary": (
            cut.note or f"合成 {cut.shot_count} 个镜头，{cut.seconds:.0f} 秒"
        ),
    }
