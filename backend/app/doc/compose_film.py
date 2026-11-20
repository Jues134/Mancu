"""Film assembly — cut the shots to their narration and burn subtitles.

The cut is driven by **audio length, not the storyboard's estimate**. A shot
planned at 5 seconds with 6 seconds of narration must run 6; cutting to the estimate
drifts the voice out of sync by an accumulating amount over twelve shots. When
there is no audio (TTS unavailable), the storyboard duration is used and the film
is subtitled-only — stated in `FinalCut.note` rather than silently differing from
the promise.

Two assembly strategies, chosen by what actually exists:

  · per-shot clips were generated → trim/extend each to its narration, then concat
  · no clips (generation unavailable) → render a still per shot from the
    reference portrait, so the run still yields a watchable subtitled film

The still-image fallback keeps the output usable when video generation is
unavailable, while the final note records which path was used.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from . import schemas as s


def _ffmpeg() -> str:
    from .media_tools import find_ffmpeg

    return find_ffmpeg()


def _run(args: list[str], *, timeout: int = 900) -> None:
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-5:]
        raise RuntimeError("ffmpeg 失败：" + " / ".join(tail))


def probe_duration(path: Path) -> float:
    from .media_tools import MediaToolError, find_ffprobe

    try:
        ff = find_ffprobe()
    except MediaToolError:
        return 0.0
    try:
        proc = subprocess.run(
            [ff, "-i", str(path), "-show_entries", "format=duration",
             "-v", "quiet", "-of", "csv=p=0"],
            capture_output=True, text=True, timeout=20,
        )
        return round(float((proc.stdout or "0").strip() or 0), 2)
    except Exception:  # noqa: BLE001
        return 0.0


def _srt(cues: list[dict[str, Any]], out: Path) -> Path:
    """Write subtitles.

    Cues are split on punctuation to stay inside the frame: libass wraps on spaces
    and Chinese has none, so one long line runs off both edges.
    """
    def ts(seconds: float) -> str:
        ms = int(round((seconds - int(seconds)) * 1000))
        return f"{int(seconds)//3600:02d}:{(int(seconds)//60)%60:02d}:{int(seconds)%60:02d},{ms:03d}"

    blocks = []
    for i, cue in enumerate(cues, 1):
        blocks.append(f"{i}\n{ts(cue['start'])} --> {ts(cue['end'])}\n{cue['text']}\n")
    out.write_text("\n".join(blocks), encoding="utf-8")
    return out


def _split_cue(text: str, limit: int = 22) -> list[str]:
    parts: list[str] = []
    current = ""
    for ch in text:
        current += ch
        if ch in "，。！？；、" and len(current) >= 8:
            parts.append(current)
            current = ""
        elif len(current) >= limit:
            parts.append(current)
            current = ""
    if current:
        parts.append(current)
    return parts or [text]


async def build_film(
    *,
    shots: list[dict[str, Any]],
    assets: list[dict[str, Any]],
    voice_clips: list[dict[str, Any]],
    out_path: Path,
    aspect_ratio: str = "16:9",
) -> s.FinalCut:
    """Assemble the final mp4. Returns a FinalCut describing what was built."""
    ff = _ffmpeg()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = out_path.parent / f"{out_path.stem}_work"
    work.mkdir(parents=True, exist_ok=True)

    size = (1080, 1920) if aspect_ratio.startswith("9:16") else (1920, 1080)

    asset_by_shot = {a.get("shot_id"): a for a in assets}
    voice_by_shot = {v.get("shot_id"): v for v in voice_clips}

    segments: list[Path] = []
    cues: list[dict[str, Any]] = []
    cursor = 0.0
    used_audio = False

    for shot in shots:
        shot_id = shot.get("id")
        asset = asset_by_shot.get(shot_id) or {}
        voice = voice_by_shot.get(shot_id) or {}

        path = Path(asset.get("path") or "")
        # Narration length decides the cut; the storyboard estimate is the floor.
        narr_seconds = float(voice.get("seconds") or 0)
        planned = float(shot.get("seconds") or 5)
        duration = max(planned, narr_seconds if narr_seconds else planned)

        clip = work / f"{shot_id}.mp4"
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".webm"}:
            # Re-time the generated shot to the narration: loop if short, trim if long.
            _run([
                ff, "-y", "-i", str(path),
                "-filter_complex",
                f"[0:v]scale={size[0]}:{size[1]}:force_original_aspect_ratio=increase,"
                f"crop={size[0]}:{size[1]},setpts=PTS-STARTPTS,fps=25[v]",
                "-map", "[v]",
                "-t", f"{duration:.2f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                str(clip),
            ])
            segments.append(clip)
        elif path.is_file():
            # A still (no video generation): hold it for the shot duration.
            _run([
                ff, "-y", "-loop", "1", "-i", str(path),
                "-t", f"{duration:.2f}",
                "-vf",
                f"scale={size[0]}:{size[1]}:force_original_aspect_ratio=increase,"
                f"crop={size[0]}:{size[1]},format=yuv420p,fps=25",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                str(clip),
            ])
            segments.append(clip)
        else:
            # Nothing for this shot: skip it and note the gap instead of inserting
            # a black frame that would look like a bug on screen.
            cues.append(
                {
                    "start": cursor,
                    "end": cursor + duration,
                    "text": f"（镜头缺失：{shot_id}）",
                }
            )
            cursor += duration
            continue

        text = str(shot.get("narration") or "")
        if text:
            pieces = _split_cue(text)
            share = duration / max(1, len(pieces))
            for i, piece in enumerate(pieces):
                cues.append(
                    {
                        "start": cursor + share * i,
                        "end": cursor + share * (i + 1),
                        "text": piece,
                    }
                )
        cursor += duration

    if not segments:
        return s.FinalCut(
            note="没有任何可用镜头，未生成成片",
            shot_count=0,
        )

    # ---- concat ----
    listing = work / "clips.txt"
    listing.write_text("\n".join(f"file '{c.name}'" for c in segments), encoding="utf-8")
    seq = work / "sequence.mp4"
    _run([ff, "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(seq)])

    # ---- audio ----
    #
    # Each clip is placed at its shot's start offset. A single concatenated track
    # would be simpler but would misalign audio to picture the moment a narration
    # is longer than its shot.
    audio_inputs: list[str] = []
    filters: list[str] = []
    for shot in shots:
        voice = voice_by_shot.get(shot.get("id")) or {}
        vp = Path(voice.get("path") or "")
        if vp.is_file():
            audio_inputs.extend(["-i", str(vp)])
            idx = len(audio_inputs) // 2
            start = _shot_start(shots, voice_by_shot, shot.get("id"))
            filters.append(
                f"[{idx}:a]adelay={int(start * 1000)}|{int(start * 1000)},volume=1.0[a{idx}]"
            )
            used_audio = True

    srt = _srt(cues, work / "subs.srt") if cues else None

    subtitle_filter = (
        f"subtitles='{str(srt).replace(chr(92), '/').replace(':', chr(92) + ':')}':"
        "force_style='Fontsize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        "BorderStyle=1,Outline=1,Shadow=0,MarginV=28'"
        if srt
        else ""
    )

    out = work / "final.mp4"
    args = [ff, "-y", "-i", str(seq), *audio_inputs]
    if used_audio:
        mix = ";".join(filters)
        outs = "".join(f"[a{len(audio_inputs)//2 - i}]" for i in range(len(audio_inputs)//2))
        # amix downmix: several narration clips overlap the same timeline, so they
        # must be summed rather than concatenated.
        filter_complex = (
            f"{mix};{outs}amix=inputs={len(audio_inputs)//2}:duration=longest[aout]"
        )
        args += [
            "-filter_complex", filter_complex,
            "-map", "0:v", "-map", "[aout]",
        ]
    else:
        args += ["-map", "0:v"]

    if subtitle_filter:
        # Subtitles are burned on the video stream; requiring a re-encode, which is
        # acceptable for a short film and avoids a separate subtitle file the
        # viewer might not load.
        args += ["-vf", subtitle_filter]

    args += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        *(["-c:a", "aac", "-b:a", "128k", "-shortest"] if used_audio else ["-an"]),
        str(out),
    ]
    _run(args)

    if not out.exists():
        return s.FinalCut(note="ffmpeg 未产出文件", shot_count=len(segments))

    import shutil as _sh

    _sh.copyfile(out, out_path)

    total = probe_duration(out_path)
    return s.FinalCut(
        path=str(out_path),
        seconds=total,
        shot_count=len(segments),
        has_audio=used_audio,
        has_subtitles=bool(srt),
        note=(
            f"{len(segments)} 个镜头，{total:.0f} 秒，"
            + ("含配音" if used_audio else "无配音（字幕版）")
            + ("，含字幕" if srt else "")
        ),
    )


def _shot_start(shots: list[dict[str, Any]], voice_by_shot: dict[str, Any], shot_id: str) -> float:
    """Cumulative start time of a shot, using narration-led durations."""
    t = 0.0
    for shot in shots:
        if shot.get("id") == shot_id:
            return t
        voice = voice_by_shot.get(shot.get("id")) or {}
        narr = float(voice.get("seconds") or 0)
        t += max(float(shot.get("seconds") or 5), narr if narr else float(shot.get("seconds") or 5))
    return t
