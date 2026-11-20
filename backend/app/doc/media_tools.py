"""Locate and validate optional local FFmpeg tools."""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any


class MediaToolError(RuntimeError):
    """Raised when an optional local media helper is unavailable."""


FFMPEG_CANDIDATES = (
    "ffmpeg",
    r"C:\ffmpeg\bin\ffmpeg.exe",
)


def _hidden_process_options() -> dict[str, Any]:
    """Keep command-line media helpers from opening Windows error dialogs."""
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {
        "startupinfo": startupinfo,
        "creationflags": subprocess.CREATE_NO_WINDOW,
    }


@lru_cache(maxsize=16)
def _probe_ffmpeg(executable: str) -> tuple[bool, str]:
    """Return whether an FFmpeg candidate can start and identify itself."""
    try:
        proc = subprocess.run(
            [executable, "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            **_hidden_process_options(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    if proc.returncode != 0:
        unsigned = proc.returncode & 0xFFFFFFFF
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        reason = detail[-1] if detail else "进程初始化失败"
        return False, f"退出码 {proc.returncode} (0x{unsigned:08X})：{reason}"
    banner = (proc.stdout or proc.stderr or "").lower()
    if "ffmpeg version" not in banner:
        return False, "程序可启动，但没有返回 ffmpeg 版本信息"
    return True, ""


def _candidate_paths() -> list[str]:
    paths: list[str] = []
    for candidate in FFMPEG_CANDIDATES:
        if candidate == "ffmpeg":
            found = shutil.which("ffmpeg")
            if found:
                paths.append(str(Path(found).resolve()))
            continue
        path = Path(candidate)
        if path.is_file():
            paths.append(str(path.resolve()))
    return list(dict.fromkeys(paths))


def find_ffmpeg() -> str:
    failures: list[str] = []
    for executable in _candidate_paths():
        usable, reason = _probe_ffmpeg(executable)
        if usable:
            return executable
        failures.append(f"{executable}（{reason}）")
    if failures:
        raise MediaToolError(
            "找到 ffmpeg 文件，但程序无法启动：" + "；".join(failures)
            + "。它仅用于可选的本地成片拼接；视频镜头仍由 API 生成。"
        )
    raise MediaToolError(
        "未安装系统 ffmpeg，已跳过本地成片拼接；视频镜头仍由 API 生成并保留。"
    )


def find_ffprobe() -> str:
    """Locate FFprobe next to FFmpeg or on PATH."""
    found = shutil.which("ffprobe")
    if found:
        return found
    try:
        ffmpeg = find_ffmpeg()
    except MediaToolError:
        raise MediaToolError("找不到 ffprobe：ffmpeg 也不可用") from None
    path = Path(ffmpeg)
    sibling = path.with_name(
        "ffprobe.exe" if path.suffix.lower() == ".exe" else "ffprobe"
    )
    if sibling.is_file():
        return str(sibling)
    raise MediaToolError(f"找不到 ffprobe（应与 ffmpeg 同目录：{path.parent}）")
