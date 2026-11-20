"""Voiceover — text to speech, with an honest fallback chain.

Why a chain rather than one provider:

The paid endpoint on the configured provider returned `429 余额不足`. A pipeline
whose narration silently depends on a funded account stops when quota is
exhausted. So this tries, in order:

  1. **edge-tts** (free, high quality, needs network)
  2. **Windows SAPI** via PowerShell (free, offline, robotic but works)
  3. **silence** — returns no audio and says so

The third option is not a failure path bolted on: a film with burned-in subtitles
and no voice is still viewable, and pretending otherwise would mean the compositor
gets no signal. Every clip therefore reports `ok`, the provider used, and how long
the audio actually runs — because the compositor needs the duration to cut shots
to the narration rather than guessing.

Duration measurement is by ffprobe on the produced file. Asking the TTS engine for
an estimate instead would let a wrong estimate silently desynchronise the edit.
"""

from __future__ import annotations

import asyncio
import shutil
from functools import lru_cache
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


#: Chinese voices, by gender and use. Edge TTS names; the first is the default.
VOICES = {
    "male": "zh-CN-YunxiNeural",
    "female": "zh-CN-XiaoxiaoNeural",
    "narrator": "zh-CN-YunjianNeural",   # deeper, documentary register
    "young": "zh-CN-YunyangNeural",
}

DEFAULT_VOICE = VOICES["narrator"]


@dataclass
class VoiceResult:
    path: str = ""
    seconds: float = 0.0
    voice: str = ""
    provider: str = ""
    ok: bool = False
    error: str = ""


async def synthesize(text: str, out_path: Path, *, voice: str = "") -> VoiceResult:
    """Produce an mp3 for `text`. Falls through the providers until one works."""
    text = (text or "").strip()
    if not text:
        return VoiceResult(error="旁白为空")

    voice_name = voice or DEFAULT_VOICE
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. edge-tts ----
    try:
        result = await _edge_tts(text, out_path, voice_name)
        if result.ok:
            return result
    except Exception as exc:  # noqa: BLE001 - try the next provider
        edge_error = str(exc)[:120]
    else:
        edge_error = "edge-tts 未安装"

    # ---- 2. Windows SAPI ----
    try:
        result = await _sapi(text, out_path)
        if result.ok:
            return result
    except Exception as exc:  # noqa: BLE001
        sapi_error = str(exc)[:120]
    else:
        sapi_error = "SAPI 不可用"

    # ---- 3. silence ----
    # Returning ok=False with a reason keeps the compositor honest: it will burn
    # subtitles and skip the audio track instead of mixing a missing file.
    return VoiceResult(
        provider="none",
        ok=False,
        error=f"无可用 TTS（edge: {edge_error}；sapi: {sapi_error}）",
    )


async def _edge_tts(text: str, out_path: Path, voice: str) -> VoiceResult:
    """Use edge-tts when installed.

    Runs the async API directly — spawning its CLI would need a child process,
    which is blocked in some sandboxes and slower besides.
    """
    import edge_tts  # noqa: WPS433 - optional dependency

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))
    if not out_path.exists() or out_path.stat().st_size < 100:
        return VoiceResult(provider="edge-tts", voice=voice, error="生成的音频为空")

    return VoiceResult(
        path=str(out_path),
        seconds=probe_duration(out_path),
        voice=voice,
        provider="edge-tts",
        ok=True,
    )


async def _sapi(text: str, out_path: Path) -> VoiceResult:
    """Windows built-in speech, used when edge-tts is unavailable.

    SAPI writes WAV to a file; ffmpeg then transcodes to mp3 so the compositor sees
    one format everywhere. Quality is robotic, which is stated in the returned
    provider name rather than hidden.
    """
    if not shutil.which("powershell") and not Path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe").exists():
        return VoiceResult(provider="sapi", error="当前系统没有 PowerShell")

    wav = out_path.with_suffix(".wav")
    # The text is passed via a temp file rather than interpolated into the script:
    # narration contains quotes and brackets, and inline interpolation would break
    # the command.
    script = out_path.with_suffix(".ps1")
    text_file = out_path.with_suffix(".txt")
    text_file.write_text(text, encoding="utf-8")
    script.write_text(
        "$t = Get-Content -Raw -Encoding UTF8 '" + str(text_file).replace("'", "''") + "'; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.SetOutputToWaveFile('" + str(wav).replace("'", "''") + "'); "
        "$s.Speak($t); $s.Dispose()",
        encoding="utf-8",
    )
    proc = await asyncio.create_subprocess_exec(
        "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()

    if not wav.exists() or wav.stat().st_size < 100:
        return VoiceResult(provider="sapi", error="SAPI 未生成音频")

    ff = _ffmpeg()
    if ff:
        subprocess.run(
            [ff, "-y", "-i", str(wav), "-codec:a", "libmp3lame", "-q:a", "4", str(out_path)],
            capture_output=True,
        )
    if not out_path.exists():
        shutil.move(str(wav), str(out_path))

    return VoiceResult(
        path=str(out_path),
        seconds=probe_duration(out_path),
        voice="windows-sapi",
        provider="sapi",
        ok=True,
    )


def probe_duration(path: Path) -> float:
    """Actual audio length, from ffprobe.

    Required, not cosmetic: the compositor cuts each shot to its narration, and an
    estimated duration would let the voice and the picture drift apart over a
    twelve-shot film.
    """
    ff = _ffprobe()
    if not ff:
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


def list_supported() -> dict[str, object]:
    """What the UI needs to know before promising audio.

    `sapi` reports whether a synthesis path *actually exists*, not merely whether
    `powershell.exe` is on PATH. The difference matters: on a machine where the
    shell is present but speech output is blocked (policy, a missing voice pack, a
    sandbox), the old check said "available" and the run then produced a silent
    film with no explanation. Presence is not capability.

    The probe is cheap and cached per process: it writes one short WAV to a temp
    path and asks for the file's existence. Cached because the capability endpoint
    is polled by the UI and spawning PowerShell on every poll would be wasteful.
    """
    edge = False
    try:
        import edge_tts  # noqa: F401
        edge = True
    except ImportError:
        pass

    return {
        "edge_tts": edge,
        "sapi": _sapi_usable(),
        "voices": VOICES,
        "note": (
            "优先使用 edge-tts（免费）；未安装时回退 Windows SAPI（离线但机械）；"
            "两者都不可用时输出无声字幕版，并在运行时明确提示。"
        ),
    }


#: Cached result of the SAPI probe: None means "not probed yet".
@lru_cache(maxsize=1)
def _sapi_usable() -> bool:
    """Whether Windows speech actually produces a file on this machine.

    A real probe rather than a PATH check. Failures are common and silent: a
    locked-down machine, a missing voice, or a sandbox that rejects the write all
    present as "powershell exists", and all of them mean no audio.

    The probe directory is created under the data directory rather than the system
    temp: a restricted environment can refuse to clean up a temp directory, and the
    resulting exception would be indistinguishable from "speech is broken". A
    failure to probe is reported as False, never raised.
    """
    if not shutil.which("powershell"):
        return False

    from pathlib import Path as _Path

    try:
        from ..config import DATA_DIR

        probe_dir = _Path(DATA_DIR) / ".tts-probe"
    except Exception:  # noqa: BLE001
        return False

    wav = probe_dir / "probe.wav"
    try:
        probe_dir.mkdir(parents=True, exist_ok=True)
        if wav.exists():
            wav.unlink()

        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.SetOutputToWaveFile('{wav}'); "
            "$s.Speak('a'); $s.Dispose()"
        )
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
            capture_output=True,
            timeout=20,
        )
        return proc.returncode == 0 and wav.exists() and wav.stat().st_size > 100
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        # Best effort: a leftover probe file is harmless, an exception here is not.
        try:
            if wav.exists():
                wav.unlink()
        except OSError:
            pass


def _ffmpeg() -> str:
    from .media_tools import find_ffmpeg

    try:
        return find_ffmpeg()
    except Exception:  # noqa: BLE001
        return ""


def _ffprobe() -> str:
    """Locate ffprobe via the shared resolver.

    Derived from the ffmpeg location rather than guessed by string replacement:
    the bundled binaries live in backend/vendor, and a blind replace would also
    rewrite a directory that happens to contain the word "ffmpeg".
    """
    from .media_tools import MediaToolError, find_ffprobe

    try:
        return find_ffprobe()
    except MediaToolError:
        return ""
