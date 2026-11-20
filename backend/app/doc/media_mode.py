"""Whether media is actually generated, or only planned.

Why a switch rather than deleting the generation code:

The image and video endpoints need a funded key. Without one, the pipeline
currently raises "没有可用的生成器（缺少供应商 Key）" and the whole run dies — so
a user with no key can see nothing at all, not even the prompts.

This switch makes "plan" the default: every stage still produces its prompt and
its validation, but nothing is billed. Supplying a key and setting
`STUDIO_MEDIA_MODE=full` turns generation back on without re-adding code.

The mode is read per call, not cached at import: tests and the CLI flip it at
runtime, and a cached value would make the switch untestable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

#: prompt — produce prompts and validate them; never call a generation endpoint
#: full   — generate images and video as well, when a key is available
MODE_PROMPT = "prompt"
MODE_FULL = "full"

_VALID = {MODE_PROMPT, MODE_FULL, "", None}


def _config_mode() -> str | None:
    """模式也可以写进配置目录的 studio.json，而不是只认环境变量。

    只认环境变量时，同一份代码用不同方式启动会得到不同行为：start.ps1 里
    设了 full，而 `python -m uvicorn ...` 手动起就退回 prompt，界面上表现为
    「出图/出视频勾不上」——用户看到的是坏掉的控件，实际是启动参数不同。
    环境变量仍然优先，方便临时覆盖；配置只是把默认值固定下来。
    """
    base = (os.environ.get("STUDIO_CONFIG_DIR") or "").strip()
    if not base:
        return None
    try:
        data = json.loads((Path(base) / "studio.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    raw = str(data.get("media_mode") or "").strip().lower()
    return raw if raw in {MODE_PROMPT, MODE_FULL} else None


def media_mode() -> str:
    """Current mode, normalised. An unknown value falls back to `prompt`.

    Falling back to the safe mode is deliberate: a typo in the environment should
    not silently start spending money.

    `None` (variable absent) means "not configured here" and falls through to the
    config file; an empty string means "explicitly prompt", so it must not be
    confused with absence.
    """
    raw = os.environ.get("STUDIO_MEDIA_MODE")
    if raw is not None:
        raw = raw.strip().lower()
        if raw in _VALID:
            return raw or MODE_PROMPT
        return MODE_PROMPT
    return _config_mode() or MODE_PROMPT


def prompt_only() -> bool:
    """True when media must not be generated."""
    return media_mode() == MODE_PROMPT


def describe() -> str:
    """One line for the UI, so the mode is visible rather than guessed."""
    if prompt_only():
        return "提示词模式：只产出提示词与校验，不生成图片/视频"
    return "完整模式：会真实生成图片与视频（消耗额度）"
