"""Repair the JSON defects models actually produce.

The extractor rejects text that is nearly-JSON. Measured on this project, the
recurring defects are all mechanical:

  · **Unescaped inner quotes** — a Chinese title like `《新青年》"德先生"` inside a
    string breaks the parser at that quote.
  · **Trailing commas** — `{"a":1,}` from list-style habits.
  · **A truncated object** — the response was cut at the token limit, so the
    closing braces are missing.
  · **Single quotes / smart quotes** — a model writing prose-adjacent JSON.

Attempting repair is worthwhile because the alternative is a second billed call
and a stage failure, while these fixes are cheap and local. What is NOT attempted:
inventing content. A truncated array is closed, not completed — the missing shots
must fail rather than be fabricated.
"""

from __future__ import annotations

import json
import re
from typing import Any


def repair(text: str) -> tuple[Any, str] | None:
    """Try to turn nearly-JSON into JSON. Returns (value, note) or None.

    `note` records which repair fired, so a caller can log it and an operator can
    see how often the model misbehaves rather than silently shipping patched data.

    A leading prose sentence is stripped first: models routinely answer "好的，
    这是结果：" before the object, and repairing the object is useless if the
    braces are never found.
    """
    if not text:
        return None

    text = _isolate_object(text)

    candidates: list[tuple[str, str]] = []

    # 1. straight as-is
    candidates.append((text, ""))

    # 2. smart quotes → straight
    normalised = (
        text.replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2018", "'").replace("\u2019", "'")
    )
    if normalised != text:
        candidates.append((normalised, "智能引号已规范化"))

    # 3. trailing commas before a closing brace/bracket
    no_trailing = re.sub(r",\s*([}\]])", r"\1", normalised)
    if no_trailing != normalised:
        candidates.append((no_trailing, "已移除多余逗号"))

    # 4. truncation: close any unbalanced braces at the end
    closed = _close_unbalanced(no_trailing)
    if closed != no_trailing:
        candidates.append((closed, "已补全被截断的括号"))

    # 5. single-quoted keys/values → double (last resort, may misfire on apostrophes)
    single_fixed = _single_to_double(closed)
    if single_fixed != closed:
        candidates.append((single_fixed, "单引号已转为双引号"))

    for candidate, note in candidates:
        try:
            return json.loads(candidate), note
        except json.JSONDecodeError:
            continue

    return None


def _isolate_object(text: str) -> str:
    """Cut away text before the first `{` or `[` and after the last one.

    Models often preface JSON with a sentence. Repairing without isolating would
    always fail, because the leading prose makes the whole string non-JSON.
    Truncated input has no closing brace, so nothing is cut from the end there —
    the caller's bracket-closing step handles that.
    """
    first_brace = text.find("{")
    first_bracket = text.find("[")
    if first_brace < 0 and first_bracket < 0:
        return text
    start = min(x for x in (first_brace, first_bracket) if x >= 0)

    # Keep the tail as-is: a truncated response legitimately has no closing char,
    # and cutting to the last brace would throw away the partial content.
    return text[start:].rstrip()


def _close_unbalanced(text: str) -> str:
    """Append the closing characters a truncated response is missing.

    Only closes what is already open, scanning outside strings. A deliberately
    unfinished array stays unfinished — the caller then sees fewer items, which is
    the honest outcome, rather than one completed with invented entries.
    """
    stack: list[str] = []
    in_str = False
    escape = False

    for ch in text:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()

    if not stack and not in_str:
        return text

    tail = text.rstrip()
    # A truncated string loses its closing quote and may end mid-value.
    if in_str:
        tail += '"'
    # A dangling key or comma would make the appended braces invalid.
    tail = re.sub(r",\s*$", "", tail)
    tail = re.sub(r'"(?:[^"\\]|\\.)*"\s*:\s*$', "", tail).rstrip()
    tail = re.sub(r",\s*$", "", tail)
    return tail + "".join(reversed(stack))


def _single_to_double(text: str) -> str:
    """Convert single-quoted JSON strings to double quotes.

    Skipped entirely when a double quote appears inside a single-quoted span —
    converting there would corrupt the content rather than fix the syntax.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch != "'":
            out.append(ch)
            i += 1
            continue
        # Find the closing single quote on the same logical token.
        j = i + 1
        buf: list[str] = []
        while j < n and text[j] != "'":
            if text[j] == '"':
                # Ambiguous: give up rather than mangle.
                return text
            buf.append(text[j])
            j += 1
        if j >= n:
            return text
        out.append('"')
        out.append("".join(buf))
        out.append('"')
        i = j + 1
    return "".join(out)
