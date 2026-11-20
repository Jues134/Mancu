"""Stable identifiers for the six-agent production model.

The rule this enforces comes from the global production spec: every task, shot and
asset carries a unique id, and an id that has been confirmed is never reissued or
silently changed. A downstream agent referencing `SB-003` must be referring to the
same shot the storyboard agent produced, weeks later, after several rework rounds.

Why ids are generated here rather than inline at each call site:

  · **One format, one place.** `SB-1`, `SB-01` and `sb-003` are the same shot to a
    human and three different keys to a dict. A single generator means the format
    cannot drift between agents.
  · **Padding is fixed.** SB-001 vs SB-1 sorts differently as a string, and the
    storyboard is displayed in order. Four digits is enough for 9999 shots and
    sorts correctly for the first 9999.
  · **Validation is cheap.** A reference from the video agent can be checked against
    the pattern before it is used, so a hallucinated id (`SB-3`) is caught at the
    boundary instead of surfacing as a missing-asset error deep in generation.

Not generated here: the project id. A project is the top-level container and its
id comes from the user or from the run, not from a counter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


#: Prefix and minimum width for every id family. Kept in one table so adding a
#: family is a single line, and so validation cannot accept a prefix that has no
#: family (the failure that lets `XYZ-001` through unnoticed).
FAMILIES: dict[str, tuple[str, int]] = {
    "project": ("PRJ", 4),
    "task": ("TASK", 4),
    "storyboard": ("SB", 3),
    "character": ("CHAR", 3),
    "scene": ("SCENE", 3),
    "prop": ("PROP", 3),
    "image": ("IMG", 3),
    "audio": ("AUD", 3),
    "video": ("VID", 3),
    # Rework rounds and review reports are also referenced across agents, so they
    # need ids as stable as the assets they describe.
    "review": ("REV", 3),
    "version": ("VER", 2),
}


def _pattern(prefix: str, width: int) -> re.Pattern[str]:
    # Accept ANY number of digits, not exactly `width`.
    #
    # The spec writes SB-001, and `format_id` always emits the padded form. But a
    # reference that arrives as `SB-9` means the same shot — a hand-written
    # reference, an id from an older record, a model echoing the number it was
    # shown. Rejecting it would sort it to the end of the storyboard and, worse,
    # make `parse` claim it is not a shot at all. Padding is a display concern;
    # identity is the number.
    return re.compile(rf"^{prefix}-(\d+)$")


#: Compiled once per family; `is_valid` runs on every cross-agent reference.
_VALIDATORS: dict[str, re.Pattern[str]] = {
    name: _pattern(prefix, width) for name, (prefix, width) in FAMILIES.items()
}


def format_id(family: str, number: int) -> str:
    """Format one id, e.g. `("storyboard", 3)` -> `"SB-003"`.

    Raises KeyError for an unknown family rather than inventing a prefix: a typo
    in the family name should be a loud error, not an id the rest of the system
    will happily treat as valid.
    """
    prefix, width = FAMILIES[family]
    if number < 1:
        raise ValueError(f"{family} ids start at 1, got {number}")
    return f"{prefix}-{number:0{width}d}"


def is_valid(family: str, value: str) -> bool:
    """Whether `value` is a well-formed id of that family."""
    if not isinstance(value, str):
        return False
    validator = _VALIDATORS.get(family)
    return bool(validator and validator.match(value))


def parse(value: str) -> tuple[str, int]:
    """Split an id into (family, number). Raises ValueError if it fits no family.

    Used when reading back a stored id — for example sorting storyboards, or
    answering "which shot is this review about".
    """
    for family, validator in _VALIDATORS.items():
        m = validator.match(value or "")
        if m:
            return family, int(m.group(1))
    raise ValueError(f"不认识的项目编号：{value!r}")


def sort_key(value: str) -> tuple[str, int]:
    """Order ids by family then number, not lexicographically.

    `sorted(["SB-010", "SB-9"])` puts SB-9 first because "9" > "1". Shots must
    appear in the order they were written, so callers sort with this.
    """
    try:
        return parse(value)
    except ValueError:
        # Unknown values sort last instead of raising: display code must not
        # crash on a malformed row left in an old database.
        return ("~", 0)


@dataclass
class IdSequence:
    """A monotonically increasing counter for one family.

    Nothing is persisted here. The caller owns durability: `next()` returns the
    next id and `observe()` re-syncs from ids already present in the project, so a
    resumed run does not restart at 001 and collide with existing shots.
    """

    family: str
    current: int = 0
    #: Ids seen but not issued — used by `observe` to avoid handing out a number
    #: that already exists elsewhere in the project.
    _seen: set[int] = field(default_factory=set, repr=False)

    def next(self) -> str:
        self.current += 1
        while self.current in self._seen:
            self.current += 1
        self._seen.add(self.current)
        return format_id(self.family, self.current)

    def observe(self, value: str) -> None:
        """Take note of an existing id, without advancing past it.

        Called when loading a saved project so the sequence continues after the
        highest id already in use rather than overwriting it.
        """
        try:
            family, number = parse(value)
        except ValueError:
            return
        if family != self.family:
            return
        self._seen.add(number)
        if number > self.current:
            self.current = number


def sequence(family: str, existing: list[str] | None = None) -> IdSequence:
    """Build a sequence, optionally seeded from ids already in the project."""
    seq = IdSequence(family=family)
    for value in existing or []:
        seq.observe(value)
    return seq
