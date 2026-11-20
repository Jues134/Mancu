"""Operational logging — the record you go back to after something went wrong.

Three things, because they answer three different questions:

  · **`app.log` / `error.log`** — the narrative: what ran, what failed, and for a
    failure the *full traceback*. The run record in SQLite keeps `str(exc)[:400]`,
    which is right for the UI and useless for debugging: it names the symptom and
    throws away the line that caused it.
  · **`events-YYYY-MM-DD.jsonl`** — one line per call. "It felt slow and then
    failed" is invisible in a narrative log; a line saying `status=429 attempt=2
    ms=8421` is the whole story.
  · **contextual ids** — a request id and a run id carried by a `ContextVar`, so a
    line written five frames deep still says which run it belongs to. Threading
    those two strings through fifty function signatures is not an option.

Why files instead of the database: SQLite here is a single writer behind
`db._lock`, and logging every model call through it would serialise log writes
against run-state writes. The logging would start causing the stalls it exists to
explain.

Two rules this module refuses to break:

  · **It never raises.** A full disk, a locked file, a bad format string — none of
    those may turn into a failed request. Every public function swallows its own
    errors (and says so in a comment, not silently).
  · **Secrets never reach disk.** Keys arrive in Authorization headers, provider
    JSON and error bodies, and those bodies are exactly what gets logged when a
    call fails. Everything written passes through `mask()`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import sys
import threading
from contextvars import ContextVar
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Iterator

from .config import DATA_DIR


#: Where the logs live. Under DATA_DIR on purpose: it is the directory a user
#: backs up or deletes, and a log that outlives the data it describes is a trap.
LOG_DIR = Path(os.environ.get("STUDIO_LOG_DIR") or (DATA_DIR / "logs"))

APP_LOG = LOG_DIR / "app.log"
ERROR_LOG = LOG_DIR / "error.log"

MAX_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 5

#: Event files are pruned by age on startup. Two weeks is enough to answer "it
#: worked last month" for the cases where that matters, and bounded on disk.
KEEP_DAYS = 14

#: How much of a prompt, response body or error text to keep. Enough to recognise
#: which request it was; short enough that a hundred calls stay greppable.
SNIPPET = 300


# ---------------------------------------------------------------------------
# Context — who is writing this line
# ---------------------------------------------------------------------------

_request_id: ContextVar[str] = ContextVar("request_id", default="-")
_run_id: ContextVar[str] = ContextVar("run_id", default="-")
_agent: ContextVar[str] = ContextVar("agent", default="-")


@contextlib.contextmanager
def request_context(request_id: str) -> Iterator[None]:
    """Tag every line written inside this request with its id."""
    token = _request_id.set(request_id or "-")
    try:
        yield
    finally:
        _request_id.reset(token)


def current_run_id() -> str:
    value = _run_id.get()
    return "" if value == "-" else value


def bind_run(run_id: str, agent: str = "-") -> None:
    """Tag this task's remaining lines with the run id, without a scope to exit.

    For entry points whose body cannot be wrapped in a `with` without indenting
    two hundred lines. Safe because a background task runs in its own copy of the
    context: setting it here cannot leak into another run.
    """
    _run_id.set(run_id or "-")
    if agent != "-":
        _agent.set(agent)


def current_request_id() -> str:
    value = _request_id.get()
    return "" if value == "-" else value


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

#: Applied to every string that reaches a handler. Order matters only in that the
#: generic key pattern must not eat the `sk-` one first — both end up as `***`.
_SECRETS: tuple[tuple[re.Pattern[str], str], ...] = (
    # OpenAI-style keys, which is what every provider here uses.
    (re.compile(r"sk-[A-Za-z0-9_\-]{8,}"), "sk-***"),
    # `Authorization: Bearer <token>` in a header echo or a repr of a dict.
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}"), r"\1***"),
    # JSON bodies: {"api_key": "...", "token": "..."}.
    (
        re.compile(r"(?i)([\"']?(?:api_key|apikey|api-key|access_token|token|secret|password)[\"']?\s*[:=]\s*[\"']?)([^\s\"',}&]{4,})"),
        r"\1***",
    ),
)


def mask(text: str) -> str:
    """Replace anything that looks like a credential.

    Deliberately blunt: a false positive turns one log field into `***`, while a
    false negative writes a live key into a file the user is about to email to
    someone.
    """
    if not text:
        return text
    for pattern, replacement in _SECRETS:
        text = pattern.sub(replacement, text)
    return text


def clip(value: Any, limit: int = SNIPPET) -> str:
    """Trim to `limit` chars, keeping the original length visible.

    The length matters: a 4000-character prompt clipped to 300 looks the same as a
    300-character one, and that difference decides whether truncation was the bug.
    """
    text = value if isinstance(value, str) else str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…（共 {len(text)} 字）"


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


class _ContextFilter(logging.Filter):
    """Fill in the context fields and mask the message in one pass.

    Masking happens here rather than in the formatter because it must also cover
    `%`-style arguments: `logger.warning("body=%s", raw)` is the common shape, and
    a formatter only ever sees the already-interpolated string.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        record.run_id = _run_id.get()
        record.agent = _agent.get()
        if isinstance(record.msg, str):
            record.msg = mask(record.msg)
        args = record.args
        if isinstance(args, tuple):
            record.args = tuple(mask(a) if isinstance(a, str) else a for a in args)
        elif isinstance(args, dict):
            record.args = {k: (mask(v) if isinstance(v, str) else v) for k, v in args.items()}
        return True


class _Formatter(logging.Formatter):
    """Timestamp, level, module, then the two ids that make a line findable."""

    def __init__(self) -> None:
        super().__init__(
            "%(asctime)s %(levelname)-5s %(name)s [req=%(request_id)s run=%(run_id)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        # Every handler exists before its formatter runs, but `extra` is not
        # guaranteed; default them so a bare `logging.warning` outside our setup
        # never raises "format requires a mapping".
        for attr in ("request_id", "run_id", "agent"):
            if not hasattr(record, attr):
                setattr(record, attr, "-")
        return super().format(record)


_setup_lock = threading.Lock()
_configured = False
_level = logging.INFO


def _resolve_level(explicit: str | int | None) -> int:
    raw: str | int | None = explicit
    if raw is None:
        raw = os.environ.get("STUDIO_LOG_LEVEL") or os.environ.get("MANCU_LOG_LEVEL") or ""
    if isinstance(raw, int):
        return raw
    name = str(raw).strip().upper()
    if not name:
        return logging.INFO
    value = logging.getLevelName(name)
    return value if isinstance(value, int) else logging.INFO


def setup(level: str | int | None = None) -> None:
    """Install the handlers once. Safe to call from several entry points.

    Called from `main.py` (so `uvicorn app.main:app` works) and from `serve.py`
    (so a crash *while importing* main is still written down — at that point
    main.py has not run a single line, and the only copy of the error would be a
    console window the user is about to close).
    """
    global _configured, _level
    with _setup_lock:
        if _configured:
            return
        _level = _resolve_level(level)

        # A Chinese Windows console is cp936; a redirected stream may be anything.
        # `replace` turns an unmappable character into '?', which is a cosmetic
        # loss, instead of an OSError that logging reports as a logging failure.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(errors="replace")  # type: ignore[union-attr]
            except (AttributeError, ValueError, OSError):
                pass

        root = logging.getLogger()
        root.setLevel(_level)
        # Re-entrant setup (tests, reloaders) must not stack duplicate handlers:
        # that is how one line turns into eight.
        for handler in list(root.handlers):
            root.removeHandler(handler)
            if getattr(handler, "_mancu", False):
                handler.close()

        context = _ContextFilter()
        root.addHandler(_build_console(context))
        root.addHandler(_build_file(APP_LOG, context, _level, BACKUP_COUNT))
        # Errors get their own file and their own, smaller rotation: this is the
        # one a user is asked to open, so it should contain nothing but problems.
        root.addHandler(_build_file(ERROR_LOG, context, logging.WARNING, 3))

        # uvicorn configures its own loggers and sets `propagate = False`, which
        # would silently bypass every handler above. Hand them back to the root.
        for name in ("uvicorn", "uvicorn.error"):
            logger = logging.getLogger(name)
            logger.handlers = []
            logger.propagate = True
            logger.setLevel(_level)
        # Its access log duplicates what our middleware records, at one line per
        # second while a run is polling.
        access = logging.getLogger("uvicorn.access")
        access.handlers = []
        access.propagate = False

        # Third-party transport chatter at INFO is one line per request.
        for name in ("httpx", "httpcore", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)

        _configured = True


def _build_console(context: logging.Filter) -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(_level)
    handler.setFormatter(_Formatter())
    handler.addFilter(context)
    handler._mancu = True  # type: ignore[attr-defined]
    return handler


def _build_file(
    path: Path, context: logging.Filter, level: int, backups: int
) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        path, maxBytes=MAX_BYTES, backupCount=backups, encoding="utf-8", delay=True
    )
    handler.setLevel(level)
    handler.setFormatter(_Formatter())
    handler.addFilter(context)
    handler._mancu = True  # type: ignore[attr-defined]
    return handler


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def level_name() -> str:
    return logging.getLevelName(_level)


# ---------------------------------------------------------------------------
# Structured events
# ---------------------------------------------------------------------------

_event_lock = threading.Lock()


def event(name: str, level: str = "info", **fields: Any) -> None:
    """Append one structured line to today's JSONL file.

    Written and closed per call. That costs a syscall and buys the thing that
    matters most: when the process is killed mid-run — which is exactly when
    someone goes looking — the last line is already on disk. A buffered handle
    would lose precisely the tail that explains the crash.

    Never raises. A logging failure must not become a request failure.
    """
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        # `run_id=` 作为字段传进来时用它填 run 列，而不是多出一个同名键：
        # 否则按 run 过滤会漏掉这些行——它们看起来属于「-」那次运行。
        explicit_run = fields.pop("run_id", None)
        record: dict[str, Any] = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "level": level,
            "event": name,
            "req": _request_id.get(),
            "run": str(explicit_run) if explicit_run else _run_id.get(),
            "agent": _agent.get(),
        }
        for key, value in fields.items():
            if isinstance(value, str):
                record[key] = mask(clip(value))
            elif isinstance(value, (int, float, bool)) or value is None:
                record[key] = value
            else:
                record[key] = mask(clip(json.dumps(value, ensure_ascii=False, default=str)))
        path = LOG_DIR / f"events-{datetime.now():%Y-%m-%d}.jsonl"
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _event_lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception:  # noqa: BLE001 - logging must never break the caller
        return


def read_events(
    *, limit: int = 200, run_id: str = "", level: str = "", days: int = 3
) -> list[dict[str, Any]]:
    """Newest-first events, filtered. Used by the diagnostics endpoints."""
    try:
        files = sorted(LOG_DIR.glob("events-*.jsonl"))[-max(1, days):]
    except OSError:
        return []
    wanted_level = level.strip().lower()
    out: list[dict[str, Any]] = []
    for path in reversed(files):
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if run_id and record.get("run") != run_id:
                continue
            if wanted_level and str(record.get("level", "")).lower() != wanted_level:
                continue
            out.append(record)
            if len(out) >= limit:
                return out
    return out


def tail(path: Path, limit: int = 200) -> str:
    """Last `limit` lines of a text log, for the diagnostics bundle."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-limit:])


def prune() -> int:
    """Delete event files older than `KEEP_DAYS`. Returns how many went."""
    cutoff = datetime.now() - timedelta(days=KEEP_DAYS)
    removed = 0
    try:
        for path in LOG_DIR.glob("events-*.jsonl"):
            stamp = path.stem.replace("events-", "")
            try:
                day = datetime.strptime(stamp, "%Y-%m-%d")
            except ValueError:
                continue
            if day < cutoff:
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    continue
    except OSError:
        return removed
    return removed


def summary() -> dict[str, Any]:
    """What is on disk, for the diagnostics page and the startup line."""
    files: list[dict[str, Any]] = []
    total = 0
    try:
        entries = sorted(LOG_DIR.glob("*")) if LOG_DIR.is_dir() else []
    except OSError:
        entries = []
    for path in entries:
        try:
            stat = path.stat()
        except OSError:
            continue
        if not path.is_file():
            continue
        total += stat.st_size
        files.append(
            {
                "name": path.name,
                "kb": round(stat.st_size / 1024, 1),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            }
        )
    return {
        "dir": str(LOG_DIR),
        "level": level_name(),
        "total_kb": round(total / 1024, 1),
        "keep_days": KEEP_DAYS,
        "files": files,
    }
