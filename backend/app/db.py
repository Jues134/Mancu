"""SQLite persistence for runs, steps, artifacts and reviews.

Why plain sqlite3 instead of an ORM: the schema is compact, and explicit column
lists keep migrations and writes easy to inspect.

Concurrency note: FastAPI may serve several requests at once, but SQLite allows
one writer. The connection uses WAL + a busy timeout, and every write goes
through `transaction()` which serialises on a module lock.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import DB_PATH, ensure_dirs

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None


SCHEMA = """
CREATE TABLE IF NOT EXISTS studio_runs (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL DEFAULT 'queued',
    brief       TEXT NOT NULL DEFAULT '{}',
    cost_usd    REAL NOT NULL DEFAULT 0,
    error       TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS studio_steps (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES studio_runs(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    agent       TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    input       TEXT NOT NULL DEFAULT '{}',
    output      TEXT NOT NULL DEFAULT '{}',
    error       TEXT NOT NULL DEFAULT '',
    cost_usd    REAL NOT NULL DEFAULT 0,
    started_at  TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_steps_run ON studio_steps(run_id, seq);
CREATE TABLE IF NOT EXISTS studio_artifacts (
    id       TEXT PRIMARY KEY,
    run_id   TEXT NOT NULL REFERENCES studio_runs(id) ON DELETE CASCADE,
    step_id  TEXT NOT NULL DEFAULT '',
    kind     TEXT NOT NULL DEFAULT 'json',
    label    TEXT NOT NULL DEFAULT '',
    path     TEXT NOT NULL DEFAULT '',
    meta     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_artifacts_run ON studio_artifacts(run_id);

CREATE TABLE IF NOT EXISTS studio_reviews (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES studio_runs(id) ON DELETE CASCADE,
    step_id     TEXT NOT NULL DEFAULT '',
    verdict     TEXT NOT NULL DEFAULT 'pass',
    score       REAL NOT NULL DEFAULT 0,
    issues      TEXT NOT NULL DEFAULT '[]',
    summary     TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reviews_run ON studio_reviews(run_id);

-- The knowledge base: what the studio already knows, so a topic is researched
-- once instead of on every run. Two facts live side by side and must not be
-- conflated — the *content* (what is true) and the *decision* (whether a human
-- accepted it as ours). `status` is the decision; `origin` says where the text
-- came from. A row can be used by the run that created it while still `pending`,
-- which is why usage and approval are separate columns rather than one flag.
CREATE TABLE IF NOT EXISTS knowledge_entries (
    id           TEXT PRIMARY KEY,
    title        TEXT NOT NULL,
    category     TEXT NOT NULL DEFAULT '',
    period       TEXT NOT NULL DEFAULT '',
    body         TEXT NOT NULL DEFAULT '',
    key_points   TEXT NOT NULL DEFAULT '[]',
    sources      TEXT NOT NULL DEFAULT '[]',
    source_material TEXT NOT NULL DEFAULT '[]', -- fetched text used to build the draft
    status       TEXT NOT NULL DEFAULT 'pending',   -- pending | approved | rejected
    -- card 是「AI 从权威来源摘要出的一条主题卡片」，book 是「用户整本上传的
    -- 资料」。两者共用一张表和同一条检索路径，因为对检索而言它们没有区别：
    -- 都是「一段可召回、需要出处的文字」。区别只在来源与生命周期——卡片要人工
    -- 审核（kind 决定默认的 status），书是用户自己给的，直接可用。
    kind         TEXT NOT NULL DEFAULT 'card',      -- card | book
    origin       TEXT NOT NULL DEFAULT 'web',       -- web | manual | seed | upload
    query        TEXT NOT NULL DEFAULT '',
    note         TEXT NOT NULL DEFAULT '',
    uses         INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    decided_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_kb_status ON knowledge_entries(status);
CREATE INDEX IF NOT EXISTS idx_kb_updated ON knowledge_entries(updated_at);

-- 知识库的语义索引：把「一个条目」切成「若干可召回的段落」。
--
-- 为什么不复用 studio_chunks：那张表的 run_id 是 NOT NULL 且 ON DELETE CASCADE，
-- 它的生命周期是「一次运行」。知识库是跨运行持久的，一条 entry 会被几十次运行
-- 复用；硬塞进去要么造一条假 run（删它会级联清掉全部知识向量），要么改列约束。
--
-- entry_hash 是重新索引的判据：正文没变就不必再付一次向量化调用。没有它，
-- 「保存一次就重嵌一次」会在批量编辑时把成本放大到不可接受。
--
-- 一本书有几百块，`entry_hash`（整条内容的指纹）对它太粗了：改一个标点会让
-- 几百块全部重嵌。所以另存 `chunk_hash`——块级指纹，索引时逐块比对，只重嵌
-- 真正变过的那几块。`entry_hash` 保留下来做「整条完全没动」的快路径。
--
-- chapter / heading 是书的定位信息：`chapter` 是「1921 年」这样的章节路径，
-- `heading` 是章内小标题（「中共一大召开」）。两个都要存，因为引用要能说到
-- 「第几章第几个事件」，而相邻片段合并要靠 chapter 判断是否同章。
--
-- vector 存 BLOB 而不是 JSON：2048 维向量用 JSON 要 27 KB/块，一本 300 页的书
-- 约 1000 块就是 26 MB，而每查一次都要把这些 JSON 全解析一遍（实测 517 ms，
-- 换 BLOB 后 3.3 ms）。列上保留 dim 是为了查询时能先用 SQL 过滤掉维度不符的
-- 行——换过嵌入模型之后，旧向量应当被跳过而不是被逐个反序列化后再丢弃。
CREATE TABLE IF NOT EXISTS studio_kb_chunks (
    id           TEXT PRIMARY KEY,
    entry_id     TEXT NOT NULL REFERENCES knowledge_entries(id) ON DELETE CASCADE,
    part         TEXT NOT NULL DEFAULT 'body',   -- title | point | body | chapter
    ordinal      INTEGER NOT NULL DEFAULT 0,
    text         TEXT NOT NULL,
    section      TEXT NOT NULL DEFAULT '',       -- 引用出处，如「书名 · 1921 年」
    chapter      TEXT NOT NULL DEFAULT '',       -- 章节路径，如「1921 年」
    heading      TEXT NOT NULL DEFAULT '',       -- 章内小标题，如「中共一大召开」
    entry_hash   TEXT NOT NULL DEFAULT '',
    chunk_hash   TEXT NOT NULL DEFAULT '',
    vector       BLOB NOT NULL DEFAULT x'',
    dim          INTEGER NOT NULL DEFAULT 0,
    model        TEXT NOT NULL DEFAULT '',
    updated_at   REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_entry ON studio_kb_chunks(entry_id);
CREATE INDEX IF NOT EXISTS idx_kb_chunks_dim ON studio_kb_chunks(dim);

-- Conversation transcript. One run owns one conversation: the initial brief is
-- the first user message, each agent result appends its own message, and any
-- mid-run instruction the user types becomes another user message followed by
-- the messages produced by whichever agents it triggered.
--
-- `role` distinguishes who spoke; `agent` names the producing agent for role
-- 'agent'; `step_id` links a message back to the step that produced it so the
-- UI can jump between conversation and execution trace.
CREATE TABLE IF NOT EXISTS studio_messages (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES studio_runs(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    role        TEXT NOT NULL,              -- user | agent | system
    agent       TEXT NOT NULL DEFAULT '',    -- director | script_adaptor | storyboard | ...
    kind        TEXT NOT NULL DEFAULT 'text',-- text | card | artifact | notice
    content     TEXT NOT NULL DEFAULT '',    -- human-readable line shown in the bubble
    payload     TEXT NOT NULL DEFAULT '{}',  -- structured card data for the UI
    step_id     TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_run ON studio_messages(run_id, seq);

-- Scheduled units of work. A task is not a graph position: it has an assignee,
-- a lifecycle, dependency edges and a resource-call trace, so the main agent can
-- create, retry, reassign and observe it.
--
-- Primary key is (run_id, id), not id alone: task ids are structural
-- ("t_verify", "t_write_03") and repeat across runs by design. A global key
-- would make the second run collide with the first — a bug that only appears
-- once the same workflow is run twice.
CREATE TABLE IF NOT EXISTS studio_tasks (
    id          TEXT NOT NULL,
    run_id      TEXT NOT NULL REFERENCES studio_runs(id) ON DELETE CASCADE,
    type        TEXT NOT NULL,
    assignee    TEXT NOT NULL,
    payload     TEXT NOT NULL DEFAULT '{}',
    depends_on  TEXT NOT NULL DEFAULT '[]',
    status      TEXT NOT NULL DEFAULT 'pending',
    priority    INTEGER NOT NULL DEFAULT 0,
    result      TEXT NOT NULL DEFAULT '{}',
    error       TEXT NOT NULL DEFAULT '',
    trace       TEXT NOT NULL DEFAULT '[]',
    attempts    INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL DEFAULT 0,
    started_at  REAL NOT NULL DEFAULT 0,
    finished_at REAL NOT NULL DEFAULT 0,
    label       TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (run_id, id)
);
CREATE INDEX IF NOT EXISTS idx_tasks_run ON studio_tasks(run_id, status);

-- Retrievable text chunks with their embeddings. Section/paragraph are stored
-- because a retrieved passage without a locator cannot serve as evidence.
-- Same composite key: chunk ids are positional ("chapter-001") per run.
CREATE TABLE IF NOT EXISTS studio_chunks (
    id           TEXT NOT NULL,
    run_id       TEXT NOT NULL REFERENCES studio_runs(id) ON DELETE CASCADE,
    text         TEXT NOT NULL,
    source       TEXT NOT NULL DEFAULT '',
    doc_id       TEXT NOT NULL DEFAULT '',
    section      TEXT NOT NULL DEFAULT '',
    paragraph    INTEGER NOT NULL DEFAULT 0,
    start_offset INTEGER NOT NULL DEFAULT 0,
    end_offset   INTEGER NOT NULL DEFAULT 0,
    meta         TEXT NOT NULL DEFAULT '{}',
    vector       TEXT NOT NULL DEFAULT '[]',
    dim          INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, id)
);
CREATE INDEX IF NOT EXISTS idx_chunks_run ON studio_chunks(run_id, source);

-- Cross-run verified facts. Keyed by a normalised text hash so the same fact
-- written slightly differently in another chapter still matches.
CREATE TABLE IF NOT EXISTS studio_facts (
    fact_key      TEXT PRIMARY KEY,
    text          TEXT NOT NULL,
    verdict       TEXT NOT NULL,
    confidence    REAL NOT NULL DEFAULT 0,
    evidence      TEXT NOT NULL DEFAULT '[]',
    source        TEXT NOT NULL DEFAULT '',
    hits          INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL DEFAULT 0,
    last_used_at  REAL NOT NULL DEFAULT 0
);

-- User choices that should survive across runs.
CREATE TABLE IF NOT EXISTS studio_preferences (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  REAL NOT NULL DEFAULT 0
);

-- Short-term per-run scratch: intermediate results and rework feedback.
CREATE TABLE IF NOT EXISTS studio_run_memory (
    run_id      TEXT NOT NULL REFERENCES studio_runs(id) ON DELETE CASCADE,
    step        TEXT NOT NULL,
    payload     TEXT NOT NULL DEFAULT '{}',
    updated_at  REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, step)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn
    ensure_dirs()
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()
    _conn = conn
    return conn


#: Column additions applied after the base schema. `CREATE TABLE IF NOT EXISTS`
#: cannot add a column to an existing table, so an installed database needs an
#: explicit ALTER. Each entry is (table, column, declaration); applied only when
#: the column is absent, which keeps startup idempotent.
_MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    # The history workflow persists its whole execution state so a run can be
    # inspected and resumed instead of only reporting a final status.
    ("studio_runs", "state", "TEXT NOT NULL DEFAULT '{}'"),
    # Keep the exact fetched text behind a pending knowledge draft.  A reviewer
    # must be able to compare the summary with its evidence instead of trusting a
    # word count shown only in the transient tool trace.
    ("knowledge_entries", "source_material", "TEXT NOT NULL DEFAULT '[]'"),
    # Whole-book uploads live in the same table as extracted cards. `kind`
    # separates them without splitting the retrieval path in two.
    ("knowledge_entries", "kind", "TEXT NOT NULL DEFAULT 'card'"),
    # A book needs locators: which chapter, which sub-heading inside it.
    ("studio_kb_chunks", "chapter", "TEXT NOT NULL DEFAULT ''"),
    ("studio_kb_chunks", "heading", "TEXT NOT NULL DEFAULT ''"),
    # Per-chunk fingerprint so editing one paragraph re-embeds one chunk instead
    # of all several hundred of a book.
    ("studio_kb_chunks", "chunk_hash", "TEXT NOT NULL DEFAULT ''"),
    # 有些消息要在原地推进（任务清单：分析中 → 执行中 → 完成）。给它一个稳定
    # 标识，重复写入就变成更新而不是又插一条——否则一次运行会在对话里留下
    # 十几张越来越新的同一张卡。
    ("studio_messages", "msg_key", "TEXT NOT NULL DEFAULT ''"),
    # 「停在等你回答」需要把问题、选项和续跑用的快照一起存下来：run 退出之后
    # 内存里的对象就没了，续跑只能靠这一列。
    ("studio_runs", "pending", "TEXT NOT NULL DEFAULT '{}'"),
)


def _migrate(conn: sqlite3.Connection) -> None:
    for table, column, declaration in _MIGRATIONS:
        try:
            existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        except sqlite3.Error:
            continue
        if column in existing:
            continue
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
        except sqlite3.Error:
            # A concurrent process may have added it between the check and the
            # ALTER; the schema is correct either way.
            pass

    # 有了 msg_key 之后才建这个索引：旧库上直接跑 CREATE INDEX 会因为列不存在
    # 报错，而 `executescript` 在建表那一刻就已经跑过了。
    try:
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_msg_key "
            "ON studio_messages(run_id, msg_key) WHERE msg_key <> ''"
        )
    except sqlite3.Error:
        pass


def connect() -> sqlite3.Connection:
    """Read handle for modules that only query.

    Same connection the writers use (one process, WAL), exposed so a read-only
    feature — the knowledge panel, a listing endpoint — does not have to import a
    private name to ask a question.
    """
    return _connect()


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    """Serialised write transaction."""
    with _lock:
        conn = _connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def new_id() -> str:
    return str(uuid.uuid4())


def _loads(raw: Any, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default


def _placeholders(values: list[Any]) -> str:
    """SQL `?,?,?` for an IN clause. SQLite has no array binding."""
    return ",".join("?" for _ in values)


# ============================================================================
# Runs
# ============================================================================


def create_run(brief: dict[str, Any], *, run_id: str = "") -> dict[str, Any]:
    # An explicit id lets a caller create the row and register it as live before
    # any background task starts, avoiding a window where a duplicate submission
    # could pass the "already running" check.
    run_id = run_id or new_id()
    ts = _now()
    with transaction() as conn:
        conn.execute(
            "INSERT INTO studio_runs (id, status, brief, created_at, updated_at) VALUES (?,?,?,?,?)",
            (run_id, "queued", json.dumps(brief, ensure_ascii=False), ts, ts),
        )
    return get_run(run_id)  # type: ignore[return-value]


def update_run(run_id: str, **fields: Any) -> None:
    allowed = {"status", "cost_usd", "error", "state", "pending"}
    sets: list[str] = []
    values: list[Any] = []
    for key, value in fields.items():
        if key not in allowed:
            continue
        sets.append(f"{key} = ?")
        values.append(
            json.dumps(value, ensure_ascii=False)
            if key in {"state", "pending"} and value is not None
            else value
        )
    if not sets:
        return
    sets.append("updated_at = ?")
    values.append(_now())
    values.append(run_id)
    with transaction() as conn:
        conn.execute(f"UPDATE studio_runs SET {', '.join(sets)} WHERE id = ?", values)


def get_run(run_id: str) -> dict[str, Any] | None:
    conn = _connect()
    row = conn.execute("SELECT * FROM studio_runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    # `state` is added by a migration; older databases lack the column, so read
    # it defensively rather than assuming every install has been upgraded.
    keys = row.keys()
    return {
        "id": row["id"],
        "status": row["status"],
        "brief": _loads(row["brief"], {}),
        "state": _loads(row["state"], {}) if "state" in keys else {},
        # 「在等你回答」的内容。老库没有这一列，读不到就当没有。
        "pending": _loads(row["pending"], {}) if "pending" in keys else {},
        "cost_usd": row["cost_usd"],
        "error": row["error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    conn = _connect()
    rows = conn.execute(
        "SELECT id, status, brief, cost_usd, created_at FROM studio_runs ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        {
            "id": r["id"],
            "status": r["status"],
            "brief": _loads(r["brief"], {}),
            "cost_usd": r["cost_usd"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def list_runs_summary(
    *, limit: int = 20, status: str | None = None
) -> dict[str, Any]:
    """Recent runs, light enough to list on every page load.

    Returns an excerpt rather than the whole brief: the panel shows a title line, and
    a full chapter per row would make the response large for no benefit.

    `status="done"` filters to finished runs — the default view, since a first
    screen full of failed debug attempts is noise rather than history.
    """
    conn = _connect()
    where = "WHERE status = ?" if status else ""
    params: tuple[Any, ...] = (status, limit) if status else (limit,)
    rows = conn.execute(
        f"""SELECT id, status, brief, created_at,
                   (SELECT COUNT(*) FROM studio_messages m WHERE m.run_id = r.id) AS msgs
            FROM studio_runs r {where}
            ORDER BY created_at DESC LIMIT ?""",
        params,
    ).fetchall()

    items = []
    for r in rows:
        brief = _loads(r["brief"], {})
        text = str((brief or {}).get("chapter_text") or "")
        items.append(
            {
                "id": r["id"],
                "status": r["status"],
                "title": text.replace("\n", " ").strip()[:44] or "（无正文）",
                "preview": text.replace("\n", " ").strip()[:90],
                "chars": len(text),
                "messages": r["msgs"],
                "created_at": r["created_at"],
                # Which pipeline produced this run. Two exist in the database and
                # they emit different agents and different artifacts: the doc
                # pipeline writes the eight 漫簇 agents with prompts/portraits/
                # clips, the history pipeline writes fact_checker/narrator/writer
                # and no media at all. Without this the list shows two rows that
                # look identical and only one of them opens to anything visual.
                "workflow_id": str((brief or {}).get("workflow_id") or ""),
            }
        )

    total = conn.execute("SELECT COUNT(*) AS n FROM studio_runs").fetchone()["n"]
    return {"runs": items, "total": total}


def reconcile_interrupted_runs(
    *, only: list[str] | None = None, error: str = ""
) -> list[str]:
    """Close out runs that a dead process left in a non-terminal state.

    A `queued`/`running` status is a *claim* that some process is executing that run.
    The executor's live set lives in memory, and this app runs one process, so at
    startup that claim is necessarily false: no run can be live inside a process that
    has just been created. Rows left as `running` are what made a long-dead
    conversation undeletable — the delete guard trusted the column instead of asking
    who was actually executing it.

    `only` restricts the sweep to given ids; the caller is then responsible for
    having established that none of them is live. Returns the ids that were closed.

    Safe to call repeatedly: it only touches non-terminal rows, and every run it
    touches becomes terminal.
    """
    conn = _connect()
    if only:
        rows = conn.execute(
            f"SELECT id FROM studio_runs WHERE id IN ({_placeholders(only)}) "
            f"AND status IN ('queued','running')",
            tuple(only),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id FROM studio_runs WHERE status IN ('queued','running')"
        ).fetchall()
    stale = [r["id"] for r in rows]
    if not stale:
        return []

    message = error or "服务重启，该任务已中断（执行进程已不存在）；这条记录可以删除"
    ts = _now()
    with transaction() as conn:
        # Drive the update off the explicit id list rather than off a status
        # predicate: the predicate would also match runs that failed for their own
        # reasons, and overwriting their real error with "interrupted" would erase
        # the only clue to what actually went wrong.
        conn.execute(
            f"UPDATE studio_runs SET status = 'failed', error = ?, updated_at = ? "
            f"WHERE id IN ({_placeholders(stale)})",
            (message, ts, *stale),
        )
        # Steps of those runs were left mid-flight too; a step stuck at `running`
        # reads as live progress in any step inspector, so close them as well.
        conn.execute(
            f"UPDATE studio_steps SET status = 'failed', error = ?, finished_at = ? "
            f"WHERE status = 'running' AND run_id IN ({_placeholders(stale)})",
            (message, ts, *stale),
        )

    # One system message per run, so opening the conversation explains the gap
    # instead of showing a truncated transcript with no reason.
    for run_id in stale:
        add_message(
            run_id,
            role="system",
            agent="director",
            kind="error",
            content=message,
            payload={"reconciled": True},
        )
    return stale


def backup_database(reason: str = "", keep: int = 20) -> Path | None:
    """Snapshot the database before something destructive runs.

    A hard delete is unrecoverable by design, so the only defence against "I
    deleted the wrong conversation" — or against a bug that deletes more than it
    meant to — is a copy taken beforehand. This writes one, prunes the oldest so
    the directory cannot grow without bound, and never raises: a backup failure
    must not be the reason a delete stops working.

    Uses SQLite's own backup API rather than copying the file. A file copy taken
    while WAL has uncommitted frames can be inconsistent; `.backup` produces a
    coherent snapshot of the database as it stood at that moment.

    Returns the snapshot path, or None when no backup could be made (an
    in-memory database, or a read-only data directory).
    """
    from datetime import datetime, timezone

    import sqlite3 as _sqlite

    try:
        from .config import DATA_DIR

        backups = Path(DATA_DIR) / "backups"
        backups.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        tag = "".join(c for c in reason if c.isalnum() or c in "-_")[:24] or "manual"
        target = backups / f"studio-{stamp}-{tag}.db"

        dest = _sqlite.connect(str(target))
        try:
            _connect().backup(dest)
        finally:
            dest.close()

        # Keep the newest `keep` snapshots. Sorted by name, which sorts by the
        # timestamp prefix, so no stat() calls are needed to order them.
        existing = sorted(backups.glob("studio-*.db"))
        for old in existing[:-keep] if len(existing) > keep else []:
            try:
                old.unlink()
            except OSError:
                pass
        _logger_backup(target, len(existing))
        return target
    except Exception:  # noqa: BLE001 - a backup must never break a delete
        return None


def _logger_backup(path: Path, count: int) -> None:
    """Best-effort note that a snapshot was taken. Silent on failure."""
    try:
        import logging

        logging.getLogger(__name__).info("database backed up to %s (%d kept)", path, count)
    except Exception:  # noqa: BLE001
        pass


def delete_run(run_id: str) -> dict[str, Any]:
    """Delete one run and every row that belongs to it.

    Order matters only for readability: SQLite is not enforcing foreign keys here, so
    every dependent table is named explicitly rather than relying on a cascade that
    does not exist. A missed table would leave rows that still reference a run id and
    still contain the user's text.

    A snapshot of the database is written first. The delete itself is
    irreversible — the endpoint's docstring says as much to the user — and the
    snapshot is what makes an accidental or over-broad delete recoverable.
    """
    conn = _connect()
    exists = conn.execute(
        "SELECT id FROM studio_runs WHERE id = ?", (run_id,)
    ).fetchone()
    if not exists:
        return {"ok": False, "reason": "运行不存在", "deleted": {}}

    backup = backup_database(f"delete-{run_id[:8]}")

    deleted: dict[str, int] = {}
    for table in (
        "studio_messages",
        "studio_steps",
        "studio_artifacts",
        "studio_reviews",
        "studio_tasks",
        "studio_chunks",
    ):
        present = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not present:
            continue
        has_run = any(
            c["name"] == "run_id"
            for c in conn.execute(f"PRAGMA table_info({table})").fetchall()
        )
        if not has_run:
            continue
        cur = conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
        deleted[table] = cur.rowcount

    cur = conn.execute("DELETE FROM studio_runs WHERE id = ?", (run_id,))
    deleted["studio_runs"] = cur.rowcount
    conn.commit()

    # The media files are the user's output; leaving them behind after a delete
    # would keep megabytes of orphaned assets for a conversation they removed.
    removed_files = _remove_run_files(run_id)
    if removed_files:
        deleted["files"] = removed_files
    if backup is not None:
        deleted["backup"] = str(backup)
    return {"ok": True, "deleted": deleted}


def _remove_run_files(run_id: str) -> int:
    """Delete this run's generated assets and film. Best effort, never fatal."""
    import shutil

    from .config import DATA_DIR

    count = 0
    assets = Path(DATA_DIR) / "assets" / run_id
    if assets.is_dir():
        count += sum(1 for _ in assets.rglob("*") if _.is_file())
        shutil.rmtree(assets, ignore_errors=True)
    video = Path(DATA_DIR) / "videos" / f"{run_id}.mp4"
    if video.is_file():
        video.unlink(missing_ok=True)
        count += 1
    return count


# ============================================================================
# Steps
# ============================================================================


def add_step(run_id: str, seq: int, agent: str, input_payload: dict[str, Any]) -> str:
    step_id = new_id()
    with transaction() as conn:
        conn.execute(
            "INSERT INTO studio_steps (id, run_id, seq, agent, status, input, started_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (step_id, run_id, seq, agent, "running", json.dumps(input_payload, ensure_ascii=False), _now()),
        )
    return step_id


def finish_step(
    step_id: str,
    *,
    status: str,
    output: dict[str, Any] | None = None,
    error: str = "",
    cost_usd: float = 0.0,
) -> None:
    with transaction() as conn:
        conn.execute(
            "UPDATE studio_steps SET status = ?, output = ?, error = ?, cost_usd = ?, finished_at = ? WHERE id = ?",
            (
                status,
                json.dumps(output or {}, ensure_ascii=False),
                error,
                cost_usd,
                _now(),
                step_id,
            ),
        )


def list_steps(run_id: str) -> list[dict[str, Any]]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM studio_steps WHERE run_id = ? ORDER BY seq ASC", (run_id,)
    ).fetchall()
    return [
        {
            "id": r["id"],
            "run_id": r["run_id"],
            "seq": r["seq"],
            "agent": r["agent"],
            "status": r["status"],
            "input": _loads(r["input"], {}),
            "output": _loads(r["output"], {}),
            "error": r["error"],
            "cost_usd": r["cost_usd"],
            "started_at": r["started_at"],
            "finished_at": r["finished_at"],
        }
        for r in rows
    ]


# ============================================================================
# Artifacts
# ============================================================================


def add_artifact(
    run_id: str,
    *,
    kind: str,
    label: str,
    path: str = "",
    step_id: str = "",
    meta: dict[str, Any] | None = None,
) -> str:
    artifact_id = new_id()
    with transaction() as conn:
        conn.execute(
            "INSERT INTO studio_artifacts (id, run_id, step_id, kind, label, path, meta) VALUES (?,?,?,?,?,?,?)",
            (artifact_id, run_id, step_id, kind, label, path, json.dumps(meta or {}, ensure_ascii=False)),
        )
    return artifact_id


def list_artifacts(run_id: str) -> list[dict[str, Any]]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM studio_artifacts WHERE run_id = ? ORDER BY rowid ASC", (run_id,)
    ).fetchall()
    return [
        {
            "id": r["id"],
            "run_id": r["run_id"],
            "step_id": r["step_id"],
            "kind": r["kind"],
            "label": r["label"],
            "path": r["path"],
            "meta": _loads(r["meta"], {}),
        }
        for r in rows
    ]


# ============================================================================
# Reviews
# ============================================================================


def add_review(
    run_id: str,
    *,
    step_id: str = "",
    verdict: str,
    score: float,
    issues: list[dict[str, Any]],
    summary: str = "",
) -> str:
    review_id = new_id()
    with transaction() as conn:
        conn.execute(
            "INSERT INTO studio_reviews (id, run_id, step_id, verdict, score, issues, summary, created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                review_id,
                run_id,
                step_id,
                verdict,
                score,
                json.dumps(issues, ensure_ascii=False),
                summary,
                _now(),
            ),
        )
    return review_id


def list_reviews(run_id: str) -> list[dict[str, Any]]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM studio_reviews WHERE run_id = ? ORDER BY created_at ASC", (run_id,)
    ).fetchall()
    return [
        {
            "id": r["id"],
            "run_id": r["run_id"],
            "step_id": r["step_id"],
            "verdict": r["verdict"],
            "score": r["score"],
            "issues": _loads(r["issues"], []),
            "summary": r["summary"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


# ============================================================================
# Messages (conversation transcript)
# ============================================================================


def add_message(
    run_id: str,
    *,
    role: str,
    content: str,
    agent: str = "",
    kind: str = "text",
    payload: dict[str, Any] | None = None,
    step_id: str = "",
    msg_key: str = "",
) -> str:
    """Append one message — or replace the one already carrying this `msg_key`.

    Two kinds of messages live here. Most are append-only: a stage finished, a
    review passed, one line each. A few are a *view of ongoing work* — the task
    plan card is贴出来 before the run starts and rewritten at every step. Those
    carry a `msg_key`; writing them again updates the existing row in place.

    `seq` is deliberately left alone on update: the card must stay where it was
    posted. Bumping it to "now" would make it jump to the bottom of the
    conversation every time a segment finishes, which reads as a new message
    rather than a progress update.
    """
    message_id = new_id()
    with transaction() as conn:
        if msg_key:
            row = conn.execute(
                "SELECT id FROM studio_messages WHERE run_id = ? AND msg_key = ?",
                (run_id, msg_key),
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE studio_messages SET role=?, agent=?, kind=?, content=?, "
                    "payload=?, step_id=?, created_at=? WHERE id=?",
                    (
                        role,
                        agent,
                        kind,
                        content,
                        json.dumps(payload or {}, ensure_ascii=False),
                        step_id,
                        _now(),
                        row["id"],
                    ),
                )
                return str(row["id"])
        nxt = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM studio_messages WHERE run_id = ?",
            (run_id,),
        ).fetchone()["n"]
        conn.execute(
            "INSERT INTO studio_messages (id, run_id, seq, role, agent, kind, content, payload, step_id, msg_key, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                message_id,
                run_id,
                nxt,
                role,
                agent,
                kind,
                content,
                json.dumps(payload or {}, ensure_ascii=False),
                step_id,
                msg_key,
                _now(),
            ),
        )
    return message_id


def get_message_by_key(run_id: str, msg_key: str) -> dict[str, Any] | None:
    """按 msg_key 取一条原地推进的消息（清单卡、判定卡都靠它找最新状态）。"""
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM studio_messages WHERE run_id = ? AND msg_key = ?",
        (run_id, msg_key),
    ).fetchone()
    if row is None:
        return None
    return {
        "id": row["id"],
        "run_id": row["run_id"],
        "seq": row["seq"],
        "role": row["role"],
        "agent": row["agent"],
        "kind": row["kind"],
        "content": row["content"],
        "payload": _loads(row["payload"], {}),
        "step_id": row["step_id"],
        "created_at": row["created_at"],
    }


def list_messages(run_id: str) -> list[dict[str, Any]]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM studio_messages WHERE run_id = ? ORDER BY seq ASC", (run_id,)
    ).fetchall()
    return [
        {
            "id": r["id"],
            "run_id": r["run_id"],
            "seq": r["seq"],
            "role": r["role"],
            "agent": r["agent"],
            "kind": r["kind"],
            "content": r["content"],
            "payload": _loads(r["payload"], {}),
            "step_id": r["step_id"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def reset_db_for_tests(path: Path | None = None) -> None:
    """Close and forget the cached connection (used by tests)."""
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None
