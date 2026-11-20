"""知识库：已经被确认过的知识，存在本地，下次直接命中。

设计上只有一条主线：**一个条目同时记录「内容」和「是否被人工接受」**。

- `body` / `key_points` / `sources` 是内容，来自检索 + 摘要。
- `status` 是决定，`pending`（待审核）/ `approved`（已入库）/ `rejected`（已放弃）。

两者分开是必须的：本次运行查到的东西要立刻能用（否则这一镜没材料可用），但
它还没经过人判断，所以是 `pending`；人点一下「入库」才变成 `approved`。如果把
「能用」和「已接受」合成一个字段，就只有两种坏选择——要么未审核的内容立刻变成
可信知识，要么当前这次运行先卡住等人。

检索匹配用字符二元组（bigram），因为中文没有空格可切：查「党的诞生」时，
「中国共产党成立」这条应该命中，靠的是「成立/诞生」这类近义线索之外的共同字面
（「中国」「党的」）。这当然不如向量检索，但它是可解释的——命中理由能写进日志，
而一个解释不了为什么命中的知识库，没人敢用。
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from .. import db
# 填充词表定义在 `query.py` 里，因为向量检索那条路也要用它——两处各写一份迟早
# 会分叉，而分叉的表现是「关键词那条路清掉了填充词、向量那条路没清」，从输出上
# 完全看不出来，只能靠对比检索结果才发现。
from .query import STOP_PHRASES

#: Status values, in the order a row moves through them.
PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"

#: 条目类型。`card` 是 AI 从权威来源摘要出的一条主题卡片，`book` 是用户整本
#: 上传的资料。两者共用一张表和一条检索路径——对检索而言它们都是「一段需要
#: 出处的可召回文字」——区别只在来源与是否需要人工审核。
CARD = "card"
BOOK = "book"
KIND_LABEL = {CARD: "主题卡片", BOOK: "整本资料"}

STATUS_LABEL = {PENDING: "待审核", APPROVED: "已入库", REJECTED: "已放弃"}

#: How much of the query must appear in an entry before it counts as a hit.
#: Too low and every topic "hits" the one entry that happens to share a common
#: word (`中国`); too high and a paraphrase of a stored topic misses and triggers
#: a pointless web round-trip. 0.45 was measured against the entries this studio
#: actually produces: two of three bigrams matching is a real match, one is noise.
MATCH_THRESHOLD = 0.45

#: Words that carry no topic signal. They appear in almost every request
#: ("帮我做一个关于……的视频"), so leaving them in the query made every request
#: 0.6-similar to every entry. 表本身在 `query.py`，见上面 import 处的说明。


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pairs(text: str) -> set[str]:
    """Character bigrams, with punctuation and filler removed.

    Chinese has no word boundaries, so a bigram is the cheapest unit that still
    carries meaning: 「党的诞生」 → {党的, 诞的生? no} — actually {党的, 的诞, 诞生}.
    Filler phrases are stripped first, because they are long enough to dominate
    the overlap score on their own.
    """
    cleaned = text or ""
    for phrase in STOP_PHRASES:
        cleaned = cleaned.replace(phrase, " ")
    cleaned = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", " ", cleaned)
    grams: set[str] = set()
    for chunk in cleaned.split():
        if len(chunk) == 1:
            grams.add(chunk)
            continue
        for i in range(len(chunk) - 1):
            grams.add(chunk[i : i + 2])
    return grams


def _loads(value: Any, default: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value or "")
    except (TypeError, ValueError):
        return default


def _row_to_entry(row: Any, *, include_material: bool = True) -> dict[str, Any]:
    keys = row.keys()
    source_material = _loads(row["source_material"], []) if "source_material" in keys else []
    if not include_material:
        source_material = [
            {
                "url": str(item.get("url") or ""),
                "title": str(item.get("title") or ""),
                "chars": int(item.get("chars") or len(str(item.get("text") or ""))),
            }
            for item in source_material if isinstance(item, dict)
        ]
    return {
        "id": row["id"],
        "title": row["title"],
        "category": row["category"],
        "period": row["period"],
        "subtitle": " · ".join(x for x in (row["category"], row["period"]) if x),
        "body": row["body"],
        "kind": (row["kind"] if "kind" in keys else CARD) or CARD,
        "kind_label": KIND_LABEL.get(row["kind"] if "kind" in keys else CARD, "主题卡片"),
        "key_points": _loads(row["key_points"], []),
        "sources": _loads(row["sources"], []),
        "source_material": source_material,
        "status": row["status"],
        "status_label": STATUS_LABEL.get(row["status"], row["status"]),
        "origin": row["origin"],
        "query": row["query"],
        "note": row["note"],
        "uses": int(row["uses"] or 0),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "decided_at": row["decided_at"],
    }


def save(entry: dict[str, Any]) -> dict[str, Any]:
    """Insert or update one entry. Returns it as stored.

    `kind` 只在新建时写入，更新时不覆盖：它决定这条数据的生命周期（书直接
    可用、卡片要审核），被一次内容保存静默改掉会让审核记录失去意义。
    """
    entry_id = str(entry.get("id") or uuid.uuid4().hex[:12])
    now = _now()
    kind = str(entry.get("kind") or CARD)
    if kind not in (CARD, BOOK):
        kind = CARD
    with db.transaction() as conn:
        existing = conn.execute(
            "SELECT id, status, created_at FROM knowledge_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        values = (
            entry_id,
            str(entry.get("title") or "").strip() or "未命名条目",
            str(entry.get("category") or "").strip(),
            str(entry.get("period") or "").strip(),
            str(entry.get("body") or "").strip(),
            json.dumps(list(entry.get("key_points") or []), ensure_ascii=False),
            json.dumps(list(entry.get("sources") or []), ensure_ascii=False),
            json.dumps(list(entry.get("source_material") or []), ensure_ascii=False),
            str(entry.get("status") or PENDING),
            str(entry.get("origin") or "web"),
            str(entry.get("query") or ""),
            str(entry.get("note") or ""),
        )
        if existing is None:
            conn.execute(
                "INSERT INTO knowledge_entries"
                " (id, title, category, period, body, key_points, sources, source_material,"
                "  status, origin, query, note, kind, uses, created_at, updated_at, decided_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,0,?,?,NULL)",
                values + (kind, now, now),
            )
        else:
            # Status is not overwritten by a content update: re-summarising an
            # entry must not silently un-approve it (or approve it).
            conn.execute(
                "UPDATE knowledge_entries SET title=?, category=?, period=?, body=?,"
                " key_points=?, sources=?, source_material=?, origin=?, query=?, note=?, updated_at=?"
                " WHERE id=?",
                (
                    values[1], values[2], values[3], values[4], values[5], values[6], values[7],
                    values[9], values[10], values[11], now, entry_id,
                ),
            )
    return get(entry_id) or {}


def get(entry_id: str) -> dict[str, Any] | None:
    conn = db.connect()
    row = conn.execute(
        "SELECT * FROM knowledge_entries WHERE id = ?", (str(entry_id),)
    ).fetchone()
    return _row_to_entry(row) if row else None


def list_entries(
    *, status: str = "", q: str = "", kind: str = "", limit: int = 100
) -> list[dict[str, Any]]:
    conn = db.connect()
    sql = "SELECT * FROM knowledge_entries"
    clauses: list[str] = []
    params: list[Any] = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if kind:
        clauses.append("kind = ?")
        params.append(kind)
    if q.strip():
        # 卡片的正文照旧参与字面搜索（一段一千字，扫起来很快）。书不参与：
        # 一本六万字，`body LIKE '%词%'` 每次都要扫全文，而书里的内容本来就
        # 走语义检索，字面搜书名和要点就够定位了。
        clauses.append(
            "(title LIKE ? OR key_points LIKE ? OR (kind = 'card' AND body LIKE ?))"
        )
        like = f"%{q.strip()}%"
        params.extend([like, like, like])
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    params.append(max(1, min(500, limit)))
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_entry(r, include_material=False) for r in rows]


#: 正文命中打七折。标题命中说明这条条目**讲的是这件事**，正文命中只说明它
#: **提到过这件事**——《中国共产党的诞生》正文里有一大代表名单，于是「毛泽东」
#: 「毛泽东生平」都能把它搜出来，然后导演以为找到了毛泽东的材料。
#: 两个分量分开算、正文打折，是这条检索唯一能区分"讲这件事"和"提到这件事"的地方。
BODY_WEIGHT = 0.7


def _chunk_rows(entry_id: str) -> list[dict[str, Any]]:
    """一本书的全部切块，按原文顺序。

    切块是索引阶段就切好的（`vector` 建索引时写进 `studio_kb_chunks`），这里只读
    不拼：拼回整本再做字面匹配，等于把粒度退回「一篇文章」，见 `_best_chunk`。

    排序用 `rowid` 而不是 `ordinal`：`ordinal` 是**章内**序号（第一章的第 3 块和
    第九章的第 3 块都是 3），拿它排全场会把各章交错起来，于是「先出现的那块」
    变成一个随机结果——实测查「中华人民共和国成立」，1949 年那一块和 2009 年
    那条纪念文章同分，交错排序让后者排到了前面。
    """
    conn = db.connect()
    rows = conn.execute(
        "SELECT text, section, chapter, heading, part FROM studio_kb_chunks"
        " WHERE entry_id = ? ORDER BY rowid",
        (str(entry_id),),
    ).fetchall()
    return [dict(row) for row in rows]


#: 块级命中打八折。块只有一两百字，五六个二元组重合就足以说明这块讲的是这件事；
#: 但「正文里提到过」仍然不该和「标题就是它」同分。
CHUNK_WEIGHT = 0.8

#: 块级命中至少要两个二元组重合。只中一个的话，中文里几乎每一段都能凑出来——
#: 「成立」「人民」这种词在六万字里到处都是，那不叫匹配。
CHUNK_MIN_SHARED = 2

#: 目录块（章节目录、要点列表）不算正文命中。
#:
#: 这两类块把一整年的若干事件压成一行（「1927 年：四一二政变、南昌起义、秋收
#: 起义」），任何一件事件名在这里都能拿满分，然后下游拿到的是目录而不是叙述。
#: 只有整本书都是这种结构、一块正文都没有时，才退回去用它们——宁可有目录可指，
#: 也好过一本有正文的书完全搜不到。
_DIRECTORY_PARTS = frozenset({"chapter", "point", "title"})


#: 章标题命中的份量。比小标题低一点、比正文高一点。
#:
#: 它是「按年份查」的唯一出路：查「1949」时，1949 年那一章的每一块正文里都有
#: 这四个数字，八十年后的读者纪念文章里也有——纯按正文比，1948 年那一块（正文
#: 里写着「1949 年」）能排到第一。章标题命中说明这一章讲的就是这个年份。
CHAPTER_WEIGHT = 0.9


def _best_chunk(entry_id: str, grams: set[str]) -> dict[str, Any] | None:
    """在这本书的切块上找最像的一块，连它的定位一起返回。

    粒度是这件事的全部关键。整本六万字上做字面匹配，几乎任何两字组合都在里面
    出现过，重合率恒等于 1——那个判断是对的，但它推出的是「别把整本当一篇文章
    比」，不是「干脆别比」。一块一两百字就不同了：同时出现「中华/华人/人民/
    民共/共和/和国/国成/成立」这八个二元组，说明这一块讲的就是那件事。

    三个分量分开算、取最高，而不是加起来：

    - 小标题命中 → 满分。这块讲的就是这件事。
    - 章标题命中 → 0.9。这一章讲的就是这个年份／这个阶段。
    - 正文命中 → 打八折。这是「提到过」，不是「讲的就是它」。

    加起来的话，一篇 2019 年的纪念文章（正文里逐字引了「宣告中华人民共和国成立」
    70 周年）会靠堆字数压过 1949 年那一块真正的原始记录——分数高的是引用得多的
    那一段，不是讲这件事的那一段。
    """
    if not grams:
        return None
    rows = _chunk_rows(entry_id)
    if not rows:
        return None
    bodies = [row for row in rows if str(row.get("part") or "") not in _DIRECTORY_PARTS]
    #: 小标题和章标题一起参与匹配，但不拼进返回的正文——`chunk` 是给下游当依据
    #: 用的，混入标题会让「原文」这两个字失去意义。
    best: dict[str, Any] | None = None
    for row in bodies or rows:
        body = str(row.get("text") or "")
        heading = str(row.get("heading") or "")
        chapter = str(row.get("chapter") or "")
        if not body and not heading:
            continue
        in_heading = grams & _pairs(heading)
        in_chapter = grams & _pairs(chapter)
        in_body = grams & _pairs(body)
        shared = in_heading | in_chapter | in_body
        if len(shared) < CHUNK_MIN_SHARED:
            continue
        score = max(
            len(in_heading) / len(grams),
            len(in_chapter) / len(grams) * CHAPTER_WEIGHT,
            len(in_body) / len(grams) * CHUNK_WEIGHT,
        )
        candidate = {
            "score": score,
            "_shared": shared,
            "text": body,
            "section": str(row.get("section") or ""),
            "chapter": chapter,
            "heading": heading,
            "part": str(row.get("part") or ""),
        }
        # 同分时留先出现的那一块：一本按年份编排的书里，先出现的那块通常就是
        # 事件本身，后面的是在复述它。
        if best is None or score > best["score"]:
            best = candidate
    return best


def search(query: str, *, limit: int = 3) -> list[dict[str, Any]]:
    """Best local matches for `query`, with the reason they matched.

    Rejected entries are excluded — "已放弃" is a decision that this text is not
    ours to reuse, and re-serving it on the next run would silently undo the
    reviewer's call.

    Title and body are scored apart, not concatenated. 拼在一起时，正文里顺带提
    到的词和标题里的主题词权重一样：《中国共产党的诞生》因为列了一大的出席代表，
    对「毛泽东」「毛泽东生平」都是最高分命中，导演于是拿"党的诞生"去拍"毛主席
    生平"，而它在界面上显示的是「知识库命中」——看起来完全正常。

    **整本资料按「块」参与字面匹配，不按整本。** 这件事上一版做反了：为了绕开
    「六万字上任何两字组合都命中、重合率恒等于 1」，干脆让书的正文不参与字面
    匹配，只比书名和章节标题。可章节标题是年份——「1921 年」——于是用户的说法
    和书里的说法只要不是同一个词，字面层就恒为 0：

        「建国」            → 0 条
        「中华人民共和国成立」 → 0 条   （而它逐字就是 1949 那一块的小标题）
        「开国大典」         → 0 条   （这个说法书里确实没有）

    一个本来能答的问题被报成「库里没有」，用户只好去联网。现在改成逐块比较：
    命中的是「哪一块」，返回的也是那一块和它的定位（第几章、哪件事），而不是
    整本六万字。字面重合率在一两百字的块上重新有了信息量。
    """
    grams = _pairs(query)
    if not grams:
        return []
    # 两个字的查询只剩一个二元组，任何提到这两个字的条目都能拿满分。这不是
    # 检索，这是撞词。短查询只认标题。
    title_only = len(grams) <= 2
    scored: list[tuple[float, dict[str, Any], list[str]]] = []
    for entry in list_entries(limit=500):
        if entry["status"] == REJECTED:
            continue
        title_grams = _pairs(
            " ".join([entry["title"], entry["category"], entry["period"]])
        )
        chunk_hit: dict[str, Any] | None = None
        if entry.get("kind") == BOOK:
            # 章节标题只是这本书的「目录」：它说明书里讲了哪些年，说不清哪一段
            # 讲了这件事。两个都要看——目录管粗定位，块管「命中哪一段」。
            body_grams = _pairs(" ".join(str(p) for p in entry["key_points"]))
            if not title_only:
                chunk_hit = _best_chunk(str(entry["id"]), grams)
        else:
            body_grams = _pairs(
                " ".join([entry["body"], " ".join(str(p) for p in entry["key_points"])])
            )
        if not title_grams and not body_grams and not chunk_hit:
            continue
        title_hits = grams & title_grams
        shared = title_hits | (grams & body_grams)
        if chunk_hit:
            shared = shared | chunk_hit["_shared"]
        if not shared:
            continue
        title_score = len(title_hits) / len(grams)
        body_score = len(grams & body_grams) / len(grams)
        score = title_score if title_only else max(title_score, body_score * BODY_WEIGHT)
        if chunk_hit:
            score = max(score, chunk_hit["score"])
        if score >= MATCH_THRESHOLD:
            scored.append((score, entry, sorted(shared)[:4], chunk_hit))
    scored.sort(key=lambda item: (-item[0], item[1]["updated_at"]))
    out: list[dict[str, Any]] = []
    for score, entry, shared, chunk_hit in scored[:limit]:
        entry = dict(entry)
        match: dict[str, Any] = {"score": round(score, 3), "shared": shared}
        if chunk_hit:
            # 书必须带「命中哪一块」：下游写脚本时要说出依据在哪一章哪件事，
            # 只说书名等于没说。卡片没有这一层，它的 body 本身就是那一段。
            match.update({
                "chunk": chunk_hit["text"],
                "section": chunk_hit["section"],
                "chapter": chunk_hit["chapter"],
                "heading": chunk_hit["heading"],
                "part": chunk_hit["part"],
            })
        entry["match"] = match
        out.append(entry)
    return out


def set_kind(entry_id: str, kind: str) -> bool:
    """强制改写条目类型，返回是否真的改了。

    为什么需要这个函数：`save()` 的 UPDATE 分支**刻意不更新 kind**——内容更新
    不该改掉一条记录的生命周期（卡片要审核、书直接可用），否则重新摘要一次就会
    把审核状态无声地掀翻。

    但那条规则有个死角：一条记录如果在写入时就被判定成了错误的类型，之后无论
    重传多少次都修不回来——因为每次同名覆盖走的都是 UPDATE，kind 原样保留。
    更糟的是它会自我强化：kind=card 的书会被按卡片切分（一百个「1921 年」变成
    独立向量），检索质量塌掉，而 `health` 只能报「结构异常」，用户点「重建索引」
    也好不了，因为重建时读到的 kind 还是 card。

    这个函数是那个死角的出口。只给明确知道正确类型的调用方用——目前只有
    `import_book`，它从上传路径来，一定知道自己在收整本资料。
    """
    if kind not in (CARD, BOOK):
        return False
    with db.transaction() as conn:
        row = conn.execute(
            "SELECT kind FROM knowledge_entries WHERE id = ?", (str(entry_id),)
        ).fetchone()
        if row is None or str(row["kind"] or CARD) == kind:
            return False
        conn.execute(
            "UPDATE knowledge_entries SET kind = ?, updated_at = ? WHERE id = ?",
            (kind, _now(), str(entry_id)),
        )
    return True


def decide(entry_id: str, status: str) -> dict[str, Any] | None:
    """Approve or reject an entry. The content is untouched."""
    if status not in (APPROVED, REJECTED, PENDING):
        raise ValueError(f"未知状态：{status}")
    with db.transaction() as conn:
        conn.execute(
            "UPDATE knowledge_entries SET status=?, decided_at=?, updated_at=? WHERE id=?",
            (status, _now() if status != PENDING else None, _now(), str(entry_id)),
        )
    return get(entry_id)


def touch(entry_id: str) -> None:
    """Record one use. Counted apart from approval so「用了几次」可查。"""
    with db.transaction() as conn:
        conn.execute(
            "UPDATE knowledge_entries SET uses = uses + 1 WHERE id = ?", (str(entry_id),)
        )


def delete(entry_id: str) -> None:
    with db.transaction() as conn:
        conn.execute("DELETE FROM knowledge_entries WHERE id = ?", (str(entry_id),))


def stats() -> dict[str, int]:
    conn = db.connect()
    row = conn.execute(
        "SELECT"
        " SUM(CASE WHEN status='approved' THEN 1 ELSE 0 END) AS approved,"
        " SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending,"
        " SUM(CASE WHEN status='rejected' THEN 1 ELSE 0 END) AS rejected,"
        " SUM(CASE WHEN kind='book' THEN 1 ELSE 0 END) AS books,"
        " COUNT(*) AS total FROM knowledge_entries"
    ).fetchone()
    return {
        key: int(row[key] or 0)
        for key in ("approved", "pending", "rejected", "books", "total")
    }


def to_source_list(items: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """Normalise source records to the four fields the UI renders.

    Every field is kept even when empty: the panel shows「来源机构 / 原始标题 /
    原文链接」as labelled rows, and a missing key would render as a blank line with
    no label, which reads as a layout bug rather than missing data.
    """
    out: list[dict[str, str]] = []
    for item in items or []:
        if isinstance(item, str):
            item = {"title": item}
        out.append(
            {
                "kind": str(item.get("kind") or "参考资料"),
                "org": str(item.get("org") or ""),
                "title": str(item.get("title") or ""),
                "url": str(item.get("url") or ""),
            }
        )
    return out
