"""知识库：本地存过的知识，加上「没有就出去找」的那条路。

调用顺序是固定的，而且顺序本身就是设计：

1. 先查本地库。只有带白名单权威网址的条目才算命中；否则继续外部检索。
2. 没有才出去找，**只认权威站点**（白名单见 `authority.py`）。
3. 找到的内容摘要成固定格式，**立刻**以 `pending` 入库——本次运行要用它，但
   它还没经过人判断，所以不能直接算「已入库」。
4. 人确认后才变 `approved`；此后再查这个主题，第 1 步就能命中，不会再联网。

权威站点没有可抓取正文时直接返回“没有找到”并中断；禁止用模型自身知识补写史料。

第 3 步是关键：只查不存，等于每次运行都重新搜一遍，这个库就永远长不大；而查到
就自动转正，又等于让搜索引擎替人做判断。所以「能用」和「已接受」分成两个状态。
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from . import document, store, vector
from .authority import describe_allowed, is_authoritative, site_label
from .query import expand, strip_filler
from .websearch import FetchError, SearchError, SearchResult, search, search_available
from .summarize import summarize


class ResearchError(RuntimeError):
    """The knowledge base could not answer, with the reason a user can act on."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str = "research_error",
        details: list[dict[str, Any]] | None = None,
        action: str = "",
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.details = details or []
        self.action = action


def _has_authoritative_source(entry: dict[str, Any]) -> bool:
    """True only when an entry cites at least one whitelisted source URL.

    整本上传的资料是例外，判据换成「谁给的」而不是「哪个网址」：

    白名单拦的是**模型自己从网上摘要出来的内容**——它可能摘自营销号、论坛
    或者根本是幻觉，所以要求有一个可核对的权威出处。书不是这个：用户主动把
    一份资料交给系统，这个动作本身就是「我认可这份材料」的判断。要求一本书
    必须引用政府网站，等于说用户只能上传政府网站，那这个功能就没法用了。

    例外只给 `kind=book`，卡片一律走白名单。放宽的边界必须窄到能一句话说清，
    否则「权威性」这个概念会在半年后变成一个谁也说不明白的东西。
    """
    if str(entry.get("kind") or "card") == store.BOOK:
        return True
    return any(
        is_authoritative(str(source.get("url") or ""))
        for source in (entry.get("sources") or [])
        if isinstance(source, dict)
    )


_BIO_SUFFIX_RE = re.compile(
    r"(?:的)?(?:生平|简介|人物介绍|事迹|年谱|一生|传记|小传|"
    r"生平经历|生平事迹|生平简介|革命经历|光辉一生|革命生涯)$"
)
#: 修饰语： 「邓小平光辉的一生」「伟人的一生」。留着它们会让主题变成「邓小平光辉」，
#: 而条目标题里没有这两个字，于是正确的条目会被误判成"主题不符"。
_BIO_QUALIFIER_RE = re.compile(r"(?:的)?(?:光辉|伟大|传奇|不朽|革命|战斗)$")
_BIO_MARKERS: dict[str, tuple[str, ...]] = {
    "early": ("出生", "生于", "少年", "青年", "求学", "留学", "赴法", "童年"),
    "revolution": ("革命", "起义", "长征", "抗日", "解放战争", "淮海战役", "渡江战役"),
    "construction": ("新中国成立", "建国", "社会主义革命", "社会主义建设", "西南局"),
    "reform": ("改革开放", "十一届三中全会", "现代化建设", "邓小平理论"),
    "late": ("南方谈话", "南巡", "逝世", "去世", "1992年", "1997年"),
}


def _biography_subject(query: str) -> str:
    text = (query or "").strip()
    if not any(
        word in text
        for word in ("生平", "年谱", "一生", "人物介绍", "简介", "传记", "小传", "事迹")
    ):
        return ""
    stem = _BIO_SUFFIX_RE.sub("", text).strip()
    return _BIO_QUALIFIER_RE.sub("", stem).strip() or stem


def _biography_coverage(text: str) -> set[str]:
    body = text or ""
    return {
        stage for stage, markers in _BIO_MARKERS.items()
        if any(marker in body for marker in markers)
    }


#: 生平材料至少要覆盖几个阶段才算「能拍」。
#:
#: 这曾经要求「早年 + 革命 + 改革」三者齐全，结果一篇 8763 字的《求是》长文
#: ——早年、革命、建设、改革、晚年里齐了四个——被判成"不是完整生平"，整条检索
#: 失败。这不是严格，这是把好材料挡在门外：一篇权威长文讲的确实是这个人，就足以
#: 支撑一分钟视频，缺一个少年时代不该等于没有素材。
#:
#: 门槛保留在「至少三个阶段」：站点首页、投稿指南、数据库导航页都是零阶段，
#: 一篇与人物无关的专题文章也是零阶段，两者仍然过不去。
_MIN_BIO_STAGES = 3

#: 主角度量：正文里提到这个名字的次数，至少要达到文中最高频姓名的一半。
#:
#: 只判「出现过」是不够的。邓小平那篇文章里「毛泽东」出现 14 次
#: （「邓小平同志作为毛泽东同志的亲密战友」），于是「毛泽东生平」也能通过
#: ——而这正是这个项目已经被咬过一次的错：库里没有毛泽东，却报「知识库命中」。
#: 用相对频次判主次，比加一份人名表稳：它不依赖知不知道这个人是谁。
_SUBJECT_DOMINANCE = 0.5

#: 中文姓名的常见长度。取正文里出现过的二到四字组合里最高频的那个作基准，
#: 只需要一个数量级，不需要真的做分词。
_NAME_LIKE_RE = re.compile(r"[\u4e00-\u9fff]{2,4}")


def _mentions(body: str, term: str) -> int:
    return body.count(term) if term else 0


def _subject_dominates(body: str, subject: str) -> bool:
    """Is `subject` central to this text, or merely mentioned inside it?

    Compares the subject's frequency with the most frequent other name-like
    string in the page. A biography of 邓小平 mentions him 75 times and most
    others a handful; an article that merely concerns a period he took part in is
    the other way round. No absolute-frequency shortcut: 14 mentions of 毛泽东 in
    a piece that is *about* 邓小平 (who appears 75 times) is exactly the case
    this has to reject.
    """
    if not subject or subject not in body:
        return False
    own = _mentions(body, subject)
    if own == 0:
        return False
    others = 0
    for candidate in set(_NAME_LIKE_RE.findall(body)):
        if len(candidate) < 2 or candidate == subject:
            continue
        others = max(others, _mentions(body, candidate))
    if others == 0:
        return True
    return own >= others * _SUBJECT_DOMINANCE


def _material_ready(query: str, pages: list[str]) -> bool:
    """Whether fetched text covers the requested shape, not merely the subject."""
    if not pages:
        return False
    subject = _biography_subject(query)
    if not subject:
        return True
    body = "\n".join(pages)
    # 主题得是这篇材料的主角，而不只是被提到的一个人。
    if not _subject_dominates(body, subject):
        return False
    return len(_biography_coverage(body)) >= _MIN_BIO_STAGES


def _draft_quality_issues(query: str, entry: dict[str, Any]) -> list[str]:
    """Deterministic gate before a machine draft may be used or persisted."""
    subject = _biography_subject(query)
    if not subject:
        return []
    title = str(entry.get("title") or "")
    body = str(entry.get("body") or "")
    issues: list[str] = []
    if subject not in title and subject not in body[:300]:
        issues.append(f"正文主题不是「{subject}」")
    if len(body) < 240:
        issues.append(f"生平正文过短（{len(body)} 字，至少需要 240 字）")
    coverage = _biography_coverage(body)
    # 与 `_material_ready` 用同一个下限。这里曾列出全部四个阶段且缺一不可，
    # 于是即使材料已经过门、摘要也写全了它，仍会被"缺少阶段：早年"打回——
    # 门的宽严必须一致，否则前一道放行的东西会被后一道以同样的理由拦住。
    if len(coverage) < _MIN_BIO_STAGES:
        labels = {
            "early": "早年/求学", "revolution": "革命经历", "construction": "建设时期",
            "reform": "改革开放", "late": "晚年/逝世",
        }
        present = "、".join(labels[x] for x in labels if x in coverage) or "无完整阶段"
        issues.append(
            f"覆盖的人生阶段不足（现有：{present}，至少需要 {_MIN_BIO_STAGES} 个）"
        )
    return issues


#: 材料到这个体量就算「详实」，摘要不该只剩几句话。
_RICH_MATERIAL_CHARS = 3000
#: 库里存着的正文短于这个数，就当作「上次没抓全」，刷新时必须重抓同一 URL。
#: 一个只有几百字的正文撑不起「完整生平」，而它正是触发刷新的原因。
_STORED_MATERIAL_MIN = 1200
#: 材料详实时，摘要的绝对下限。12000 字材料出了一条 121 字的条目，这个条目
#: 在界面上看起来完全正常（有标题、有正文、有来源），但它撑不起一分钟视频——
#: 后面的分镜只能反复拍同一句话。条目一旦以 pending 落库，下一次检索还会把它
#: 当素材用，所以必须在**入库前**拦住，而不是等人在面板里发现它太短。
_RICH_BODY_MIN = 300
#: 材料再多，摘要也不该缩到材料的这个比例以下。取 2%：12000 字材料对应 240 字，
#: 一部详实的党史长文摘要成 240 字已经是很狠的压缩，再少就不是摘要而是标题了。
_BODY_RATIO_MIN = 0.02


def _summary_quality_issues(
    query: str, draft: Any, material: str
) -> list[str]:
    """Does the summary actually carry what the fetched material contains?

    Two different failures live here, and the older gate only caught the first:

      1. **Topic** — the entry is about something else (checked in
         `_draft_quality_issues`). 「邓小平生平」 must not be answered with an
         article about 中国特色社会主义事业.
      2. **Volume** — the entry *is* about the subject but has been compressed
         past usefulness. This gate is material-length aware on purpose: a short
         source is *supposed* to yield a short entry (the prompt says so), so a
         fixed 300-character floor would reject honest summaries of thin pages.
    """
    issues = list(
        _draft_quality_issues(
            query,
            {"title": draft.title, "body": draft.body, "key_points": draft.key_points},
        )
    )
    body = str(getattr(draft, "body", "") or "")
    source_chars = len(material or "")
    if not body.strip():
        issues.append("摘要正文为空")
        return issues
    if source_chars >= _RICH_MATERIAL_CHARS:
        floor = max(_RICH_BODY_MIN, int(source_chars * _BODY_RATIO_MIN))
        if len(body) < floor:
            issues.append(
                f"材料 {source_chars} 字，摘要只有 {len(body)} 字（至少应有 {floor} 字）"
            )
    # 材料本身很短时不设字数下限：摘要提示词明确要求「材料只有概要就写短」，
    # 给薄材料加一条字数门槛，等于逼模型注水——那正是这份提示词在防的事。
    points = [str(p).strip() for p in (getattr(draft, "key_points", None) or []) if str(p).strip()]
    if source_chars >= _RICH_MATERIAL_CHARS and len(points) < 3:
        issues.append(f"关键知识点只有 {len(points)} 条，材料足够时可核对点应不少于 3 条")
    return issues


def _research_queries(text: str, expanded: list[str]) -> list[str]:
    """Produce bounded alternate searches after stale authoritative links.

    Official-site indexes contain many moved pages.  Repeating the exact query
    only returns the same 404s, so a biographical request also tries the forms
    official obituaries, chronologies and profile pages commonly use.
    """
    stem = re.sub(r"(?:的)?(?:生平|简介|人物介绍|事迹|年谱)$", "", text).strip()
    # First keep the best natural-language form.  If that index is stale, switch
    # to domain-constrained searches instead of asking the same index for another
    # generic wording (which tends to return the same deleted People/Xinhua URLs).
    raw = [expanded[1] if len(expanded) > 1 else text, text]
    if stem:
        raw.extend((
            f"site:12371.cn {stem}",
            f"site:gov.cn {stem}",
            f"site:dswxyjy.org.cn {stem}",
            f"site:news.cn {stem}",
        ))
    out: list[str] = []
    for query in raw:
        query = str(query or "").strip()
        if query and query not in out:
            out.append(query)
    return out[:6]


def _apply_gate(
    query: str, found: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str]:
    """Retrieval candidates → the ones allowed to be used as evidence.

    闸门只有这一份，主题检索和语义检索都必须过它。两套检索各自实现一遍闸门，
    迟早会分叉——而分叉的后果是「某条内容从语义层绕过了权威性校验」，这是最
    难在测试里发现的一类回归，因为它只在特定的召回路径上才发生。

    Returns `(passed, blocked_entry_id)`. The id is the entry that failed on
    quality while still `pending`, which the caller turns into a refresh target.
    """
    hits: list[dict[str, Any]] = []
    blocked = ""
    for hit in found:
        # Notes without a whitelisted URL may stay visible in the knowledge panel,
        # but strict historical generation must not treat them as evidence.
        if not _has_authoritative_source(hit):
            continue
        issues = _draft_quality_issues(query, hit)
        # Human approval is an explicit override. Machine-created pending
        # drafts must pass the deterministic gate before reuse.
        if issues and hit.get("status") != store.APPROVED:
            blocked = blocked or str(hit.get("id") or "")
            continue
        hits.append(hit)
    return hits, blocked


async def research(
    query: str,
    *,
    allow_web: bool = True,
    max_pages: int = 4,
    entry_id: str = "",
) -> dict[str, Any]:
    """Answer `query` from the knowledge base, or go and build the entry.

    Returns `{"entry", "source", "trace"}` where `source` is `kb` or `web`.
    A `kb` hit costs one local query; a `web` hit costs a search, a few fetches
    and one summarisation call, so the difference is worth recording in the trace.

    `entry_id` forces a refresh of that one entry instead of creating a new one.
    The knowledge panel uses it to repair a draft that was stored before the
    summary gate existed: without it, "重新检索" would leave the thin entry in
    place and add a second entry beside it.
    """
    text = (query or "").strip()
    if not text:
        raise ResearchError("检索词为空")

    # 用户的说法和库里的标题常常一个字都不重叠（「党是怎么出来的」vs
    # 《中国共产党的诞生》），所以先查原句，再查改写过的那几种说法。原句优先：
    # 它命中时最精确，改写只在必要时接手。
    candidates = expand(text)
    hits: list[dict[str, Any]] = []
    refresh_entry_id = (entry_id or "").strip()
    matched_as = text
    #: 这次答案是从哪一层召回出来的：topic | semantic（web/refresh 在下面改写）。
    retrieval = "topic"

    # An explicit refresh must skip the local hit path entirely: the entry the
    # caller named may pass the gate (so it would come back as a `kb` hit and
    # never be re-fetched). Asking for a refresh means "go get better material
    # for this one", so the only local work left is loading its stored text.
    for candidate in ([] if refresh_entry_id else candidates):
        hits, blocked = _apply_gate(text, store.search(candidate))
        if blocked and not refresh_entry_id:
            refresh_entry_id = blocked
        if hits:
            matched_as = candidate
            break

    # 主题检索是字面匹配，用户换个说法就可能一个字都对不上（「党是怎么出来的」
    # 对《中国共产党的诞生》）。`query.expand` 用规则补了一部分，但规则只能补
    # 它见过的句式；语义检索补的是规则漏下的那些说法。
    #
    # 顺序是有意的：主题检索先跑。它是一次本地扫描，没有网络往返；语义检索要
    # 先把查询嵌入一次。命中即返回的快路径不能被改变——知识库有一半价值就是
    # 「比联网快」，为了排序好看而给每次命中都加一次嵌入调用，是把快的部分改慢。
    semantic_note = ""
    #: 语义层没跑通（没配模型、配额用尽、网络失败）和「跑了但没有相似内容」
    #: 是两件事。前者用户能修，后者只能换来源；都报成 local_not_found，界面就会
    #: 让他去找材料，而真正的毛病在充值。
    semantic_failed = False
    if not hits and not refresh_entry_id:
        # 向量用**去掉指令外壳**的查询，不是原句。
        #
        # 「帮我做一个国民党一大的视频」嵌进去的是整个意图：做视频、国民党一大、
        # 帮我。模型会把这三样平均成一个既不像视频、也不像历史事件的向量。实测
        # 同一本书同一个问题：原句只能到章节块 0.4515（定位到「1924 年」），去掉
        # 填充词后正文块 0.6135（定位到「1924 年 · 国民党一大召开」）。
        #
        # 关键词那条路早就这么做了（`store._pairs` 里删 STOP_PHRASES），只有向量
        # 一直在用原句——同一个问题在两层的处理不一致，而界面上看不出来。
        semantic_query = strip_filler(text) or text
        semantic = await vector.search(semantic_query, limit=3)
        semantic_note = str(semantic.get("reason") or "")
        semantic_failed = semantic.get("ok") is False
        if semantic.get("hits"):
            loaded: list[dict[str, Any]] = []
            for item in semantic["hits"]:
                entry = store.get(str(item.get("entry_id") or ""))
                if not entry:
                    continue
                entry = dict(entry)
                # 段落只用来解释「凭哪一句命中」，判定仍然看条目本身。
                #
                # 书要特别处理：`item["text"]` 是分数最高的那一块，而它常常是
                # 章节目录（「1927 年：四一二反革命政变、南昌起义、……」）。
                # 把目录当成「命中依据」交给下游，等于告诉它「这一段就是答案」，
                # 而它一个字都答不了。所以有 `best_body` 时用它——那才是真正
                # 描述这件事的那一段。
                best = item.get("best_body") or {}
                entry["match"] = {
                    "score": item.get("score"),
                    "shared": [],
                    "chunk": str(best.get("text") or item.get("text") or "")[:400],
                    "part": item.get("part") or "",
                    "chapter": str(best.get("chapter") or item.get("chapter") or ""),
                    "heading": str(best.get("heading") or item.get("heading") or ""),
                }
                loaded.append(entry)
            if loaded:
                gated, blocked = _apply_gate(text, loaded)
                if blocked and not refresh_entry_id:
                    refresh_entry_id = blocked
                if gated:
                    hits = gated
                    retrieval = "semantic"
                else:
                    # 召回到了、但一条都没过闸。这和「什么都没召回到」是两种
                    # 情况：前者说明库里有沾边的东西但不可用，后者说明库里没有。
                    # 事后排查时这两句话指向完全不同的修法。
                    semantic_note = (
                        f"召回 {len(loaded)} 条，但均未通过权威来源或质量校验"
                    )

    if hits:
        entry = hits[0]
        store.touch(str(entry["id"]))
        if retrieval == "semantic":
            how = f"语义召回，相似度 {entry['match']['score']}"
        else:
            how = (
                f"相似度 {entry['match']['score']}"
                if matched_as == text
                else f"按「{matched_as}」检索，相似度 {entry['match']['score']}"
            )
        # 「命中有权威来源的《X》」对卡片成立，对书不成立：书是用户自己上传的，
        # 它没有白名单网址，来源就是那份文件。沿用那句话会让用户以为这本书是从
        # 政府网站抓来的——一个关于「这段材料能不能信」的判断被换了根据，却还
        # 显示着同样的措辞。
        is_book = str(entry.get("kind") or store.CARD) == store.BOOK
        locator = ""
        if is_book:
            chapter = str(entry["match"].get("chapter") or "")
            heading = str(entry["match"].get("heading") or "")
            locator = " · ".join(x for x in (chapter, heading) if x)
            trace = (
                f"命中已上传资料《{entry['title']}》"
                + (f"的「{locator}」" if locator else "")
                + f"（{how}），未联网"
            )
        else:
            trace = f"资料检索命中有权威来源的《{entry['title']}》（{how}），未联网"
        # 主题检索没有「命中的那一段」，就把正文开头当依据；书不能用这一招——
        # 它的 body 是整本六万字，塞进匹配片段等于把一次检索变成一个上下文炸弹。
        chunk = str(entry["match"].get("chunk") or "")
        if not chunk and not is_book:
            chunk = str(entry.get("body") or "")[:400]
        return {
            "entry": entry,
            "source": "kb",
            "retrieval": retrieval,
            "unverified": False,
            "trace": trace,
            "alternatives": [h["title"] for h in hits[1:]],
            # 下游（写脚本的 agent）要的是「哪一段能当依据」，不是「命中哪本书」。
            # 书有几百段，不给这一条它就得自己再检索一遍。
            "matched_chunk": chunk,
            "locator": locator,
            "kind": "book" if is_book else "card",
        }

    if not allow_web:
        # 语义层为什么没帮上忙，要一起说出来。否则「本地没有」和「本地有但
        # 索引挂了」看起来一模一样，而后者是能修的。
        suffix = f"（语义检索：{semantic_note}）" if semantic_note else ""
        raise ResearchError(
            f"本地资料库 没有「{text}」的可用材料{suffix}",
            error_type="local_error" if semantic_failed else "local_not_found",
        )

    if not search_available():
        raise ResearchError("没有可用的外部检索服务，无法从权威网站取得材料")

    pages: list[str] = []
    used: list[SearchResult] = []
    results: list[SearchResult] = []
    failures: list[dict[str, Any]] = []
    search_failures: list[SearchError] = []
    seen_urls: set[str] = set()
    tried = 0

    # An entry that exists but failed the gate is being *refreshed*, not created.
    # Its already-fetched text must be carried into the new summary: the point of
    # a refresh is to replace a thin draft with a better one, and dropping the old
    # material would leave the new summary strictly worse-informed than the run
    # before it.
    #
    # A stored page is only treated as "already fetched" when it is substantial.
    # A short stored body is exactly the symptom that triggered the refresh, so
    # refusing to download it again would freeze the bad state in place — the
    # original "later fetched 12000 chars, body never updated" bug, in miniature.
    material_by_url: dict[str, dict[str, Any]] = {}
    if refresh_entry_id:
        for item in ((store.get(refresh_entry_id) or {}).get("source_material") or []):
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "")
            text_value = str(item.get("text") or "")
            if not url or not text_value.strip():
                continue
            if len(text_value) < _STORED_MATERIAL_MIN:
                continue
            material_by_url[url] = item
            seen_urls.add(url)
        pages.extend(
            str(item.get("text") or "") for item in material_by_url.values()
        )
        used.extend(
            SearchResult(title=str(item.get("title") or url), url=url)
            for url, item in material_by_url.items()
        )

    # Search variants are tried sequentially.  A 404 proves the search capability
    # worked but the index is stale; it must trigger a different query and source,
    # not be reported as "the model cannot browse".
    for web_query in _research_queries(text, candidates):
        try:
            found = await search(web_query, limit=max(12, max_pages * 4))
        except SearchError as exc:
            search_failures.append(exc)
            continue

        fresh = [
            result for result in found
            if is_authoritative(result.url) and result.url not in seen_urls
        ]
        results.extend(fresh)
        # Do not let one stale query consume the whole attempt budget.  Eight dead
        # links are enough evidence to switch wording; up to four query forms are
        # available and all URLs are deduplicated.
        page_limit = 8 if _biography_subject(text) else max(3, max_pages)
        for result in fresh[:8]:
            if tried >= 24:
                break
            seen_urls.add(result.url)
            tried += 1
            try:
                page = await _fetch(result)
            except FetchError as exc:
                failures.append({
                    "url": result.url,
                    "title": result.title,
                    **exc.as_dict(),
                })
                continue
            except Exception as exc:  # noqa: BLE001 - classify unexpected fetcher bugs
                failures.append({
                    "url": result.url,
                    "title": result.title,
                    "error_type": "fetcher_error",
                    "reason": f"抓取器内部错误：{type(exc).__name__}: {str(exc)[:120]}",
                    "action": "检查后端抓取器日志",
                })
                continue
            if page:
                pages.append(page)
                used.append(result)
            if len(pages) >= page_limit:
                break
        # Alternate query wording is a dead-link recovery mechanism. Once this
        # query yielded any usable body there is no reason to pay for more search
        # calls; the inner loop has already collected as many pages as it can.
        if _material_ready(text, pages):
            break

    if not results and not pages:
        if search_failures:
            last = search_failures[-1]
            raise ResearchError(
                f"外部检索没有取得链接：{last}",
                error_type=last.kind,
                action="检查模型是否支持 web_search，或配置博查/Tavily/Serper 检索 Key",
            ) from last
        raise ResearchError(
            f"外部检索没有找到权威来源（只接受 {describe_allowed()}）",
            error_type="no_authoritative_result",
            action="更换检索词后重试",
        )

    if not pages:
        counts = Counter(str(item.get("error_type") or "unknown") for item in failures)
        labels = {
            "not_found": "失效链接 HTTP 404/410",
            "connection_error": "本机连接失败",
            "dns_error": "DNS 解析失败",
            "proxy_error": "代理失败",
            "timeout": "连接超时",
            "access_denied": "网站拒绝抓取",
            "rate_limited": "网站限流",
            "empty_or_antibot": "空页面/反爬",
            "not_article": "首页/栏目列表",
            "multiple_failures": "多种网络错误",
            "fetcher_error": "抓取器内部错误",
        }
        breakdown = "、".join(
            f"{labels.get(kind, kind)} {count} 条" for kind, count in counts.items()
        ) or "页面没有返回正文"
        dominant = counts.most_common(1)[0][0] if counts else "fetch_failed"
        action = (
            "检索结果均为失效链接；程序已自动更换检索词和权威来源，仍未找到可用正文"
            if set(counts) == {"not_found"}
            else "程序已自动更换权威来源；请按错误类型检查网络、代理或站点访问限制"
        )
        raise ResearchError(
            f"联网检索能力正常，找到 {len(results)} 个权威链接；"
            f"自动更换并尝试了 {tried} 条，仍未取得正文（{breakdown}）",
            error_type=dominant,
            details=failures,
            action=action,
        )

    if not _material_ready(text, pages):
        coverage = _biography_coverage("\n".join(pages))
        labels = {
            "early": "早年/求学", "revolution": "革命经历", "construction": "建设时期",
            "reform": "改革开放", "late": "晚年/逝世",
        }
        present = "、".join(labels[x] for x in labels if x in coverage) or "无完整阶段"
        raise ResearchError(
            f"已抓到 {len(pages)} 篇权威正文，但内容不是完整生平材料（现有：{present}），"
            "已继续更换权威来源，仍未补齐早年、革命、改革和晚年阶段",
            error_type="insufficient_topic_coverage",
            details=[
                {"url": result.url, "title": result.title, "chars": len(page)}
                for result, page in zip(used, pages)
            ],
            action="需要找到生平、年谱或人物简介类权威正文，专题评论只能作为补充来源",
        )

    try:
        draft = await summarize(text, pages)
    except Exception as exc:  # noqa: BLE001 - return an actionable MCP failure
        raise ResearchError(f"权威网页正文已取得，但整理材料失败（{str(exc)[:120]}）") from exc

    material = "\n\n".join(pages)
    draft_issues = _summary_quality_issues(text, draft, material)
    if draft_issues:
        raise ResearchError(
            "网页正文已取得，但整理结果未达到可用标准：" + "；".join(draft_issues),
            error_type="summary_quality_failed",
            action="保留抓取材料并重新整理，不使用或入库这份过短摘要",
        )

    source_rows = [
        {"kind": "参考资料", "org": r.site or site_label(r.url), "title": r.title, "url": r.url}
        for r in used
    ]
    entry = _persist(
        text,
        draft,
        source_rows,
        source_material=[
            {"url": r.url, "title": r.title, "chars": len(page), "text": page}
            for r, page in zip(used, pages)
        ],
        entry_id=refresh_entry_id,
    )

    # 新条目立刻进语义索引。等下次重建才生效的话，同一次运行里后续的语义召回
    # 就看不到它——而「刚查到的材料马上能用」是这个库的设计前提。索引失败不改
    # 变本次结果，只是如实报告这一层没生效。
    indexed = await vector.index_entry(entry)

    return {
        "entry": entry,
        "source": "web",
        "retrieval": "web",
        "indexed": bool(indexed.get("ok")),
        "index_note": "" if indexed.get("ok") else str(indexed.get("reason") or ""),
        "trace": (
            f"知识库没有，外部检索到 {len(used)} 个权威来源，已生成《{entry['title']}》"
            "（待审核，确认后下次直接命中）"
        ),
        "alternatives": [],
    }


def _persist(
    query: str,
    draft: Any,
    sources: list[dict[str, str]],
    *,
    source_material: list[dict[str, Any]] | None = None,
    entry_id: str = "",
) -> dict[str, Any]:
    """把一份摘要写成条目。`pending` 而不是 `approved`：这是机器找来的，
    还没有人看过。"""
    authoritative_sources = [
        source for source in sources
        if is_authoritative(str(source.get("url") or ""))
    ]
    if not authoritative_sources:
        raise ResearchError("正文没有可追溯的权威来源链接，拒绝写入知识库")
    return store.save(
        {
            "id": entry_id,
            "title": draft.title,
            "category": draft.category,
            "period": draft.period,
            "body": draft.body,
            "key_points": draft.key_points,
            # Persist only the exact URLs that the MCP fetch path actually used.
            # The summariser may condense prose, but it must not invent citations.
            "sources": authoritative_sources,
            "source_material": source_material or [],
            "status": store.PENDING,
            "origin": "web",
            "query": query,
            "note": f"由「{query}」从权威网站自动检索生成，待人工确认",
        }
    )


async def entry_from_pages(
    query: str,
    pages: list[str],
    used: list[dict[str, str]] | None = None,
    *,
    entry_id: str = "",
    existing_material: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """把「已经抓好的一批页面」整理成一条知识条目。

    给那些自己抓了页面的调用方用（导演判断时可能绕过 `research`，直接
    `search_web` + `fetch_web_page`）。没有这条路，它抓下来的东西会停在上下文里，
    这次用完就没了——而这正是知识库最容易变成死库的原因。
    """
    text = (query or "").strip()
    usable = [p for p in pages if (p or "").strip()]
    if not text or not usable:
        raise ResearchError("没有可整理的材料")
    if not _material_ready(text, usable):
        raise ResearchError(
            "抓到的正文与主题有关，但不足以覆盖完整生平阶段",
            error_type="insufficient_topic_coverage",
            action="继续寻找生平、年谱或人物简介类权威来源",
        )
    draft = await summarize(text, usable)
    issues = _summary_quality_issues(text, draft, "\n\n".join(usable))
    if issues:
        raise ResearchError(
            "整理结果未达到可用标准：" + "；".join(issues),
            error_type="summary_quality_failed",
            action="不使用或入库这份摘要，继续补充权威材料",
        )
    # `used` arrives from the caller and may contain rows that were never
    # fetched (the model's own citation list, a section index, an aggregator
    # page). A source is only real if this run actually holds its body text:
    # the stored material is the evidence, and a citation without evidence is
    # exactly the kind of plausible-but-unverifiable line this project refuses.
    source_rows: list[dict[str, str]] = []
    seen_source: set[str] = set()
    for index, page in enumerate(usable):
        source = used[index] if used and index < len(used) else {}
        url = str((source or {}).get("url") or "")
        if not url or url in seen_source:
            continue
        seen_source.add(url)
        source_rows.append(
            {
                "kind": str((source or {}).get("kind") or "参考资料"),
                "org": str((source or {}).get("org") or site_label(url)),
                "title": str((source or {}).get("title") or url),
                "url": url,
            }
        )
    materials = list(existing_material or [])
    known = {str(item.get("url") or "") for item in materials if isinstance(item, dict)}
    for index, page in enumerate(usable):
        source = used[index] if used and index < len(used) else {}
        url = str((source or {}).get("url") or "")
        if url and url in known:
            continue
        materials.append({
            "url": url,
            "title": str((source or {}).get("title") or url),
            "chars": len(page),
            "text": page,
        })
        if url:
            known.add(url)
    entry = _persist(
        text,
        draft,
        source_rows,
        source_material=materials,
        entry_id=entry_id,
    )
    # 与 research() 的联网分支同理：写库和写索引要挨着，中间隔一段时间就会出现
    # 「条目在、但语义检索搜不到」的窗口，而这类不一致最难被人想到去查。
    await vector.index_entry(entry)
    return entry


async def _fetch(result: SearchResult) -> str:
    from .websearch import fetch_text

    return await fetch_text(result.url)


# ============================================================================
# 整本资料导入
# ============================================================================

#: 一本书至少要有这么多字才值得建索引。
#: 低于这个量的文本用主题检索就够，进向量库只会多一条几乎不会被召回的记录。
MIN_BOOK_CHARS = 200

#: 面板上最多列多少个章节标题。超过这个量的书该按卷拆开上传。
_MAX_CHAPTER_LABELS = 500


async def import_book(
    *,
    text: str,
    title: str = "",
    filename: str = "",
    category: str = "整本资料",
    overwrite: bool = True,
) -> dict[str, Any]:
    """把一本上传的资料解析、入库并建立语义索引。

    三个决定，都和「书和卡片的区别」有关：

    1. **直接 `approved`**。卡片要审核是因为内容由模型从网上摘要而来，可能是错的；
       书是用户自己提供的，审核等于要用户确认自己刚上传的东西，只是多一次点击。

    2. **原文整篇存进 `body`**。切块参数会改（目标字数、章节正则），存了原文才
       能在改规则后重新切；只存切好的块，旧书就会永远停在旧规则上。

    3. **同名覆盖**。用户重新上传同一本书是在更新版本，不是在造第二本——两本同名
       书同时参与检索，会让同一条内容在结果里占两个位置。书名传空时用文件名兜底。

    写库和写索引挨着做：中间隔一段时间就会出现「书在、但搜不到」的窗口，而这类
    不一致最难被人想到去查。
    """
    raw = (text or "").strip()
    if not raw:
        raise ResearchError("文件是空的", error_type="empty_document")

    doc = document.parse(raw, filename=filename or title)
    book_title = (title or "").strip() or doc.title or ""
    if not book_title:
        raise ResearchError("无法确定书名，请手动填写", error_type="missing_title")
    doc.title = book_title

    if doc.char_count < MIN_BOOK_CHARS:
        raise ResearchError(
            f"内容只有 {doc.char_count} 字，太少，不足以建索引（至少 {MIN_BOOK_CHARS} 字）",
            error_type="document_too_short",
        )

    entry_id = ""
    replaced = False
    if overwrite:
        # 按标题在所有条目里找同名，**不筛 kind**。
        #
        # 上一版筛了 `kind=book`，于是留下一个死角：一条被错标成 card 的书永远
        # 匹配不上，重传会新建第二条同名记录，旧的那条 card 继续躺在库里被按卡片
        # 切分。查找条件依赖一个可能已经错了的字段，等于把错误固化下来。
        for existing in store.list_entries(limit=500):
            if str(existing.get("title") or "").strip() == book_title:
                entry_id = str(existing.get("id") or "")
                replaced = True
                break

    entry = store.save(
        {
            "id": entry_id or None,
            "title": book_title,
            "kind": store.BOOK,
            "category": category,
            "period": "",
            "body": raw,
            # 章节标题存进 key_points，供面板显示「这本书收了什么」。上限 500
            # 是给网络小说那种上千章的长篇留个边界——超过它前端渲染会明显变慢，
            # 而超过 500 章的书本来也该按卷拆开上传。
            "key_points": [c.title for c in doc.chapters if c.title][:_MAX_CHAPTER_LABELS],
            "sources": [
                {
                    "kind": "上传文件",
                    "org": "",
                    "title": filename or book_title,
                    "url": "",
                }
            ],
            "status": store.APPROVED,
            "origin": "upload",
            "query": "",
            "note": doc.note,
        }
    )

    # 同名覆盖走的是 `store.save` 的 UPDATE 分支，而那条分支不改 kind——所以一条
    # 历史上被错误标成 `card` 的书，重传之后仍然是 `card`，然后继续被按卡片切分。
    # 这里显式校正：上传路径一定知道自己在收整本资料。
    kind_fixed = store.set_kind(entry["id"], store.BOOK)
    if kind_fixed:
        entry = store.get(entry["id"]) or entry

    index_state = await vector.index_entry(entry)
    return {
        "entry": entry,
        "structure": document.preview(doc),
        "replaced": replaced,
        "index": index_state,
    }
