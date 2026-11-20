"""导演自己决定去查什么，而不是拿用户那句话原样查一次。

老做法对「遵义会议」有效，对「建国」无效：书里没有「建国」这个说法，1949 那一块
的小标题逐字是「中华人民共和国成立」。这不是检索坏了，是**同一个事件的两种叫法**，
只有知道这件事的人能跨过去。所以查什么由导演想。

但「让模型想」很容易变成「让模型背模板」：给它「背景/经过/影响」三个格子，它换
到人物生平就填「出生/工作/现在」，三个格子都满着，检索词一个都落不了地。这个模块
的写法就是为了防这件事——**写死验收方法，不写死问题和检索词**。

三条不许变的事，全部由程序核对，不采信模型自述：

1. **每个问题必须落成能查的词，至少三个字。** 两个字的词在中文里几乎在任何文档
   里都出现过，命中率恒等于 1，那不是检索。四个字的空话（「历史背景」）同样查
   不到东西，交给 `_looks_like_filler` 拦。
2. **词从哪儿来必须标。** `grounded` 是从材料原文里抄的，`inferred` 是导演按
   常识补的。标记由程序做子串比对得出——让模型自己标，它会一律标成 grounded。
3. **覆盖率由程序算。** 导演可以说「这条我查到了」，不可以自己报一个百分比。

至于「一条视频该覆盖哪些方面」，不写成清单。程序只认五问（何时/何地/何人/
发生了什么/为何重要），导演拿什么凑齐它自己想办法。换主题时验收标准不变，问题
清单应该完全不同——如果清单没怎么变，那说明它在背模板，`_template_smell` 会
把它记进 `notes`，让人看见。
"""

from __future__ import annotations

import re
from typing import Any, Awaitable, Callable

#: 五个验收位。不是「维度模板」，是「一条片子能不能立起来」的最小事实集合：
#: 少了哪一项，观众都会在那一秒出戏。怎么凑齐不管。
SLOTS = ("when", "where", "who", "what", "why")
SLOT_LABEL = {
    "when": "何时",
    "where": "何地",
    "who": "何人",
    "what": "发生了什么",
    "why": "为何重要",
}

#: 检索词下限。两个字的词在中文里几乎命中一切，见模块说明。
MIN_QUERY_CHARS = 3

#: 检索词总预算，两轮合起来算。不是成本考虑，是两件别的事：
#:
#: 一是清单要贴着给人看，八行以上读者就不看了；二是每次都走一遍「字面 + 语义」
#: 两层，语义层要调一次嵌入——嵌入服务连不上时，每个词都要等一次连接超时。
#: 十几个词排下来，用户看到的是「一直转圈」，而不是「它在想」。
MAX_QUERIES = 8

#: 最多想几轮。第二轮的存在是为了补缺口，不是为了「再想想」——三轮以上说明第一
#: 轮的出题就是错的，继续问下去只会把错的题问得更细。
MAX_ROUNDS = 2

#: 材料包上限。6000–8000 字大约是一条一分钟片子的 8–10 倍冗余；再多，改编那一步
#: 会开始摘抄而不是改编。
MATERIAL_CHARS = 8000

#: 一个片段留多长。一块书正文本来就一两百字，卡片可能上千字，截断到这一段够讲
#: 清一件事，又不至于把材料包一次占满。
CLIP_CHARS = 900

#: 五问缺几个还算「材料够」。缺两个以上才算缺口——一分钟的片子讲不清五件事，
#: 但缺三个的话，剩下两件撑不起一条片子。
MAX_MISSING = 2

#: 这些词出现在检索词里，说明它在写标题而不是在查东西。「历史背景」「重要意义」
#: 不是任何一份资料里的说法，查不到才是正常的。
_FILLER_QUERY = re.compile(
    r"^(?:.{0,4})?(?:背景|经过|意义|影响|过程|情况|介绍|概述|内容|相关|资料|"
    r"发展|演变|始末|全貌|详情)$"
)
#: 制作要求混进检索词。它们属于 brief，不属于检索。
_REQUEST_NOISE = re.compile(
    r"(?:视频|短片|纪录片|片子|成片|镜头|画面|字幕|配音|风格|时长|分钟|秒钟)"
)


class PlanError(RuntimeError):
    """导演想不出可执行的检索计划，且兜底那条路也没走通。"""


def _compact(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", str(text or ""))


def _config() -> Any:
    from .. import models_config

    profile = models_config.load().provider_for("director")
    if profile is None:
        raise PlanError("没有可用于检索规划的文本模型")
    return models_config.provider_to_llm_config(profile)


async def _think(system: str, user: str, *, max_tokens: int = 2400) -> Any:
    from ..tools import llm as llm_tools

    return await llm_tools.chat_json(
        _config(), system=system, user=user, max_tokens=max_tokens
    )


#: 问「查什么」。要求它给问题而不是给维度，是为了让每条出题都带一个意图——
#: 没有意图的检索词，事后没人能判断它该不该算「查到了」。
PLAN_SYSTEM = """你是纪录片的导演，正在决定开机前要去资料里查什么。

输入是一个主题（可能还有目标时长）。你要想出 3–6 个具体的问题，每个问题配上
1–3 个能直接拿去检索的词。

硬要求：
- 检索词至少三个汉字。不要用「背景」「经过」「意义」「介绍」这种词——那不是
  任何一份资料里的说法，查不到才是正常的。
- 检索词是专有名词或短语，不是句子。不要带「视频」「镜头」「时长」这类制作要求。
- 问题必须具体到能判断「查到没查到」。写「了解历史背景」不合格；写「1949 年
  之前国共谈判为什么破裂」合格。
- 不要套模板。不同主题的问题清单应该几乎不重叠；如果你写出来的问题和「背景／
  经过／影响」可以互换到任何主题上，那说明你在填格子。
- 每条问题标出它能确定哪几项事实，只能用这几个值：when（何时）where（何地）
  who（何人）what（发生了什么）why（为何重要）。
- 不要编造事实。你只负责提出要查什么，不负责回答。

另外，用 `analysis` 写一段你自己的分析（120–260 字，第一人称）。这不是给用户看的
总结，是你出题之前的思考过程。按这个顺序讲三件事：

1. **这片子要讲清楚，还缺哪几方面**——点名，不要写「缺少细节」这种空话
   （「要讲清还缺谁参加、在哪开的、定了什么」）。
2. **我怀疑材料里会怎么称呼它**（正式名？俗称？人名？事件名？），所以我配了
   这几个词——把词点出来，别只说「配了几个词」。
3. **现在就按这几个词去查**。

这段话会被原样贴出来，紧接着就是你查到的结果——所以它只能停在「正在做的事」
上，不要预告后面（别写「如果查不到我还会补查」），也不要写「我将从背景、经过、
意义三个角度」这种套话。

只输出 JSON：
{
  "anchor": {
    "kind": "event|person|concept|org|place",
    "subject": "这件事/这个人最通行的叫法",
    "when": "大致年代，说不准就留空",
    "note": "一句话：这条片子要讲清楚的是什么"
  },
  "analysis": "第一人称的分析过程，见上",
  "questions": [
    {
      "ask": "要弄清楚的这一件具体的事",
      "slots": ["when", "what"],
      "queries": ["检索词一", "检索词二"],
      "why": "为什么这条片子需要它"
    }
  ]
}
"""

#: 第二轮：看过第一轮找到的东西之后补缺口。这一轮同时也是「排查」——导演要判断
#: 拿回来的片段能不能用，而不是只看命中数。
REVIEW_SYSTEM = """你是纪录片的导演。第一轮检索已经跑完，现在做两件事：
排查拿回来的东西，然后决定还要补查什么。

排查的规矩：
- 「库里没有这件事」和「库里不叫这个名」是两件事。后者说明该换个说法再查一次，
  不是缺材料。
- 保留的片段必须真的回答了那个问题。只是提到了关键词不算。
- `surprise` 必须填：实际材料里和你预想不一样的地方。如果你的预想和材料完全
  一致，说明你没有真读，回去读。

补查的规矩和第一轮一样：检索词至少三个汉字、是专有名词、不套模板。只补没查到的
那几项，已经够了的部分不要重复查。

另外用 `analysis` 写一段你看到第一轮结果之后的判断（120–260 字，第一人称），
按这个顺序讲：

1. 哪几样已经拿到了，拿到的片段是**按什么角度写的**（按事件经过？按会议名？
   按人名？）——这说明书里是怎么组织的。
2. 哪一样还空着，你判断是「库里没有这件事」还是「库里不叫这个名」。
3. **紧接着你要做的事**：
   - `stop` 为 false 时：「我换成这几个词再查一次」——把词点出来，现在就查。
   - `stop` 为 true 时：「够了，不查了」。这时**绝对不许**写「我再查一次」
     「稍后补查」之类的话——程序不会再去查，那句话就成了空话。

这段话会被原样贴出来，紧跟其后的就是第二轮的结果，所以不要预告更远的事。

只输出 JSON：
{
  "analysis": "第一人称：看完第一轮结果我怎么判断，为什么这样补查",
  "reviews": [
    {"ask": "第一轮那条问题的原文", "verdict": "answered|partial|missing",
     "note": "一句话：拿到的片段回答了没有，差在哪"}
  ],
  "gaps": ["还缺的验收位，只能用 when/where/who/what/why"],
  "questions": [
    {"ask": "补查的具体问题", "slots": ["why"], "queries": ["检索词"], "why": "为什么还缺"}
  ],
  "surprise": "材料里和你预想不一样的地方；确实没有就写「无，与预想一致」",
  "stop": false
}
"""


def _looks_like_filler(query: str) -> bool:
    """这个词是不是在写标题而不是在查东西。

    「历史背景」「事件经过」这类词长度够、看起来像检索词，但任何一份资料里都不会
    这样写标题，拿它去查必然是 0 命中，然后被记成「库里没有」——一个假的缺口。
    """
    text = _compact(query)
    if not text:
        return True
    if _REQUEST_NOISE.search(text):
        return True
    return bool(_FILLER_QUERY.match(text))


def _grounded(query: str, pool: str) -> bool:
    """这个词是从材料里抄的，还是导演按常识补的。

    判据是「有连续四个字在材料里出现过」。四个字足够排除「中国」「人民」这种
    到处都有的词，又不至于要求整词命中——库里写「中华人民共和国成立」，导演
    改写出的「中华人民共和国中央人民政府」仍然算抄来的。

    这个判断**必须由程序做**。让模型自己标，它会一律标成 grounded——因为在它
    看来，它写的每个词都「有依据」。
    """
    text = _compact(query)
    if not text:
        return False
    if len(text) < 4:
        return text in pool
    return any(text[i : i + 4] in pool for i in range(len(text) - 3))


def _accept_questions(
    raw: Any, *, pool: str, round_no: int, seen: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """把模型给的题单变成能执行的东西，同时记下丢掉了什么。

    丢掉的东西要留下来，不是为了日志好看：卡上写着「导演查了 5 个词都没命中」，
    而实际上它写了 11 个、其中 6 个因为「两个字」被程序拦下——这两句话对用户
    的含义完全不同，前者是库里没有，后者是导演不会提问。
    """
    questions: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    items = raw if isinstance(raw, list) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        ask = str(item.get("ask") or "").strip()
        if not ask:
            continue
        slots = [
            str(slot).strip().lower()
            for slot in (item.get("slots") or [])
            if str(slot).strip().lower() in SLOTS
        ]
        queries: list[dict[str, Any]] = []
        for query in item.get("queries") or []:
            text = re.sub(r"\s+", "", str(query or ""))
            if len(_compact(text)) < MIN_QUERY_CHARS:
                dropped.append({"query": text, "why": f"少于 {MIN_QUERY_CHARS} 个字，命中率恒等于 1"})
                continue
            if _looks_like_filler(text):
                dropped.append({"query": text, "why": "这是标题说法，不是任何资料里的词"})
                continue
            if text in seen:
                dropped.append({"query": text, "why": "前面已经查过同一个词"})
                continue
            seen.add(text)
            queries.append({"text": text, "origin": "grounded" if _grounded(text, pool) else "inferred"})
        if not queries:
            continue
        questions.append({
            "id": f"q{round_no}-{len(questions) + 1}",
            "round": round_no,
            "ask": ask[:120],
            "slots": slots,
            "queries": queries,
            "why": str(item.get("why") or "").strip()[:160],
            "hits": [],
        })
    return questions, dropped


def _within_budget(
    questions: list[dict[str, Any]], room: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """把这一轮的题单裁到预算以内，返回 (留下的, 裁掉的)。

    模型被要求出 3–6 个问题、每题 1–3 个词，它很容易出到十几条，而总预算只有
    `MAX_QUERIES`。不裁的话会出现最糟的一种展示：清单上有六条，后三条写着
    「未命中」——它们其实一次都没查过，只是被预算截断了。用户读到的是「查过
    没有」，而事实是「没查」。

    宁可不显示，也不显示一句假的结论。
    """
    kept: list[dict[str, Any]] = []
    cut: list[dict[str, Any]] = []
    used = 0
    for question in questions:
        size = len(question.get("queries") or [])
        if used + size > room and kept:
            cut.append(question)
            continue
        kept.append(question)
        used += size
    return kept, cut


def coverage(questions: list[dict[str, Any]]) -> dict[str, Any]:
    """五个验收位现在覆盖了哪几项。**只由程序算。**

    导演自报的覆盖率会向「我已经做完了」倾斜，而它没有办法核对——它能看到的是
    自己写的问题，不是材料包。这里的分子分母都来自实际命中的片段。
    """
    filled: dict[str, list[str]] = {slot: [] for slot in SLOTS}
    answered = 0
    for question in questions:
        if not question.get("hits"):
            continue
        answered += 1
        for slot in question.get("slots") or []:
            if slot in filled:
                filled[slot].append(str(question.get("id") or ""))
    covered = [slot for slot in SLOTS if filled[slot]]
    missing = [slot for slot in SLOTS if not filled[slot]]
    return {
        "covered": covered,
        "missing": missing,
        "answered": answered,
        "asked": len(questions),
        "enough": answered > 0 and len(missing) <= MAX_MISSING,
    }


#: 「通用问题」的句式。整张题单都长这样，说明它是填格子填出来的，不是从这件事
#: 长出来的。
_TEMPLATE_ASK = re.compile(r"背景|经过|意义|影响|发展过程|来龙去脉|相关情况")


def _template_smell(questions: list[dict[str, Any]]) -> str:
    """这张题单像不像模板填出来的。

    不做跨主题比对（那要跑两次模型，成本翻倍），只看一件事：**是不是整张单子
    都在问同一类通用项**。三题里有两题以上带「背景/经过/意义」，而这三样可以
    原样搬到任何主题上——那就是模板。

    真正从材料里长出来的题长得不一样：「政协之前为什么谈崩」「定都定在哪」这种
    问题带着这件事自己的名词，搬不到别的主题上。
    """
    if len(questions) < 3:
        return ""
    hits = [q for q in questions if _TEMPLATE_ASK.search(str(q.get("ask") or ""))]
    if len(hits) < len(questions) - 1:
        return ""
    return "大部分问题都在问「背景／经过／意义」这类通用项，换个主题也能照搬"


def pack(clips: list[dict[str, Any]], *, limit: int = MATERIAL_CHARS) -> str:
    """把片段拼成材料包。**每一段都带 locator。**

    没有 locator 的片段不进包：下游写脚本时要能说出「依据在哪一章哪件事」，而
    一份说不出依据的材料，最后一定会被写成「据记载」这种谁也不用负责的句子。
    """
    parts: list[str] = []
    used = 0
    for index, clip in enumerate(clips, start=1):
        text = str(clip.get("text") or "").strip()
        locator = str(clip.get("locator") or "").strip()
        title = str(clip.get("title") or "").strip()
        if not text or not (locator or title):
            continue
        head = "《{}》{}".format(title or "资料", f" {locator}" if locator else "")
        block = f"【片段 {index}】{head}\n{text[:CLIP_CHARS]}"
        if used + len(block) > limit:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def _clip_key(clip: dict[str, Any]) -> str:
    return "|".join([
        str(clip.get("entry_id") or ""),
        str(clip.get("locator") or ""),
        _compact(str(clip.get("text") or ""))[:40],
    ])


async def _no_search(query: str) -> dict[str, Any]:
    """默认不查。检索能力由调用方注入，见 `run` 的 `search` 参数。"""
    return {"found": False, "clips": [], "reason": "没有接入检索", "error_type": "tool_disabled"}


def _material_pool(topic: str, questions: list[dict[str, Any]], clips: list[dict[str, Any]]) -> str:
    """判「抄来的还是想出来的」用的底本。

    包含主题本身：用户说的「建国」虽然没有出现在书里，但它确实是用户给的说法，
    把它判成「导演编的」没有意义。
    """
    chunks = [str(topic or "")]
    for question in questions:
        chunks.append(str(question.get("ask") or ""))
    for clip in clips:
        chunks.append(str(clip.get("text") or ""))
        chunks.append(str(clip.get("locator") or ""))
        chunks.append(str(clip.get("title") or ""))
    return _compact("\n".join(chunks))


def _digest(questions: list[dict[str, Any]], clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """给第二轮看的摘要。整段材料太贵，只给「哪条题、命中没有、命中在哪」。"""
    out: list[dict[str, Any]] = []
    by_locator = {
        str(clip.get("locator") or ""): clip for clip in clips
    }
    for question in questions:
        hits = question.get("hits") or []
        out.append({
            "ask": question.get("ask"),
            "slots": question.get("slots") or [],
            "queries": [entry["text"] for entry in question.get("queries") or []],
            "found": len(hits),
            "where": [
                f"{by_locator.get(loc, {}).get('title') or ''} {loc}".strip()
                for loc in hits[:2]
            ],
        })
    return out


async def _notify(on_progress: Callable[[dict[str, Any]], Any] | None, snapshot: dict[str, Any]) -> None:
    """把「到目前为止想到哪、查到哪」推给调用方，让它原地更新卡片。

    同步回调（记录）和异步回调（发消息）都接住，跟 `on_classified` 一个规矩：
    调用方不必关心这里怎么调它。回调自己出错不能把检索搞失败——账本是旁路，
    不是结果。
    """
    if on_progress is None:
        return
    try:
        maybe = on_progress(snapshot)
        if hasattr(maybe, "__await__"):
            await maybe
    except Exception:  # noqa: BLE001 - 更新卡片失败不影响检索本身
        pass


def _snapshot(
    *,
    topic: str,
    anchor: dict[str, Any],
    questions: list[dict[str, Any]],
    tried: list[dict[str, Any]],
    analysis: list[dict[str, Any]],
    notes: list[str],
    surprise: str,
    rounds: int,
    dropped: list[dict[str, Any]],
) -> dict[str, Any]:
    """一次「进度快照」。内容与最终返回的账本同形，只是 tried 里少了还没查的词。"""
    return {
        "topic": topic,
        "anchor": dict(anchor),
        "questions": list(questions),
        "tried": list(tried),
        "coverage": coverage(questions),
        "analysis": list(analysis),
        "notes": list(notes),
        "surprise": surprise,
        "rounds": rounds,
        "dropped": list(dropped),
    }


async def run(
    topic: str,
    *,
    search: Callable[[str], Awaitable[dict[str, Any]]] | None = None,
    seconds: int = 0,
    think: Callable[..., Awaitable[Any]] | None = None,
    on_tool: Callable[[str, dict[str, Any], dict[str, Any]], None] | None = None,
    on_progress: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """主题 → 一份带出处的材料包 + 一张覆盖率账单。

    返回的 `clips` 为空不代表失败：它代表「导演按自己的问法查过了，本地确实没有」。
    这两种情况的下一步完全不同——前者该重新出题，后者该去问用户从哪儿找。

    `search` 由调用方注入，这个模块不认识 MCP、不认识知识库：它只负责「想查什么」
    和「记账」。检索那条路只有一份实现（`intake._search_one` → `search_knowledge`
    工具），换接口时不用改两次。
    """
    text = str(topic or "").strip()
    if not text:
        return _empty("没有主题，无法出题")

    do_search = search or _no_search
    do_think = think or _think
    clips: list[dict[str, Any]] = []
    seen_clips: set[str] = set()
    questions: list[dict[str, Any]] = []
    tried: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    seen_queries: set[str] = set()
    anchor: dict[str, Any] = {}
    notes: list[str] = []
    surprise = ""
    rounds = 0
    plan_error = ""
    #: 模型每一轮自己写的分析（第一人称）。这不是给人看的总结，是它的思考过程：
    #: 怎么理解主题、怀疑材料里叫什么、看到命中/未命中后怎么判断。原样留着，
    #: 「换了主题也换个讲法」靠的就是它——程序拼不出这种话。
    analysis: list[dict[str, Any]] = []

    for round_no in range(1, MAX_ROUNDS + 1):
        pool = _material_pool(text, questions, clips)
        if round_no == 1:
            system = PLAN_SYSTEM
            user = f"主题：{text}\n目标时长：{seconds or '未指定'} 秒"
        else:
            system = REVIEW_SYSTEM
            user = (
                f"主题：{text}\n目标时长：{seconds or '未指定'} 秒\n"
                f"已有锚点：{anchor or '无'}\n"
                f"第一轮的题单和命中情况：{_json_like(_digest(questions, clips))}\n"
                f"还没覆盖的验收位：{coverage(questions)['missing']}"
            )
        try:
            data = await do_think(system, user)
        except Exception as exc:  # noqa: BLE001 - 出题失败不能变成「库里没有」
            plan_error = f"{type(exc).__name__}：{str(exc)[:150]}"
            break
        rounds = round_no
        if isinstance(data, dict):
            if not anchor and isinstance(data.get("anchor"), dict):
                anchor = {
                    "kind": str(data["anchor"].get("kind") or ""),
                    "subject": str(data["anchor"].get("subject") or ""),
                    "when": str(data["anchor"].get("when") or ""),
                    "note": str(data["anchor"].get("note") or ""),
                }
            surprise = str(data.get("surprise") or "").strip() or surprise
            # 这一轮它自己讲的分析。缺了不补——程序编一段「我怎么想」只会变成
            # 又一张固定模板，那正是这次要拆掉的东西。
            said = str(data.get("analysis") or "").strip()
            if said:
                analysis.append({"round": round_no, "text": said[:900]})
            for review in data.get("reviews") or []:
                if isinstance(review, dict) and str(review.get("note") or "").strip():
                    notes.append(
                        "{}：{}".format(
                            str(review.get("ask") or "")[:24], str(review.get("note"))[:80]
                        )
                    )
            if data.get("stop") is True and round_no > 1:
                break
        fresh, rejected = _accept_questions(
            (data or {}).get("questions") if isinstance(data, dict) else None,
            pool=pool, round_no=round_no, seen=seen_queries,
        )
        dropped.extend(rejected)
        fresh, cut = _within_budget(fresh, room=MAX_QUERIES - len(tried))
        for question in cut:
            # 裁掉的题要把它用掉的词从「查过了」里撤回：下一轮再提到同一个词时，
            # 不能说「已经查过」——它一次都没查过。
            for entry in question.get("queries") or []:
                seen_queries.discard(entry["text"])
            dropped.append({
                "query": str(question.get("ask") or ""),
                "why": "这一轮的检索词额度用完了，没去查",
            })
        questions.extend(fresh)
        if not fresh:
            if round_no == 1:
                notes.append("导演没有给出可执行的检索词")
            break

        # 想完了先说话：这一轮的判断和题单立刻贴上，用户看到的是「它打算查什么」，
        # 而不是等所有词查完才第一次出现。同一个 msg_key 原地覆写，不会堆卡片。
        await _notify(on_progress, _snapshot(
            topic=text, anchor=anchor, questions=questions, tried=tried,
            analysis=analysis, notes=notes, surprise=surprise,
            rounds=round_no, dropped=dropped,
        ))

        for question in fresh:
            for entry in question["queries"]:
                if len(tried) >= MAX_QUERIES:
                    break
                result = await do_search(entry["text"])
                found = [clip for clip in result.get("clips") or [] if clip.get("text")]
                tried.append({
                    "query": entry["text"],
                    "origin": entry["origin"],
                    "found": len(found),
                    "reason": str(result.get("reason") or "")[:160],
                    "error_type": str(result.get("error_type") or ""),
                })
                tool_calls.append({
                    "name": "search_knowledge",
                    "args": {"query": entry["text"], "allow_web": False, "strategy": "director"},
                    "failed": not found,
                })
                if on_tool is not None:
                    on_tool(
                        "search_knowledge",
                        {"query": entry["text"], "allow_web": False},
                        {
                            "found": bool(found),
                            "query": entry["text"],
                            "source": "kb" if found else "",
                            "reason": str(result.get("reason") or ""),
                            "error_type": str(result.get("error_type") or ""),
                        },
                    )
                for clip in found:
                    key = _clip_key(clip)
                    if key in seen_clips:
                        continue
                    seen_clips.add(key)
                    clips.append(clip)
                    question["hits"].append(str(clip.get("locator") or ""))

                # 每个词查完就贴一次回执。
                #
                # 用户的原话是「先去本地库里查……后面必须接上他在找的行动」：
                # 话里说「现在去查这几个」，紧接着就该一个词一个词地报回来。
                # 等整轮查完再一次性给，中间那段就是空的——他读到的是一句承诺
                # 加一段静默，然后突然出现一张终稿。
                #
                # 一次 notify 的成本是一次原地覆写（同一个 msg_key），不堆卡片；
                # 词数上限只有 MAX_QUERIES 个，多几次写库换「边查边报」是值的。
                await _notify(on_progress, _snapshot(
                    topic=text, anchor=anchor, questions=questions, tried=tried,
                    analysis=analysis, notes=notes, surprise=surprise,
                    rounds=round_no, dropped=dropped,
                ))

        # 这一轮查完了：把每个词的结果贴回去。用户能当场看见「哪个词命中、哪个
        # 没命中」，而不是最后一次性看到一张对不上号的终稿。
        await _notify(on_progress, _snapshot(
            topic=text, anchor=anchor, questions=questions, tried=tried,
            analysis=analysis, notes=notes, surprise=surprise,
            rounds=round_no, dropped=dropped,
        ))

        if coverage(questions)["enough"]:
            break

    # 出题失败（没配模型、模型挂了）时不能直接报「库里没有」——那会把一次配置
    # 故障说成一次检索结论。用原句查一次，行为退回上一版，但结论措辞不同。
    if not questions and not clips:
        result = await do_search(text)
        found = [clip for clip in result.get("clips") or [] if clip.get("text")]
        for clip in found:
            if _clip_key(clip) not in seen_clips:
                seen_clips.add(_clip_key(clip))
                clips.append(clip)
        tried.append({
            "query": text,
            "origin": "inferred",
            "found": len(found),
            # 原因原样留着（「database unavailable」这类信息只有工具知道），
            # 「没拿到计划」写进 notes，不要覆盖掉真正的失败原因。
            "reason": str(result.get("reason") or "")[:160],
            "error_type": str(result.get("error_type") or ""),
        })
        notes.append(f"没有拿到检索计划（{plan_error or '模型没有返回题单'}），按原句查了一次")
        if on_tool is not None:
            on_tool(
                "search_knowledge",
                {"query": text, "allow_web": False},
                {
                    "found": bool(found),
                    "query": text,
                    "source": "kb" if found else "",
                    "reason": str(result.get("reason") or ""),
                    "error_type": str(result.get("error_type") or ""),
                },
            )

    cover = coverage(questions)
    smell = _template_smell(questions)
    if smell:
        notes.append(f"这些题不像从材料里长出来的：{smell}")

    reason = _reason(
        topic=text, clips=clips, cover=cover, tried=tried,
        dropped=dropped, plan_error=plan_error, rounds=rounds,
    )
    return {
        "ok": bool(clips),
        "topic": text,
        "anchor": anchor,
        "questions": questions,
        "clips": clips,
        "material": pack(clips),
        "coverage": cover,
        "tried": tried,
        "dropped": dropped,
        "analysis": analysis,
        "notes": notes,
        "surprise": surprise,
        "tool_calls": tool_calls,
        "rounds": rounds,
        "plan_error": plan_error,
        "reason": reason,
    }


def _json_like(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


def _empty(reason: str) -> dict[str, Any]:
    return {
        "ok": False, "topic": "", "anchor": {}, "questions": [], "clips": [],
        "material": "", "coverage": coverage([]), "tried": [], "dropped": [],
        "notes": [], "surprise": "", "tool_calls": [], "rounds": 0,
        "plan_error": "", "reason": reason,
    }


def _reason(
    *,
    topic: str,
    clips: list[dict[str, Any]],
    cover: dict[str, Any],
    tried: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
    plan_error: str,
    rounds: int,
) -> str:
    """一句话说清这次到底发生了什么。

    这几句话的区别是必要的：「库里没有」是材料问题，「导演没给出可查的词」是
    出题问题，「模型没跑起来」是配置问题。三种都说成「信息不足」，用户只会一直
    点重试。
    """
    if clips:
        covered = "、".join(SLOT_LABEL[slot] for slot in cover["covered"]) or "无"
        missing = "、".join(SLOT_LABEL[slot] for slot in cover["missing"])
        text = (
            f"按「{topic}」查到 {len(clips)} 段材料（{rounds} 轮、"
            f"{len(tried)} 个检索词），覆盖{'：' + covered}"
        )
        if missing:
            text += f"；还缺{missing}"
        return text
    if plan_error:
        return (
            f"检索规划没跑起来（{plan_error}），按原句查「{topic}」也没有命中本地资料"
        )
    # 工具坏了要说「没跑通」，不能说「没有」。前者用户能修（换 Key、看日志），
    # 后者他只能去别处找材料——而这两种下一步完全不同。判据只有一份，在
    # `tool_feedback.SYSTEM_FAILURES` 里。
    from ..platform.tool_feedback import SYSTEM_FAILURES

    broken = next(
        (item for item in tried if str(item.get("error_type") or "") in SYSTEM_FAILURES),
        None,
    )
    if broken is not None:
        detail = str(broken.get("reason") or "").strip()
        return (
            f"检索没跑通，没法确认「{topic}」在库里有没有"
            + (f"（{detail}）" if detail else "")
        )
    if not tried:
        return f"没能为「{topic}」想出可执行的检索词" + (
            f"（{len(dropped)} 个词被拦下）" if dropped else ""
        )
    words = "」「".join(entry["query"] for entry in tried[:4])
    more = f" 等 {len(tried)} 个" if len(tried) > 4 else ""
    return f"导演试过「{words}」{more}检索词，本地没有可用材料"
