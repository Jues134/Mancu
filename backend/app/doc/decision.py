"""导演决策卡：把「输入判定」从一句结论做成一张能读的卡。

以前这一步在界面上只有一行「输入识别：主题要求，先查本地知识库」，读者要自己
从五个固定标签里猜它到底读了什么、凭什么这么判。卡回答四个问题：收到了什么、
读出了什么、判成了什么、接下来干什么。

三条硬规则，写在代码里而不是文档里：

1. **第一段只分诊，不下结论。** 「是素材还是主题」属于判定段；第一段若也说了，
   一张卡上就有两处判定，早晚说岔。所以第一段只说形态和我要先做什么。
2. **带「」的引用必须是原文子串。** 不是子串就丢掉——显示一条假引用，比不显示
   更糟，因为读者会信它。
3. **不产生置信度，也不产生进度百分比。** 没有数据支撑的数字会被读成事实。

字段全部来自一次判定（`intake.judge` 的返回值）加上对原文的客观测量。测量部分
不依赖模型，模型胡说时它是唯一能对账的东西。
"""

from __future__ import annotations

import re
from typing import Any

#: 「要成片」的说法。允许动词和名词之间隔着题材（「做一个××的视频」），但不跨句号：
#: 跨过去就会把整段材料当成制作要求，而那一整段是正文。
_VIDEO_INTENT = re.compile(
    r"(?:帮我|请|给我|我想|我要)?(?:做|制作|生成|剪|出)[^，。；！？\n]{0,12}?"
    r"(?:视频|短片|纪录片|片子|成片)"
)
#: 只写了名词、没有动词时，至少引那个名词——比什么都不引强。
_VIDEO_NOUN = re.compile(r"(?:视频|短片|纪录片|片子|成片)")
#: 时长说法。取到的是原文片段，直接当引用用。
_DURATION = re.compile(r"(?:\d+|[一二两三四五六七八九十半])\s*(?:秒钟|秒|分钟|分)")
#: 链接。带链接的输入要先解析，不能拿链接本身去分类。
_LINK = re.compile(r"https?://\S+")
#: 用户明确说不要查资料。这是唯一能压过模型的规则信号。
_NO_RESEARCH = re.compile(r"(?:不要|不用|别|无需)(?:去)?(?:查|检索|找)(?:资料|材料|文献)?")
_FEISHU = re.compile(r"飞书|feishu|larksuite", re.IGNORECASE)

#: 「检索本身没跑通」的判据只有一份，在 tool_feedback 里。判定卡和问卡各写一份，
#: 就会出现一张卡说「没找到」、另一张说「坏了」。
from ..platform.tool_feedback import SYSTEM_FAILURES

#: 验收位的名字只有一份，在 director_search 里。这里再写一份「何时/何地/何人」，
#: 改了一处就会有一处过时。
from .director_search import SLOT_LABEL


def _broken(intake: Any) -> bool:
    error = getattr(intake, "error", None)
    kind = str((error or {}).get("error_type") or "") if isinstance(error, dict) else ""
    return kind in SYSTEM_FAILURES


# 「有没有事件叙述」的判据和 intake 共用一份：两处各写一套，结果就是一边说这是
# 素材、一边说这是主题。私有名照旧引用，就是为了不出现第二份实现。
from .intake import _EVENT  # noqa: PLC2701,E402 - 同一份判据

#: 判定 → 卡片标题。结论必须能一眼读完，不写"倾向"这种半句话。
#:
#: `insufficient` 不在这里：它有两种完全不同的处境（没给主题 / 查过没有），
#: 合成一句就会说出「信息不足，不能开拍」——而用户明明给了主题，界面却在让他
#: 补一句具体的人或事。标题由 `_verdict()` 按实际情况生成。
_VERDICTS: dict[str, tuple[bool, str]] = {
    "topic_request": (True, "需要查资料 — 先检索本地资料库"),
    "question": (True, "需要查资料 — 先检索本地资料库"),
    "feishu_request": (False, "调用飞书云文档，不查本地资料库"),
    "material": (False, "不需要查资料 — 原文可以直接改编"),
    "story": (False, "不需要查资料 — 原文可以直接改编"),
}


def _quote(pattern: re.Pattern[str], text: str) -> str:
    """原文里的命中片段；没命中返回空串。

    只返回 `finditer` 真的找到的文本，所以「引用必须是原文子串」这条规则是结构性
    保证的，不靠调用方自觉。
    """
    match = pattern.search(text or "")
    return match.group(0).strip() if match else ""


def _has_event(body: str) -> bool:
    return bool(_EVENT.search(body or ""))


def _received(text: str, *, attachments: int) -> dict[str, Any]:
    """第一段：收到了什么，我要先做什么。只描述形态，不给结论。"""
    body = text or ""
    links = len(_LINK.findall(body))
    chars = len(body.strip())

    if not chars:
        form = "用户发来一段空输入"
        triage = "没有内容可判，不再往下走"
    elif links:
        form = f"用户发来一段文字（{chars} 字，其中 {links} 个链接）"
        triage = "先把链接里的内容取出来，再判断它够不够用"
    elif attachments:
        form = f"用户发来一段文字（{chars} 字，{attachments} 个附件）"
        triage = "先看清附件和这段文字的关系，再判断它够不够用"
    else:
        form = f"用户发来一段文字（{chars} 字，无附件、无链接）"
        triage = "我先弄清它是什么：是要我照着改编，还是要我先把内容找出来。"

    return {"line": form, "triage": triage, "chars": chars, "links": links}


def _understanding(text: str, intake: Any) -> list[dict[str, str]]:
    """第二段：读出了什么。每行都必须能在原文里指出来，指不出来就明说没有。"""
    body = text or ""
    rows: list[dict[str, str]] = []

    intent = str(getattr(intake, "intent", "") or "")
    seconds = int(getattr(intake, "desired_seconds", 0) or 0)
    topic = str(getattr(intake, "topic", "") or "")

    video = _quote(_VIDEO_INTENT, body) or _quote(_VIDEO_NOUN, body)
    if video:
        rows.append({"label": "要做什么", "value": "制作一条视频", "quote": video})
    elif intent in {"material", "story", "feishu_request"}:
        # 这类输入走的就是成片流程，但原文没说。写清楚"原文没写"比假装它写了干净。
        rows.append(
            {"label": "要做什么", "value": "制作一条视频（原文没写成品，按流程默认）", "quote": ""}
        )
    elif intent == "question":
        rows.append({"label": "要做什么", "value": "先把这个问题讲清楚", "quote": ""})
    else:
        rows.append({"label": "要做什么", "value": "未从原文看出要成片", "quote": ""})

    if topic:
        # 主题可能是模型归纳的，不一定是原文片段——只有真的是子串才当引用。
        rows.append(
            {
                "label": "题材",
                "value": topic,
                "quote": topic if topic in body else "",
            }
        )
    else:
        rows.append({"label": "题材", "value": "未从原文识别到", "quote": ""})

    if seconds:
        rows.append(
            {
                "label": "时长",
                "value": f"{seconds} 秒",
                "quote": _quote(_DURATION, body),
            }
        )
    else:
        rows.append({"label": "时长", "value": "未指定，按默认时长", "quote": ""})

    narrative = _has_event(body)
    if narrative:
        rows.append({"label": "是否已含正文", "value": "是，原文里有事件叙述", "quote": ""})
    else:
        rows.append({"label": "是否已含正文", "value": "否，只有主题和制作要求", "quote": ""})

    return rows


def _tried(intake: Any) -> dict[str, str] | None:
    """「已经查过哪儿」——单独一行，来自真实发生过的调用。

    这一行是用户决定「要不要让他联网」的唯一依据。没有它，卡上只有一句「没找到」，
    读者分不清是本地没有、还是工具坏了。
    """
    calls = getattr(intake, "tool_calls", None) or []
    names = [str(c.get("name") or "") for c in calls]
    if "search_knowledge" in names:
        web = any(
            (c.get("args") or {}).get("allow_web")
            for c in calls if c.get("name") == "search_knowledge"
        )
        text = "本地资料库 已查过，没有命中可用正文"
        if web:
            text += "；联网也试过，没有取到权威正文"
        return {"text": text, "from": "程序测量"}
    if any(n in {"search_feishu_docs", "fetch_feishu_doc"} for n in names):
        return {"text": "飞书已查过，没有取到可用正文", "from": "程序测量"}
    return None


def _basis(text: str, intake: Any, *, need_kb: bool) -> list[dict[str, str]]:
    """第三段的依据。三行分别来自：程序测量、规则判断、模型自述。

    分开写而不是揉成一句，是因为它们的可信度不同：测量不会错、规则可以被反驳、
    模型自述可能是它编的。混在一起，读者就没有办法质疑其中一部分。
    """
    body = text or ""
    chars = len(body.strip())
    narrative = _has_event(body)
    out: list[dict[str, str]] = []

    measured = f"原文 {chars} 字，{'有' if narrative else '没有'}时间、地点或事件叙述"
    measured += "，可以直接改编" if narrative else "，不足以直接改编"
    out.append({"text": measured, "from": "程序测量"})

    tried = _tried(intake)
    if tried:
        out.append(tried)

    no_research = _quote(_NO_RESEARCH, body)
    if no_research:
        out.append({"text": f"你说了「{no_research}」，按你的要求直接用原文", "from": "规则判断"})
    elif _FEISHU.search(body):
        out.append({"text": "提到飞书，只检索你已授权的云文档", "from": "规则判断"})
    elif need_kb:
        out.append({"text": "未提及飞书或外部资料，也没有「不要查资料」", "from": "规则判断"})
    else:
        out.append({"text": "未要求检索，按已有内容处理", "from": "规则判断"})

    reason = str(getattr(intake, "reason", "") or "").strip()
    if reason:
        out.append({"text": reason, "from": "导演自述"})

    return out


def _verdict(intake: Any) -> tuple[bool, str]:
    """判定标题。它必须说清「本地到底有没有」，那是用户下一步的唯一依据。

    「信息不足 — 还不能开拍」这句被去掉了：它把两种处境说成了一种。查过没有，
    下一步是选一个来源；没给主题，下一步才是补一句话。合在一起，等于让用户去
    补一句他已经说过的话。
    """
    intent = str(getattr(intake, "intent", "") or "")
    topic = str(getattr(intake, "topic", "") or "").strip()
    # 有 entry 就等于拿到了正文：ready 是管线的开关，entry 是事实。只看 ready 会
    # 在调用方漏设它时把一张已经命中材料的卡说成「本地没有」。
    ready = bool(getattr(intake, "ready", False)) or bool(getattr(intake, "entry", None))
    need_kb = intent in {"topic_request", "question"}
    source = str(getattr(intake, "source", "") or "")

    if intent == "insufficient":
        return False, "没看出要做什么 — 等你补一句"
    if ready and source == "user":
        # 用户自己贴的正文。说成「本地资料库 已命中」是错的：他贴的东西刚从手上进来，
        # 系统一个字都没检索。
        return need_kb, "用你提供的正文 — 直接开拍"
    if ready and source == "feishu":
        return need_kb, "已从飞书取到正文 — 直接开拍"
    if need_kb:
        if ready:
            return True, "需要查资料 — 本地资料库已命中"
        if _broken(intake):
            # 检索本身挂了和「库里没有」是两件事。说成后者，用户会去翻材料，
            # 而真正要修的是配置或额度。
            return True, "检索没跑通 — 等你选从哪儿找"
        where = f"本地没有「{topic}」" if topic else "本地资料库没有现成正文"
        return True, f"{where} — 等你选从哪儿找"
    if intent == "feishu_request" and not ready:
        return False, "飞书里没取到正文 — 等你选从哪儿找"
    return _VERDICTS.get(intent, (need_kb, "先查本地资料库再决定"))


def _next_step(intake: Any, *, need_kb: bool, target_seconds: int) -> dict[str, str]:
    """第四段：接下来干什么。命中就报命中，没命中就说下一步等谁决定。"""
    entry = getattr(intake, "entry", None) or {}
    title = str(entry.get("title") or "").strip()
    source = str(getattr(intake, "source", "") or "")
    ready = bool(getattr(intake, "ready", False)) or bool(entry)

    if getattr(intake, "needs_external_material", False):
        if title:
            return {"line": f"读取飞书文档《{title}》", "hint": "只用于本次任务，不写入知识库"}
        return {
            "line": "飞书里没有取到这篇文档，已停下来等你选",
            "hint": "可以贴正文、让我联网另找，或换一篇飞书文档",
        }

    # 用户贴的正文走这里，不需要经过「知识库」那一支：他要的东西已经进来了。
    if ready and source == "user":
        chars = len(str(entry.get("body") or ""))
        return {
            "line": f"用你提供的正文（{chars} 字）开拍",
            "hint": "只用于这次任务，不写入知识库",
        }

    if need_kb:
        # 导演按自己想的词查回来之后，材料是一份由几段拼成的包，标题就是主题名。
        # 说「命中《建国》」会让用户以为库里有一份叫《建国》的资料——那是把
        # 「主题」当成了「文档」。
        plan = getattr(intake, "search_plan", None) or {}
        clips = plan.get("clips") or []
        if ready and clips:
            covered = "、".join(
                SLOT_LABEL.get(slot, slot)
                for slot in (plan.get("coverage") or {}).get("covered") or []
            )
            return {
                "line": f"从本地资料里拼出 {len(clips)} 段材料" + (f"（覆盖{covered}）" if covered else ""),
                "hint": "每一段都带出处，不再联网",
            }
        if ready and title:
            where = "已上传的整本资料" if source == "kb" else "权威网站"
            return {
                "line": f"本地资料库 已命中《{title}》（{where}）",
                "hint": "直接分发任务，不再联网",
            }
        if ready:
            return {"line": "本地资料库 已命中可用正文", "hint": "直接分发任务，不再联网"}
        return {
            "line": "本地没有现成正文，已停下来等你选从哪儿找",
            "hint": "可以贴正文、让我联网搜、去飞书里查，也可以先不做",
        }

    if getattr(intake, "needs_clarification", False):
        return {
            "line": "提到了飞书但没说查哪篇，已停下来等你确认",
            "hint": "给出文档标题、关键词或飞书链接",
        }

    if not ready:
        return {
            "line": "没看出要做什么，已停下来等你补一句",
            "hint": "写清主题（比如「做一个××的生平视频」），或直接贴一段正文",
        }

    seconds = target_seconds or int(getattr(intake, "desired_seconds", 0) or 0)
    tail = f"，目标 {seconds} 秒" if seconds else ""
    return {"line": f"直接分发任务，不检索{tail}", "hint": ""}


def build(
    *,
    text: str,
    intake: Any,
    attachments: int = 0,
    target_seconds: int = 0,
    searching: bool = False,
) -> dict[str, Any]:
    """组装卡片 payload。纯函数：同样的输入永远得到同样的卡。

    `searching=True` 是检索还没跑完时的中间态。判定卡本来要等检索结束才发——
    但那意味着用户先看到一串工具调用，不知道导演把他的话读成了什么。现在分类一
    完成就先发这张「正在查」的卡，检索结束后用同一个 msg_key 原地覆写成完整版。
    中间态只改判定和下一步两段：收到/原文/理解在分类完成后就是事实，不依赖检索。
    """
    intent = str(getattr(intake, "intent", "") or "")
    if searching:
        need_kb = intent in {"topic_request", "question"}
        title = "需要查资料 — 正在检索本地资料库"
        nxt = {
            "line": "导演正在想检索词，逐个去查本地资料库",
            "hint": "查完这张卡会原地更新，不用刷新",
        }
    else:
        need_kb, title = _verdict(intake)
        nxt = _next_step(intake, need_kb=need_kb, target_seconds=target_seconds)

    return {
        "received": _received(text, attachments=attachments),
        "source": {"text": text or "", "chars": len((text or "").strip())},
        "understanding": _understanding(text, intake),
        "verdict": {
            "title": title,
            "need_kb": need_kb,
            "intent": intent,
            "basis": _basis(text, intake, need_kb=need_kb),
        },
        "next": nxt,
    }
