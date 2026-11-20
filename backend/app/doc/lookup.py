"""素材没找到时，停下来问用户：接下来从哪儿找。

以前这里是自动的：本地资料库 未命中就直接联网，联网也没有就报「信息不足，不能开拍」。
两处都不对。第一处错在替用户做了决定——联网要花时间、抓来的页面可能根本不是他要的
那件事；第二处错在把「我查过了没有」说成「你没给我东西」。

现在的规则：

  · 本地资料库 命中 → 直接用，不问。
  · 未命中 → 停下来，把「从哪儿找」交回用户。选项里包括贴正文、联网、飞书，
    以及以后接的钉钉／微信——**只有用户点了，才允许碰那个 MCP**。
  · 选的那条路也失败 → 回到同一张卡，把失败原因写上去，让他再决定。
    次数不设上限：问的是人，不是程序在自旋。

落地方式：断点即退出。`AskUser` 一路抛到 API 层，run 标成 `waiting`，把「在等什么」
存进库；用户回答后重新入队，从断点继续（`resolve`）。这样不需要一个挂着的协程，
也就不存在「进程重启把等待中的任务丢了」。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from typing import Any

from .intake import (
    Intake,
    _fetch_feishu,  # noqa: PLC2701 - 飞书那条路只此一份实现
    _search_knowledge_once,  # noqa: PLC2701 - 检索那条路只此一份实现
)
from ..platform.tool_feedback import SYSTEM_FAILURES


class AskUser(Exception):
    """停下来等用户做一次选择。

    带上 intake：回答问题时要接着用它（主题、时长、已经试过哪些路），而 run 一旦
    退出，内存里的对象就没了。序列化后存库，续跑时原样恢复。
    """

    def __init__(self, question: dict[str, Any], intake: Intake) -> None:
        super().__init__(str(question.get("title") or "需要你确认材料来源"))
        self.question = question
        self.intake = intake


class StopRun(Exception):
    """用户选了「先不做了」。这不是失败，是取消。"""


#: 选项 id → 固定含义。id 是前后端之间的契约，文案可以改，id 不能。
PROVIDE = "provide"
WEB = "web"
FEISHU = "feishu"
STOP = "stop"


def feishu_available() -> bool:
    """飞书只有在「已连上」时才出现在选项里。

    显示了但点下去报「没授权」，比不显示更糟：用户会以为是 bug，而不是以为自己
    少做了一步授权。
    """
    try:
        from ..connectors import feishu

        return str(feishu.public_status().get("state") or "") == "connected"
    except Exception:  # noqa: BLE001 - 读配置失败就当不可用
        return False


def options(*, topic: str, feishu: bool | None = None) -> list[dict[str, Any]]:
    """这次实际能走的路。做不到的不列，不写「敬请期待」。"""
    has_feishu = feishu_available() if feishu is None else bool(feishu)
    out: list[dict[str, Any]] = [
        {
            "id": PROVIDE,
            "label": "我给你正文",
            "hint": "贴一段文字，只用于这次任务，不写入知识库",
            "needs_text": True,
        }
    ]
    if topic:
        out.append(
            {
                "id": WEB,
                "label": "你去联网搜",
                "hint": "抓权威网站的正文；没抓到会回来再问你",
                "needs_text": False,
            }
        )
    if has_feishu:
        out.append(
            {
                "id": FEISHU,
                "label": "去飞书里查",
                "hint": "读取你已授权的云文档，只用于这次任务",
                "needs_text": False,
            }
        )
    out.append(
        {
            "id": STOP,
            "label": "先不做了",
            "hint": "结束这次任务，不留半成品",
            "needs_text": False,
        }
    )
    return out


def _question(
    *,
    title: str,
    reason: str,
    topic: str,
    tried: list[dict[str, Any]],
    options_: list[dict[str, Any]],
) -> dict[str, Any]:
    """一张问卡。`tried` 是「已经试过哪几条路、为什么没成」。

    它必须显示：用户点了第二次联网还是失败时，他需要知道自己到底试过几次、
    每次是超时还是网站没有正文——不然只会一直点重试。
    """
    return {
        "id": uuid.uuid4().hex[:10],
        "asked_at": time.time(),
        "phase": "material",
        "topic": topic,
        "title": title,
        "reason": reason,
        "tried": tried,
        "options": options_,
    }


def _tried_from(intake: Intake) -> list[dict[str, Any]]:
    """把已经跑过的检索整理成「试过了」清单。

    来源是 tool_calls：它记录的是真正发生过的调用，不是推测。没有它，卡上就只剩
    一句「没找到」，用户没法判断该不该再试一次。

    导演的检索会一次跑五六个词，逐个列出来卡上就装不下了，而且六行「本地资料库｜
    未命中」对用户是零信息。所以那一批压成一行，把词数写进去——用户真正需要知道
    的是「它试了很多种说法」还是「它只试了一种」，不是每一个词。
    """
    out: list[dict[str, Any]] = []
    plan = getattr(intake, "search_plan", None) or {}
    searched = [item for item in (plan.get("tried") or []) if isinstance(item, dict)]
    if searched:
        hits = [f"「{item.get('query')}」" for item in searched if item.get("found")]
        out.append({
            "source": f"本地资料库（导演出了 {len(searched)} 个检索词）",
            "ok": bool(hits),
            "reason": ("命中了 " + "、".join(hits[:3])) if hits else "都没有命中",
        })
    for call in intake.tool_calls or []:
        name = str(call.get("name") or "")
        if plan.get("tried") and name == "search_knowledge":
            # 上面那行已经说过这一批了，不再逐个重复。
            continue
        if name == "search_knowledge":
            label = "权威网站检索" if (call.get("args") or {}).get("allow_web") else "本地资料库"
        elif name in {"search_feishu_docs", "fetch_feishu_doc"}:
            label = "飞书"
        else:
            label = name
        out.append({"source": label, "ok": not call.get("failed"), "reason": ""})
    return out[:4]


def _short(text: str, limit: int = 160) -> str:
    """截到第一句。检索失败的原因常常拖着三段「任务影响／下一步」，卡上装不下。"""
    import re

    first = re.split(r"[；;]", str(text or ""), maxsplit=1)[0].strip()
    return first[:limit]


def for_material(
    intake: Intake,
    *,
    reason: str = "",
    tried: list[dict[str, Any]] | None = None,
    prefer_feishu: bool = False,
) -> dict[str, Any]:
    """有主题、但没拿到正文。

    `prefer_feishu` 只影响「要不要把飞书列出来」，不影响是否可用：用户是冲着飞书
    来的（明确提了、或链接失效），但没连通就不该出现在选项里。
    """
    topic = str(intake.topic or intake.source_query or "").strip()
    error = intake.error if isinstance(intake.error, dict) else {}
    broken = str(error.get("error_type") or "") in SYSTEM_FAILURES

    if broken:
        title = (
            f"检索没跑通，没法确认有没有「{topic}」"
            if topic
            else "检索没跑通"
        )
    elif topic:
        title = f"本地没有「{topic}」的现成材料，接下来从哪儿找？"
    else:
        title = "没有拿到可用材料，接下来从哪儿找？"

    feishu = True if (prefer_feishu and feishu_available()) else None
    return _question(
        title=title,
        reason=reason or _short(intake.reason or error.get("message") or "本地检索没有返回可用正文"),
        topic=topic,
        tried=tried if tried is not None else _tried_from(intake),
        options_=options(topic=topic, feishu=feishu),
    )


def for_missing_topic(intake: Intake) -> dict[str, Any]:
    """连主题都看不出来。没有东西可查，所以只给「贴正文」和「不做了」。

    这不是「信息不足」那种含糊说法：系统确实没有可以拿来检索的词。硬去搜一个
    猜出来的主题，最贵的结果是花几分钟做出一部不是用户要的片子。
    """
    return _question(
        title="没看出要做什么，给我一段正文或一个主题",
        reason=str(intake.reason or "原文里没有可检索的人物、事件或题目")[:300],
        topic="",
        tried=_tried_from(intake),
        options_=options(topic="", feishu=False),
    )


def _provided(intake: Intake, text: str) -> Intake:
    body = (text or "").strip()
    if not body:
        raise AskUser(
            _question(
                title="还没收到正文",
                reason="你选了「我给你正文」，但内容是空的",
                topic=str(intake.topic or ""),
                tried=_tried_from(intake),
                options_=options(topic=str(intake.topic or "")),
            ),
            intake,
        )
    title = str(intake.topic or "").strip() or "你提供的正文"
    intake.entry = {
        "id": "",
        "title": title,
        "body": body,
        "key_points": [],
        "status": "user",
        "origin": "user",
        "sources": [{"kind": "用户提供", "org": "", "title": "本次任务正文", "url": ""}],
        "note": "用户直接提供的正文，只用于本次任务；未写入知识库",
    }
    intake.source = "user"
    intake.ready = True
    intake.needs_knowledge = False
    intake.needs_external_material = False
    intake.error = None
    intake.reason = "按你的选择，直接使用你提供的正文"
    return intake


async def resolve(
    intake: Intake, answer: dict[str, Any], *, on_tool: Any = None
) -> Intake:
    """把用户在问卡上的选择变成一次取材。失败就抛一个新的 `AskUser`。

    每一轮都从**原始 intake** 重新开始：上一轮失败留下的 error / reason 不能带进
    这一轮，否则成功之后界面上还挂着上次的失败原因。
    """
    choice = str(answer.get("choice") or "").strip()

    if choice == STOP:
        raise StopRun()

    if choice == PROVIDE:
        return _provided(intake, str(answer.get("text") or ""))

    if choice == WEB:
        if not str(intake.topic or "").strip():
            raise AskUser(for_missing_topic(intake), intake)
        intake.error = None
        entry, result = await _search_knowledge_once(
            intake, allow_web=True, on_tool=on_tool
        )
        if entry is not None:
            # 命中时 `_search_knowledge_once` 已经把 entry 写进 intake 了；这里再
            # 兜一次，是因为「返回了 entry 但 intake 没变成可用」是最坏的一种失败：
            # 管线看到 ready=False 会停下来，而材料其实已经在手上。
            if not intake.entry:
                intake.entry = entry
            intake.ready = True
            intake.source = intake.source or "web"
            return intake
        reason = str(result.get("reason") or result.get("error") or "联网检索没有取到正文")
        raise AskUser(
            for_material(
                intake,
                reason=f"联网检索没拿到可用正文：{reason[:200]}",
                tried=_tried_from(intake),
            ),
            intake,
        )

    if choice == FEISHU:
        intake.error = None
        if not str(intake.source_query or intake.topic or "").strip():
            intake.source_query = str(intake.topic or "").strip()
        await _fetch_feishu(intake, on_tool=on_tool)
        if intake.ready and intake.entry:
            return intake
        raise AskUser(
            for_material(
                intake,
                reason=f"飞书里没有取到可用正文：{str(intake.reason or '')[:200]}",
                tried=_tried_from(intake),
            ),
            intake,
        )

    # 未知选项：不能静默当成「不做了」，也不能猜一个默认路去跑。
    raise AskUser(
        _question(
            title="没看懂这个选择，请重新选一次",
            reason=f"收到的选项是「{choice or '空'}」，不在这次给出的选项里",
            topic=str(intake.topic or ""),
            tried=_tried_from(intake),
            options_=options(topic=str(intake.topic or "")),
        ),
        intake,
    )


def _slim_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """存快照时把检索计划压小。

    正文不要存第二份：`intake.entry.body` 里已经有整份材料包，计划里再存一遍
    `clips[].text` 就是同一段材料写两次，而它可能有一万字。留下来的都是界面
    要用的：锚点、覆盖、查过哪些词、每段在哪儿（locator）——问卡上「试过哪些
    说法」和判定卡上「拼了几段」都靠这几项。
    """
    return {
        "anchor": plan.get("anchor") or {},
        "coverage": plan.get("coverage") or {},
        "tried": plan.get("tried") or [],
        "dropped": plan.get("dropped") or [],
        "notes": plan.get("notes") or [],
        "rounds": plan.get("rounds") or 0,
        "reason": plan.get("reason") or "",
        "plan_error": plan.get("plan_error") or "",
        "surprise": plan.get("surprise") or "",
        "questions": [
            {
                "ask": question.get("ask"),
                "slots": question.get("slots") or [],
                "queries": question.get("queries") or [],
                "hits": question.get("hits") or [],
            }
            for question in plan.get("questions") or []
            if isinstance(question, dict)
        ],
        "clips": [
            {
                "locator": clip.get("locator") or "",
                "title": clip.get("title") or "",
                "kind": clip.get("kind") or "",
            }
            for clip in plan.get("clips") or []
            if isinstance(clip, dict)
        ],
    }


def snapshot(intake: Intake) -> dict[str, Any]:
    """存进 run 的续跑快照。字段全部来自 dataclass，恢复时一一对应。"""
    data = jsonable(asdict(intake))
    if isinstance(data.get("search_plan"), dict):
        data["search_plan"] = _slim_plan(data["search_plan"])
    return data


def restore(data: dict[str, Any]) -> Intake:
    fields = set(Intake.__dataclass_fields__)  # type: ignore[attr-defined]
    return Intake(**{k: v for k, v in (data or {}).items() if k in fields})


def jsonable(value: Any) -> Any:
    """把 dataclass 里的值压成能进 JSON 的样子（tuple / set / 嵌套都过一遍）。"""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
