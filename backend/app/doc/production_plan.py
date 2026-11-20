"""导演的任务清单：开工前贴出来的那张「先做什么、交给谁、等谁」的表。

和决策卡的分工：

  · 决策卡回答「这段话被读成了什么、凭什么」——判定；
  · 清单回答「接下来分几步、每步谁做、产出什么」——规划。

两张卡都挂在导演名下，但只有清单会**原地推进**：开工前先贴出来，跑到哪一步就
改到哪一步。这条「先贴后改」是它唯一比事后记录有价值的地方——事前贴出来才能被
打断，事后补的只是流水账。

三条硬规则，写在代码里：

1. **每一步必须能被对上账。** 清单上的每一行，跑完之后都要有状态（完成 / 未完成 /
   不需要）。对不上的行不写进去——一张对不上账的计划表比没有计划表更糟，因为用户
   会拿它当依据。
2. **顺序不是排出来的，是算出来的。** 取材产出正文 → 分段消费正文产出 segments →
   逐段生产消费 segments。链路由产出的 key 决定，所以「改编和分镜并行」这种计划
   在结构上就写不出来，不靠在提示词里提醒。
3. **不能跳过。** 以前一致性失败会「跳过、本段继续」，理由是至少还有画面；但跳过
   把「修不动」伪装成了「没事」。现在这一步没做成，这一段就没完成，其余段照常跑，
   并且要说清楚是谁的问题（模型能修 → 自动返工；只有人能修 → 等用户改文字）。
"""

from __future__ import annotations

from typing import Any

#: 三步骨架。能出现在这里的只有**当前真能执行**的步骤：能力表（registry 里每个
#: agent 声明读什么写什么）里没有的东西写上去，跑起来第一步就拿不到输入。
STEP_IDS = ("material", "segment", "produce")

STEP_TITLES = {
    "material": "取材",
    "segment": "分段",
    "produce": "逐段生产",
}

#: 步骤状态。`not_needed` 和 `done` 分开：一个步骤「不需要做」和「做完了」在进度上
#: 一样（不阻塞），但在解释上完全不同——混成一个值，用户会以为系统偷偷做了什么。
STEP_STATUS = ("pending", "running", "done", "failed", "not_needed")

#: 分段状态。比步骤多一个 `rework`：被打回重做的段既不是失败也不是完成。
SEG_STATUS = ("pending", "running", "done", "failed", "rework")


def _step(card: dict[str, Any], step_id: str) -> dict[str, Any]:
    for item in card.get("steps") or []:
        if item.get("id") == step_id:
            return item
    raise KeyError(step_id)


def _segment(card: dict[str, Any], segment_id: str) -> dict[str, Any] | None:
    for item in _step(card, "produce").get("segments") or []:
        if item.get("id") == segment_id:
            return item
    return None


def _recount(card: dict[str, Any]) -> None:
    """重算「完成 / 总数」。只在状态变化后调用，读的地方永远只有一个数。

    顺序不能颠倒：③ 的状态要**先**由它下面的分段算出来，再统计步骤完成数。
    先统计再更新，得到的是上一次的数字——计数器和被统计的东西差一拍，界面上
    就是「三步都绿了，头上还写着 2/3」。
    """
    steps = card.get("steps") or []
    produce = _step(card, "produce")

    segs = produce.get("segments") or []
    if segs and produce.get("status") != "pending":
        settled_segs = [s for s in segs if s.get("status") in ("done", "failed")]
        produce["seg_done"] = len(settled_segs)
        produce["seg_total"] = len(segs)
        if len(settled_segs) < len(segs):
            # 还有段在跑：既不是完成也不是失败，别让汇总提前收敛。
            produce["status"] = "running"
        elif any(s.get("status") == "failed" for s in segs):
            produce["status"] = "failed"
        else:
            produce["status"] = "done"

    done = [s for s in steps if s.get("status") in ("done", "not_needed")]
    card["done"] = len(done)
    card["total"] = len(steps)

    if any(s.get("status") == "failed" for s in steps):
        card["state"] = "failed"
    elif card["done"] >= card["total"]:
        card["state"] = "done"
    else:
        card["state"] = "running"


def start(
    *,
    material: dict[str, Any],
    stages: list[dict[str, str]],
    rework_cap: int = 1,
) -> dict[str, Any]:
    """开工前的清单：① 已由判定决定，② 正在做，③ 等着。

    ① 在这里就写「完成」不是抢跑——取材是输入识别那一步真的做完的事（查了库、
    读了飞书、或者判定原文够用），清单只是把它记下来。② 正在跑：导演此刻正在读
    原文切段，所以下一步就是它。
    """
    route = str(material.get("route") or "user")
    title = str(material.get("title") or "").strip()
    chars = int(material.get("chars") or 0)
    chapter = str(material.get("chapter") or "").strip()
    locator = str(material.get("locator") or "").strip()

    if route == "provided":
        # 用户在问卡上贴的正文。和「直接用原文」分开：一个是本来就有，一个是
        # 找过没找到之后补进来的，清单上要能看出这个区别。
        material_step = {
            "summary": "用你贴的正文，不再检索",
            "result": f"{chars} 字；只用于这次任务，不写入知识库",
        }
    elif route == "feishu":
        material_step = {
            "summary": f"读飞书文档《{title or '指定文档'}》，取正文",
            "result": f"{chars} 字；只用于本次任务，不写入知识库",
        }
    elif route == "kb":
        where = f"《{title}》" if title else "本地资料库"
        if chapter:
            where += f" {chapter}"
        if locator:
            where += f"·{locator}"
        material_step = {
            "summary": f"查本地资料库，拿{('「' + title + '」') if title else '这次主题'}的正文",
            "result": f"已命中{where}，共 {chars} 字",
        }
    elif route == "none":
        material_step = {
            "summary": "没有取到材料，停在取材这一步",
            "result": "",
        }
    else:
        material_step = {
            "summary": "不检索，直接用你给的原文",
            "result": f"原文 {chars} 字，可直接改编",
        }

    if route == "none":
        material_status = "failed"
    elif route == "user":
        # 「不需要」不等于「没做」：原文本身够用，这一步没有可执行的动作。
        material_status = "not_needed"
    else:
        material_status = "done"

    steps = [
        {
            "id": "material",
            "index": 1,
            "title": STEP_TITLES["material"],
            "owner": "导演",
            "output": "可用正文",
            "depends": "",
            "status": material_status,
            "summary": material_step["summary"],
            "result": material_step["result"],
        },
        {
            "id": "segment",
            "index": 2,
            "title": STEP_TITLES["segment"],
            "owner": "导演",
            "output": "seg1 / seg2 / …",
            "depends": "① 取材",
            "status": "running",
            "summary": "读正文，按事件切成若干段，每段给一个可拍摄的要点",
            "result": "",
        },
        {
            "id": "produce",
            "index": 3,
            "title": STEP_TITLES["produce"],
            "owner": "分段流水线",
            "output": "每段完成审查的产物",
            "depends": "② 分段",
            "status": "pending",
            "summary": "逐段按同一条链生产，每步之后由质量审查放行",
            "result": "",
            "stages": stages,
            "segments": [],
        },
    ]

    card: dict[str, Any] = {
        "state": "running",
        "done": 0,
        "total": len(steps),
        "rework_cap": int(rework_cap),
        # 返工规则写在卡上，因为它是「计划会不会变」的唯一说明。不写，用户看到
        # 某一步被打回时会以为流程失控了。
        "rule": (
            f"某一步的审查不过，就退回上一步重做，每步最多 {int(rework_cap)} 次；"
            "重做后仍不过，这一步未完成，其余分段继续往下走。"
        ),
        # 清单的最后一步是「逐段生产」，但成片确实还要在后台拼一次。不写这一句，
        # 就会出现「清单上的步骤都做完了，用户手里却多出一个清单没写的东西」，
        # 正好违反「每行都要能对上账」。写成边界，不改行为。
        "beyond": (
            "成片在后台合成（拼接 / 字幕；模型不产出声音时会补一遍本地配音），"
            "不在这次清单里。"
        ),
        "notes": [],
        "steps": steps,
    }
    if route == "none":
        card["notes"].append("取材没有拿到可用正文，后面的步骤不会开始")
    _recount(card)
    return card


def begin_segments(
    card: dict[str, Any],
    segments: list[dict[str, Any]],
    *,
    summary: str = "",
) -> dict[str, Any]:
    """② 分段落地：把导演切出来的段挂到 ③ 下面。

    段数只有这里知道。开工前写「切成 3 段」是编的——切段是导演下一步做的事，
    这也是清单要先贴、后改的原因。
    """
    segment_step = _step(card, "segment")
    produce = _step(card, "produce")

    titles = " / ".join(str(s.get("id") or "") for s in segments) or "—"
    segment_step["status"] = "done" if segments else "failed"
    segment_step["output"] = titles
    segment_step["result"] = (
        f"切成 {len(segments)} 段：" + "、".join(str(s.get("title") or "") for s in segments)
        if segments
        else "没有切出可用分段"
    )
    if summary:
        segment_step["summary"] = summary

    stage_ids = [str(s.get("id") or "") for s in produce.get("stages") or []]
    produce["segments"] = [
        {
            "id": str(s.get("id") or ""),
            "title": str(s.get("title") or ""),
            "status": "pending",
            "stages": {sid: "pending" for sid in stage_ids},
            "note": "",
        }
        for s in segments
    ]
    if segments:
        produce["status"] = "running"
        produce["summary"] = (
            f"{len(segments)} 段并行，每段依次走："
            + " → ".join(str(s.get("label") or s.get("id")) for s in produce.get("stages") or [])
        )
    _recount(card)
    return card


def touch(
    card: dict[str, Any],
    segment_id: str,
    stage_id: str,
    status: str,
    *,
    note: str = "",
) -> dict[str, Any]:
    """更新某段某一步的状态。UI 上就是那个圆点变色。"""
    seg = _segment(card, segment_id)
    if seg is None:
        return card
    seg.setdefault("stages", {})[stage_id] = status
    if note:
        seg["note"] = note
    if status == "failed":
        seg["status"] = "failed"
    elif seg.get("status") not in ("done", "failed"):
        # 有阶段在跑，这一段就是「进行中」。已经完成/失败的段不往回改。
        seg["status"] = "running"
    _recount(card)
    return card


def settle(
    card: dict[str, Any],
    segment_id: str,
    *,
    approved: bool,
    note: str = "",
) -> dict[str, Any]:
    """某一段跑完了：通过就是完成，没通过就是未完成。没有第三种结局。"""
    seg = _segment(card, segment_id)
    if seg is None:
        return card
    # 通过时可能有阶段没标记（审查失败后 continue），补成完成，否则圆点会停在
    # 「进行中」，而这一段其实已经交付了。
    if approved:
        for key, value in (seg.get("stages") or {}).items():
            if value in ("pending", "running"):
                seg["stages"][key] = "done"
    seg["status"] = "done" if approved else "failed"
    # 通过时必须有**人**负责擦掉旧备注：段的结局是「完成」，身上还挂着上一轮
    # 写的「卡在分镜设计」「准备重跑」，读起来像没跑完——状态刷新了、备注没刷，
    # 正是用户看到的「明明跑成功了还满屏失败红字」。失败时照旧写这次的
    # 原因；传了新备注就覆写。
    if note or approved:
        seg["note"] = note
    _recount(card)
    return card


def reopen(card: dict[str, Any] | None, *, retry_ids: set[str]) -> dict[str, Any] | None:
    """把上一轮的清单重新打开，只重跑点名的段。

    重跑不是重开：上一轮已经通过审查的段原样保留（视频已经生成过，重跑一遍是
    白烧一次调用），只有点名的失败段回到「等着跑」。清单必须能表达这个区别——
    用一张全新的卡盖掉旧卡，用户会以为所有段都要重来。

    上一轮的卡找不到（旧版本运行过的任务）返回 None，调用方退回 `start()`。
    """
    if not isinstance(card, dict) or not card.get("steps"):
        return None
    try:
        material = _step(card, "material")
        segment = _step(card, "segment")
        produce = _step(card, "produce")
    except KeyError:
        return None

    for seg in produce.get("segments") or []:
        sid = str(seg.get("id") or "")
        if sid in retry_ids:
            seg["status"] = "pending"
            seg["note"] = "准备重跑"
            for key, value in (seg.get("stages") or {}).items():
                if value in ("failed", "running", "rework"):
                    seg["stages"][key] = "pending"
        else:
            # 上一轮收尾时凡是「没跑完」的都算失败段；不在 retry_ids 里说明
            # 用户只要重跑点名的几段，其余失败段保持失败原样，不悄悄重跑。
            pass

    # 步骤级：取材和分段上一轮已经做完（它们的产出在恢复的 state 里），别再
    # 显示成「等着做」；逐段生产回到进行中。
    if material.get("status") in ("failed", "pending", "running"):
        material["status"] = "done"
    if segment.get("status") in ("failed", "pending", "running"):
        segment["status"] = "done"
    produce["status"] = "running"
    card["state"] = "running"
    card["notes"] = [n for n in card.get("notes") or [] if "未完成" not in str(n)]
    note(card, f"继续生成：重跑 {'、'.join(sorted(retry_ids)) if retry_ids else '失败的段'}，已通过的段不重做")
    _recount(card)
    return card


def note(card: dict[str, Any], text: str) -> dict[str, Any]:
    if text and text not in card.setdefault("notes", []):
        card["notes"].append(text)
    return card


def finish(card: dict[str, Any], *, reason: str = "") -> dict[str, Any]:
    """收尾：还挂在「进行中」的步骤一律判成未完成。

    留着 running 是最坏的选择——任务已经结束了，界面上还有一步在转圈，用户会一直
    等下去。
    """
    for step in card.get("steps") or []:
        if step.get("status") in ("pending", "running"):
            step["status"] = "failed"
            if not step.get("result"):
                step["result"] = reason or "没有跑完"
    for seg in _step(card, "produce").get("segments") or []:
        if seg.get("status") in ("pending", "running", "rework"):
            seg["status"] = "failed"
            for key, value in (seg.get("stages") or {}).items():
                if value in ("pending", "running"):
                    seg["stages"][key] = "failed"
    if reason:
        note(card, reason)
    _recount(card)
    if card.get("state") == "running":
        card["state"] = "failed"
    return card


def headline(card: dict[str, Any]) -> str:
    """列表页那一行副标题。不写「进行中 2/3」以外的任何东西。"""
    done = int(card.get("done") or 0)
    total = int(card.get("total") or 0)
    produce = _step(card, "produce")
    seg_total = int(produce.get("seg_total") or 0)
    seg_done = int(produce.get("seg_done") or 0)

    if card.get("state") == "done":
        return f"已完成 {done}/{total}"
    if seg_total and produce.get("status") == "running":
        return f"执行中 {done}/{total} · 分段 {seg_done}/{seg_total}"
    if card.get("state") == "failed":
        return f"未完成 {done}/{total}"
    return f"执行中 {done}/{total}"
