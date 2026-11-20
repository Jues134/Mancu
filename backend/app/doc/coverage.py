"""全片覆盖检查：所有分段跑完之后，整部片子到底拍到了什么。

为什么不在段内判：一段通过审查，只说明它完成了**自己那一份**任务。全片有没有漏掉
一条必现项，只有把各段的产物按 requirement_id 汇总才看得出来。这两件事混在一起时，
系统会犯两种相反的错——要么拿全片的清单去卡每一段（撤离段被迫拍开会，返工到死），
要么各段都"通过"了却没人发现南湖游船根本没拍。

三档状态是有意分开的：
  covered      分镜里有对应镜头，且该镜头渲染成功
  planned_only 分镜里有，但没有渲染（提示词模式，或渲染失败/未开）
  missing      没有任何段认领，或认领了但分镜里没出现

`planned_only` 不能算通过，也不能算失败：提示词模式下本来就不渲染，把它当失败会把
一个正常配置报成故障；但当渲染开启时，它确实是"还没拍到"。
"""

from __future__ import annotations

from typing import Any

COVERED = "covered"
PLANNED = "planned_only"
MISSING = "missing"


def _shots_of(storyboards: dict[str, Any], segment_id: str) -> list[dict[str, Any]]:
    board = storyboards.get(segment_id) or {}
    shots = board.get("shots") or []
    return [s for s in shots if isinstance(s, dict)]


def _rendered_ok(shot_assets: list[dict[str, Any]], shot_id: str) -> bool:
    for a in shot_assets:
        if str(a.get("shot_id") or "") == shot_id and a.get("ok"):
            return True
    return False


def check(
    requirements: list[dict[str, Any]],
    by_segment: dict[str, list[dict[str, Any]]],
    storyboards: dict[str, Any],
    shot_assets: list[dict[str, Any]],
    *,
    rendered: bool,
) -> dict[str, Any]:
    """按 requirement_id 汇总完成情况。

    `rendered=False` 时（提示词模式或未开视频）不把"没有视频"当缺失，只报
    `planned_only`，并把原因写进 note——否则每次提示词运行都会报一堆假缺失。
    """
    items: list[dict[str, Any]] = []
    warnings: list[str] = []

    for req in requirements:
        rid = str(req.get("requirement_id") or "")
        owners = [str(x) for x in (req.get("segments") or [])]
        text = str(req.get("requirement") or "")

        if not owners:
            items.append(
                {"requirement_id": rid, "requirement": text, "status": MISSING,
                 "segments": [], "shots": [], "note": "没有分配到任何分段"}
            )
            warnings.append(f"{rid}「{text}」没有分配到任何分段，不会有人拍")
            continue

        hits: list[str] = []
        for sid in owners:
            for shot in _shots_of(storyboards, sid):
                ids = [str(x) for x in (shot.get("requirement_ids") or [])]
                if rid in ids:
                    hits.append(str(shot.get("shot_id") or ""))

        if not hits:
            items.append(
                {"requirement_id": rid, "requirement": text, "status": MISSING,
                 "segments": owners, "shots": [],
                 "note": f"已分配 {('、'.join(owners))}，但分镜里没有镜头认领"}
            )
            warnings.append(f"{rid}「{text}」分配给了 {'、'.join(owners)}，但分镜里没有对应镜头")
            continue

        if not rendered:
            status, note = PLANNED, "分镜已完成，未渲染视频（提示词模式或媒体开关关闭）"
        else:
            ok = [s for s in hits if _rendered_ok(shot_assets, s)]
            if ok:
                status, note = COVERED, ""
            else:
                status, note = PLANNED, "分镜有，但没有渲染成功的镜头"

        items.append(
            {"requirement_id": rid, "requirement": text, "status": status,
             "segments": owners, "shots": hits, "note": note}
        )

    counts = {COVERED: 0, PLANNED: 0, MISSING: 0}
    for it in items:
        counts[it["status"]] = counts.get(it["status"], 0) + 1

    unassigned = [str(r.get("requirement_id") or "") for r in requirements
                  if not (r.get("segments") or [])]

    return {
        # 只有"完全覆盖"才算 ok。planned_only 是"还没拍到"，不能对外说完成。
        "ok": bool(requirements) and counts[COVERED] == len(requirements),
        "total": len(requirements),
        "covered": counts[COVERED],
        "planned_only": counts[PLANNED],
        "missing": counts[MISSING],
        "unassigned": unassigned,
        "items": items,
        "warnings": warnings,
        "has_requirements": bool(requirements),
    }


def summary_line(report: dict[str, Any]) -> str:
    """一句话给人看，不啰嗦。"""
    if not report.get("has_requirements"):
        return "全片覆盖：导演没有给出必现项清单，未做覆盖检查"
    total = report["total"]
    return (
        f"全片覆盖：{report['covered']}/{total} 条已拍"
        + (f"，{report['planned_only']} 条仅分镜" if report["planned_only"] else "")
        + (f"，{report['missing']} 条缺失" if report["missing"] else "")
    )
