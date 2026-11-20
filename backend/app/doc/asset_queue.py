"""全片资产登记表与串行出图队列。

为什么要有这一层
----------------
以前每段在自己的「一致性」阶段里顺手把参考图画完。两段并发时，同一个人物会被
画两次、得到两张不一样的脸、付两次钱；而每段内部还三张图同时在飞，正是图片接口
限流的来源。

现在的分工是：

  · **一致性只写提示词**。不管用户勾没勾「出图」，提示词都是交付物；
  · `link()` 把资产登记进一张全片表，键是 (类型, 规范化名称)。第二段要用第一段
    已经画好的人，直接拿条目里的图和提示词，不再自己造一个同名角色；
  · `render_all()` 只走一遍这张表：按段序、人物→场景→道具，**提交一条、等文件
    回来、再提交下一条**。任何时刻在飞的都只有一条。

表存在 `ctx.state["asset_registry"]`，跟着运行一起落库：重跑时不重新计费已经
存在的图。

为什么按名称去重而不是按 id
---------------------------
id 是每段各自申请的，同一个人在两段里本来就有两个 id；名称才是模型和用户共同
认得的那个身份。规范化只做保守的修饰词剥离（见 `canonical_name`）——把两个真实
角色误合并，比留下一个重复更难发现。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..agents.base import AgentContext


#: 三类资产的读法：(类型, manifest 里的列表键, id 键, 提示词键, 出图比例)
#:
#: 顺序就是出图顺序：人物先出，因为视频要拿人物设定稿当参考；场景次之，道具最后。
_BUCKETS: tuple[tuple[str, str, str, str, str], ...] = (
    ("character", "character_assets", "character_id", "four_view_prompt", "1:1"),
    ("scene", "scene_assets", "scene_id", "prompt", "16:9"),
    ("prop", "prop_assets", "prop_id", "prompt", "1:1"),
)


def canonical_name(raw: Any) -> str:
    """把模型写出的人物名收敛成一个人。

    与 `agents_six_impl._canonical_name` 同规则：模型会写「孙中山」「孙中山的手」
    「孙中山（特写）」，三个名字会变成三张定妆照、三个角色。只剥离明确的修饰词，
    宁可留下重复，也不要把真实的第二个人合并掉。
    """
    text = str(raw or "").strip()
    text = re.split(r"['’]s\s|\bof\b|（|\(|\s-\s", text)[0].strip()
    text = re.sub(r"(的手|的背|的侧脸|面部特写|特写|近景)$", "", text).strip()
    return text[:40]


def asset_key(kind: str, name: Any) -> str:
    """One key per asset, across every segment of the run."""
    return f"{kind}:{canonical_name(name)}"


def registry(ctx: AgentContext) -> dict[str, Any]:
    """The run-level table. Created on first use and saved with the run."""
    return ctx.state.setdefault("asset_registry", {})


def link(
    ctx: AgentContext,
    manifest: dict[str, Any],
    *,
    segment_index: int,
) -> dict[str, int]:
    """Adopt what earlier segments already own, then register what is new.

    Called by the consistency stage once the manifest is assembled. Mutates the
    manifest in place: an asset whose name is already in the table takes that
    entry's prompt and picture, so two segments cannot describe one person twice
    and end up with two faces.

    Returns counters for the trace line — a silent adoption is indistinguishable
    from "the model produced nothing".
    """
    table = registry(ctx)
    segment_id = str(manifest.get("segment_id") or "")
    reused_prompt = 0
    reused_image = 0

    for kind, bucket, id_key, prompt_key, aspect in _BUCKETS:
        for asset in manifest.get(bucket) or []:
            if not isinstance(asset, dict):
                continue
            name = asset.get("name")
            key = asset_key(kind, name)
            if not key or key == f"{kind}:":
                continue

            prompt = str(asset.get(prompt_key) or "").strip()
            entry = table.get(key)
            if entry is None:
                entry = {
                    "key": key,
                    "kind": kind,
                    "name": canonical_name(name),
                    "asset_id": str(asset.get(id_key) or ""),
                    "prompt": prompt,
                    "negative_prompt": str(asset.get("negative_prompt") or ""),
                    "aspect_ratio": aspect,
                    "owner_segment": segment_id,
                    "owner_index": segment_index,
                    "order": len(table),
                    "status": "pending" if prompt else "no_prompt",
                    "image_uri": "",
                    "image_path": "",
                    "error": "",
                    "consumers": [],
                }
                table[key] = entry
            else:
                # The first segment to register owns the look. Later segments
                # adopt its prompt instead of re-describing the same person:
                # one person must have one face across the whole film.
                if entry.get("prompt") and prompt_key == "four_view_prompt":
                    if prompt and prompt != entry["prompt"]:
                        reused_prompt += 1
                    asset["four_view_prompt"] = entry["prompt"]
                elif entry.get("prompt") and prompt_key == "prompt":
                    if prompt and prompt != entry["prompt"]:
                        reused_prompt += 1
                    asset["prompt"] = entry["prompt"]
                if entry.get("negative_prompt") and not asset.get("negative_prompt"):
                    asset["negative_prompt"] = entry["negative_prompt"]

            if entry.get("image_path"):
                asset["image_uri"] = entry["image_uri"]
                asset["image_path"] = entry["image_path"]
                reused_image += 1

            if segment_id and segment_id not in entry["consumers"]:
                entry["consumers"].append(segment_id)
            _sync_task(manifest, str(asset.get(id_key) or ""), entry)

    return {"reused_prompt": reused_prompt, "reused_image": reused_image}


def jobs(ctx: AgentContext) -> list[dict[str, Any]]:
    """Everything that still needs a picture, in the order it should be drawn.

    Segment order first, then kind, then the order the assets were registered.
    A picture already on disk is not a job: re-drawing it would bill the same
    image twice, which is the waste this module exists to remove.
    """
    table = registry(ctx)
    if not table:
        # A run saved before this module existed has manifests but no table.
        # Seed it from what is there, or a resumed ancient run would silently
        # render nothing at all.
        segments = [str(s.get("id") or "") for s in (ctx.state.get("segments") or [])]
        manifests = ctx.state.get("asset_index") or {}
        order = [sid for sid in segments if sid in manifests]
        order += [sid for sid in manifests if sid not in order]
        for index, sid in enumerate(order):
            manifest = manifests.get(sid)
            if isinstance(manifest, dict):
                link(ctx, manifest, segment_index=index)
    ordered = sorted(
        (e for e in table.values() if isinstance(e, dict)),
        key=lambda e: (int(e.get("owner_index") or 0), int(e.get("order") or 0)),
    )
    return [
        e for e in ordered
        # A failure is not permanent: the next run may well succeed, and the
        # asset review is what decides whether it is acceptable to ship without
        # it. Missing-prompt entries are listed too — `render_all` refuses them
        # by name, so the user sees which asset is unfinished instead of a
        # picture that silently never arrives.
        if not e.get("image_path")
    ]


async def render_all(ctx: AgentContext) -> dict[str, Any]:
    """Draw every picture the run still needs, one submission at a time.

    Serial on purpose. The endpoints behind this are the ones that rate-limit,
    and a serial queue also makes the order observable: the log says which
    picture is in flight, so a stall is identifiable rather than a mystery.
    """
    from .cancellation import requested
    from .media_steps import _assets_dir, _generator

    pending = jobs(ctx)
    if not pending:
        return {"total": 0, "rendered": 0, "failed": 0, "skipped": 0, "reason": ""}

    gen = _generator(ctx, "image")
    if gen is None:
        # Prompt-only, switch off, or no usable image model. The prompts stay;
        # the run continues. Recorded once rather than per asset.
        reason = str(ctx.state.get("__media_skip") or "未启用出图")
        ctx.trace_step("decide", f"未出图：{len(pending)} 个资产只保留提示词", reason)
        return {
            "total": len(pending), "rendered": 0, "failed": 0,
            "skipped": len(pending), "reason": reason,
        }

    out_dir = _assets_dir(ctx, "portraits")
    model = str(getattr(gen, "image_model", "") or "")
    rendered = 0
    failed = 0

    for index, entry in enumerate(pending, 1):
        if requested(ctx.run_id):
            ctx.trace_step("warn", "已取消，停止出图", f"已完成 {rendered} 张", ok=False)
            break

        prompt = str(entry.get("prompt") or "").strip()
        if not prompt:
            # An empty prompt is not a picture request: the endpoint either
            # errors out or draws something unrelated, and either way it is
            # billed. Refuse locally and say which asset is missing one.
            entry["status"] = "no_prompt"
            entry["error"] = "提示词为空，未提交"
            failed += 1
            ctx.trace_step(
                "warn", f"跳过 {entry.get('name')}", "没有提示词，不提交空串", ok=False
            )
            break

        ctx.state["render_progress"] = {
            "segment": str(entry.get("owner_segment") or ""),
            "asset": str(entry.get("name") or ""),
            "index": index,
            "total": len(pending),
            "kind": "image",
            "model": model,
        }
        entry["status"] = "rendering"
        ctx.trace_step(
            "tool",
            f"提交出图 {index}/{len(pending)}：{entry.get('name')}",
            f"等待上游回传后再提交下一条；{entry.get('kind')}；"
            f"比例 {entry.get('aspect_ratio')}",
        )

        try:
            image = await gen.render_image(
                prompt=prompt,
                out_dir=out_dir,
                size="1024x1024" if entry.get("aspect_ratio") == "1:1" else "1344x768",
            )
            local_path = str(getattr(image, "local_path", "") or "")
            local_file = Path(local_path) if local_path else None
            if not local_file or not local_file.is_file() or local_file.stat().st_size <= 0:
                raise RuntimeError("上游图片未成功保存到本地，停止把本条标记为完成")
        except Exception as exc:  # noqa: BLE001 - one picture must not stop the film
            entry["status"] = "failed"
            entry["error"] = str(exc)[:160]
            failed += 1
            ctx.trace_step(
                "warn", f"出图失败：{entry.get('name')}", entry["error"], ok=False
            )
            # The task rows have to say "failed" too. Left at `pending` the card
            # shows a picture that is still coming, and nobody ever tells the
            # user it is not.
            _fan_out(ctx, entry)
            # Local persistence is the queue barrier.  Advancing after this
            # failure would submit item N+1 even though item N was never safely
            # stored, contradicting the strict serial contract.
            break

        entry["status"] = "done"
        entry["image_uri"] = str(getattr(image, "url", "") or "")
        entry["image_path"] = local_path
        entry["error"] = ""
        rendered += 1
        _fan_out(ctx, entry)

    ctx.state.pop("render_progress", None)
    summary = {
        "total": len(pending),
        "rendered": rendered,
        "failed": failed,
        "skipped": len(pending) - rendered - failed,
        "reason": "",
    }
    ctx.trace_step(
        "decide" if not failed else "warn",
        f"全片出图 {rendered}/{len(pending)} 张",
        "串行提交，一条成功回传并保存到本地后才提交下一条"
        + (f"；失败 {failed} 张，后续任务保留待重试" if failed else ""),
        ok=not failed,
    )
    return summary


def _fan_out(ctx: AgentContext, entry: dict[str, Any]) -> None:
    """Copy a finished picture into every segment that asked for it.

    A picture belongs to the run, not to the segment that happened to draw it.
    Without this, re-drawing segment 1's portrait would leave segment 2 pointing
    at the old file — the same person, two faces, in one film.
    """
    for manifest in (ctx.state.get("asset_index") or {}).values():
        for kind, bucket, id_key, _prompt_key, _aspect in _BUCKETS:
            if entry.get("kind") != kind:
                continue
            for asset in (manifest or {}).get(bucket) or []:
                if not isinstance(asset, dict):
                    continue
                if asset_key(kind, asset.get("name")) != entry.get("key"):
                    continue
                asset["image_uri"] = entry.get("image_uri") or ""
                asset["image_path"] = entry.get("image_path") or ""
                _sync_task(manifest, str(asset.get(id_key) or ""), entry)


def _sync_task(
    manifest: dict[str, Any], asset_id: str, entry: dict[str, Any]
) -> None:
    """Keep one asset's task row in step with the registry.

    The task list is what the UI and the asset review read; a manifest whose
    picture exists but whose task still says `pending` reads as "not rendered".
    Matching is by this segment's own asset id — segments issue their own ids,
    and the registry only remembers the owner's.
    """
    if not asset_id:
        return
    for task in manifest.get("image_generation_tasks") or []:
        if not isinstance(task, dict):
            continue
        if str(task.get("asset_id") or "") != asset_id:
            continue
        if entry.get("image_path"):
            task["status"] = "done"
            task["image_uri"] = entry.get("image_uri") or ""
            task["local_path"] = entry.get("image_path") or ""
            task.pop("error", None)
        elif entry.get("status") == "failed":
            # A failure is recorded on every segment that asked for this picture.
            # Left as `pending`, the card promises a render that is not running.
            task["status"] = "failed"
            task["error"] = str(entry.get("error") or "")[:160]
        else:
            task["status"] = "pending"


def pending_count(ctx: AgentContext) -> int:
    """How many pictures the whole run still needs. Used by the UI summary."""
    return len(jobs(ctx))
