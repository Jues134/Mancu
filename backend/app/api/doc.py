"""HTTP surface for the documentary pipeline.

One endpoint starts a run, one streams its state. The run executes as a background
task because the film takes minutes (video generation alone is ~150 s per shot),
and a request that waits for it would time out in every browser and proxy.

Progress is written to the message table as it happens, so the UI shows the agents
working rather than a frozen spinner. That is also how the "several agents at once"
claim becomes visible: each segment's stage emits its own event.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .. import db, logs
from ..platform import executor as pf_exec
from ..platform.registry import registry

router = APIRouter()

#: One logger for the run lifecycle. Everything the pipeline reports through the
#: trace system is for the *user*; this is for whoever has to work out afterwards
#: why it stopped.
logger = logs.get_logger("mancu.run")


class CreateRun(BaseModel):
    #: 正文，或一句主题式请求（「帮我做一个关于党的诞生的视频，1 分钟」）。
    #: 这里不设字数门槛：材料够不够由导演判断——它看的是「有没有可拍的情节」，
    #: 而「我爷爷 1949 年在码头扛包」只有十几个字，也是完整的一段素材。
    chapter_text: str = Field(min_length=1, description="正文，或一句主题式请求")
    #: Reference length for this material, not a requirement. The storyboard
    #: derives the real duration from the narration, dialogue and action each shot
    #: needs; this value only sizes the token budget and gives the trace something
    #: to compare against. Clients may omit it entirely.
    target_seconds: int = Field(
        default=60, ge=15, le=300,
        description="材料体量参考（秒），非硬性目标；实际时长由内容决定",
    )
    max_rework: int = Field(default=1, ge=0, le=3, description="最多返工次数")
    #: How strictly this run is reviewed.
    #:
    #: STRICT — item-by-item checking of facts, source fidelity and the
    #:          user’s explicit requirements. P0/P1/P2 all block.
    #: NORMAL — the delivery standard (default). P0/P1 block.
    #: FAST   — draft and batch work. Only P0 blocks.
    #:
    #: Blank means "let the director decide", which resolves to NORMAL. Accepting
    #: a blank value keeps every existing client working unchanged.
    review_mode: str = Field(
        default="",
        description="审核模式 STRICT / NORMAL / FAST；留空由导演决定，默认 NORMAL",
    )
    #: Whether to render the character portraits as well as write their prompts.
    #:
    #: Off by default: a prompt is always produced, and switching this on is the
    #: only thing that spends image credit. A failed or missing image model must
    #: leave the prompt intact — that is the point of the mode.
    generate_images: bool = Field(
        default=False, description="是否同时生成定妆照（关闭则只出提示词）"
    )
    #: Whether to render each shot as video as well as write its prompt.
    generate_videos: bool = Field(
        default=False, description="是否同时生成分镜视频（关闭则只出提示词）"
    )

    #: Voice samples the user uploaded, each bound to a character.
    #:
    #: Shape: [{"character_name": "陈独秀", "uri": "/data/uploads/xxx.mp3"}].
    #: A character without an entry keeps the model's own voice — the documented
    #: fallback, not an error. Entries naming an unknown character come back as a
    #: warning rather than being ignored silently.
    voice_uploads: list[dict[str, Any]] = Field(
        default_factory=list, description="用户上传的角色声音，绑定到具体人物"
    )


def _check_media_switches(payload: "CreateRun") -> None:
    """Validate switches without coupling image and video rendering.

    Prompts are always produced first. Images and videos are independent choices
    after that point, so a video-only run is valid and simply has no generated
    consistency reference pictures.
    """
    del payload



def _check_material(text: str) -> None:
    """只挡「什么都没写」，不再数字数。

    以前这里是「≥150 字，否则必须是主题式请求」。两个方向都错过：一个一百字的
    完整故事被拒（它本来就能拍），而「帮我做个视频」这种什么也没说的请求，因为
    含「视频」两个字反而放行了，最后由导演拿着空材料硬凑。

    材料够不够，导演在这一轮里判断得出——它读的是情节，不是长度。接口只负责
    让请求进来，把判断留给能判断的那一层。
    """
    if text.strip():
        return
    raise HTTPException(
        422,
        "请先写一段材料（人物、情节、场景都行），"
        "或者写成「帮我做一个关于××的视频，1 分钟左右」这样带明确主题的请求",
    )


def _active_versions():
    """Regex matching the active agent-definition versions."""
    import re

    return re.compile(r"^3\.")


@router.get("/api/doc/agents")
async def list_agents() -> dict:
    """The active roster, with what each one does."""
    from ..doc.roster import ensure_registered
    from ..doc.tts import list_supported

    roster = ensure_registered()
    cat = [a for a in registry.catalog() if _active_versions().match(a["current_version"])]
    return {
        "agents": cat,
        "tts": list_supported(),
        "roster": roster,
        "note": "6 个 Agent：导演 / 内容改编 / 剧本分镜 / 一致性 / 视频生成 / 质量审查",
    }


@router.get("/api/doc/capabilities")
async def capabilities() -> dict:
    """What this machine can actually do, before a run is started."""
    from ..doc.consistency import describe_limits
    from ..doc.generator import Generator
    from ..doc.tts import list_supported
    from ..doc import media_mode, review_mode

    # Whether media can be RENDERED is two questions, not one: is generation
    # switched on, and is there an endpoint to call. Asking only the second made a
    # prompt-only install look broken — `Generator.from_settings()` succeeds with
    # any keyed provider, so the page said "ready" while nothing was ever rendered,
    # and the switches beside it promised images that would never appear.
    prompt_only = media_mode.prompt_only()
    can_generate = True
    reason = ""
    if prompt_only:
        reason = media_mode.describe()
    else:
        try:
            Generator.from_settings()
        except Exception as exc:  # noqa: BLE001
            can_generate = False
            reason = str(exc)[:160]

    # The upload rules are reported here rather than hard-coded in the UI: the
    # accepted formats and size ceilings live in doc/uploads.py, and a client that
    # invents its own list goes stale the moment one is added.
    from ..doc.uploads import describe_accepted

    return {
        "can_generate": can_generate,
        "reason": reason,
        # The mode is stated explicitly so the page can label the run honestly:
        # "提示词模式" is the intended state for a machine with no local renderer,
        # not a failure to be reported in red.
        "media_mode": media_mode.media_mode(),
        "prompt_only": prompt_only,
        "media_note": media_mode.describe(),
        "tts": list_supported(),
        "consistency": describe_limits(),
        "video_seconds_per_shot": 5,
        # The picker beside 开始生成 renders this list rather than hard-coding
        # three labels: what a mode blocks is decided in doc/review_mode.py, and a
        # UI owning a second copy of that sentence is how "快速审核" ends up
        # advertising behaviour the pipeline no longer has.
        "review_modes": review_mode.options(),
        "uploads": describe_accepted(),
        "voice": {
            # How a character's voice is decided. The UI states this next to the
            # audio upload control so the fallback is never a surprise.
            "binding": "上传声音时必须指定属于哪个角色；未上传的角色使用视频模型生成的声音。",
            "fallback": "model",
            "audio_kind": "audio",
        },
        "note": (
            "提示词模式：只产出提示词与校验，不渲染画面。"
            if prompt_only
            else "画面由视频模型生成（图生视频，以定妆照为首帧保证人物一致）；"
            "配音优先 edge-tts；字幕由 ffmpeg 烧录。"
        ),
    }


@router.post("/api/doc/runs")
async def create_run(payload: CreateRun, background: BackgroundTasks) -> dict:
    from ..doc.roster import ensure_registered

    ensure_registered()
    _check_material(payload.chapter_text)
    _check_media_switches(payload)
    run_id = uuid.uuid4().hex[:12]
    db.create_run(
        {
            "chapter_text": payload.chapter_text,
            "target_seconds": payload.target_seconds,
            "workflow_id": "doc_pipeline",
            # 续跑时要原样重放这次请求：run 一旦停在「等你回答」，内存里的
            # payload 就没了，只剩库里这一份。不存它，续跑就不知道当时开没开
            # 出片、审核模式是什么。
            "options": payload.model_dump(),
        },
        run_id=run_id,
    )
    if not pf_exec.mark_active(run_id):
        raise HTTPException(409, "该运行已在执行中")

    db.add_message(
        run_id, role="user", agent="human", kind="request",
        # Store what the user actually typed, not a label. A conversation view
        # that replaces the input with "开始生成纪录片" cannot be audited: the
        # reader has no way to check whether the film was made from the text they
        # meant to submit.
        content=payload.chapter_text,
        payload={
            "chars": len(payload.chapter_text),
            "target_seconds": payload.target_seconds,
        },
    )
    background.add_task(_execute, run_id, payload)
    return {"run_id": run_id, "status": "queued", "target_seconds": payload.target_seconds}


async def _run_pipeline(**kwargs: Any) -> dict[str, Any]:
    """Run the active document pipeline."""
    from ..doc.pipeline_six import run_pipeline_six

    return await run_pipeline_six(**kwargs)


def _plan_segment_titles(run_id: str) -> dict[str, str]:
    """从清单卡里拿「段 id → 标题」。

    结果消息只拿到段 id（seg1/seg2），「哪一段」得看清单卡——分段标题是导演
    起的，只有它存了一份。拿不到就退回用 id，信息少一点好过编一个。
    """
    try:
        row = db.get_message_by_key(run_id, "director:plan")
    except Exception:  # noqa: BLE001 - 标题只是点缀，不能挡住结果消息
        return {}
    steps = ((row or {}).get("payload") or {}).get("steps") or []
    produce = next((s for s in steps if s.get("id") == "produce"), {})
    return {
        str(seg.get("id")): str(seg.get("title") or "")
        for seg in produce.get("segments") or []
        if seg.get("id")
    }


def _failure_details(
    segments: list[dict[str, Any]], titles: dict[str, str]
) -> list[dict[str, str]]:
    """每个未通过段一行「卡在哪一步、为什么」。

    两类失败分开说：阶段抛了异常的（超时、审核拦截——错误类型给一句人话），
    和审查没通过的（返工次数用尽——第一条问题就是原因）。混成一句「未完成」，
    用户分不清该点「继续」还是该先改东西。
    """
    from ..tools.llm import ERROR_TYPES  # noqa: PLC0415 - 避免模块级循环导入

    from ..doc.pipeline_six import STAGE_LABELS  # noqa: PLC0415

    out: list[dict[str, str]] = []
    for seg in segments:
        if seg.get("approved"):
            continue
        sid = str(seg.get("segment_id") or "?")
        stages = [s for s in seg.get("stages") or [] if isinstance(s, dict)]
        broken = next((s for s in stages if s.get("ok") is False), None)
        if broken is not None:
            agent = str(broken.get("agent") or "")
            kind = str(broken.get("error_type") or "")
            reason = str(ERROR_TYPES.get(kind) or "").strip()
            out.append({
                "segment_id": sid,
                "title": titles.get(sid, ""),
                "stage": STAGE_LABELS.get(agent, agent),
                "error_type": kind,
                "reason": reason,
                "message": str(broken.get("error") or "")[:200],
                "retryable": kind
                in ("timeout", "transport", "unreachable", "rate_limit", "http_error",
                    "bad_response", "truncated", "empty_reply"),
            })
            continue
        issues = [i for i in seg.get("issues") or [] if isinstance(i, dict)]
        reworks = max(0, int(seg.get("attempts") or 0) - 1)
        out.append({
            "segment_id": sid,
            "title": titles.get(sid, ""),
            "stage": "质量审查",
            "error_type": "review_rejected",
            "reason": f"审查未通过（返工 {reworks} 次后未过）",
            "message": str(
                (issues[0].get("summary") or issues[0].get("description") or "") if issues
                else ""
            )[:200],
            "retryable": True,
        })
    return out


async def _execute(
    run_id: str, payload: CreateRun | None = None, retry: dict[str, Any] | None = None
) -> None:
    """跑一次任务；`payload` 为空表示这是一次续跑，`retry` 非空表示断点重跑。

    续跑与首次执行走同一个函数，是因为它们要做的事完全一样——区别只有两个：
    请求参数从库里恢复而不是从内存拿，以及把上次的回答交给管线。分成两个函数，
    迟早有一处忘记补 `finally`，留下一个永远显示「进行中」的 heartbeat。

    `retry` 是「继续生成」：上一轮跑完有段失败，用户点了继续。带三样东西——
    上一轮的持久化 state（分段计划、已通过的产物都在里面）、要点名重跑的段、
    上一轮已通过审查的段 id。三者一起交给管线，已通过的段一个字都不重跑。
    """
    from .. import models_config
    from ..doc.lookup import AskUser, StopRun, snapshot
    from ..doc.trace import take as take_trace

    # Every line written from here on carries this run id, including the ones
    # produced deep inside the pipeline by code that never sees `run_id`.
    logs.bind_run(run_id)
    logger.info("任务开始 %s（%s）", run_id, "续跑" if payload is None else "首次执行")

    def _archive_steps() -> list[dict]:
        """把 live 桶里还没归档的过程倒出来。

        管线正常走完时，检索过程会跟着「导演查了什么」那条账本消息落库；但三条
        非正常出口——停下来问、用户选了不做、以及崩了——不会经过那条消息。不
        在这里接一把，用户跑的时候看见的那串过程框就会跟着 run 的内存一起消失，
        回看时一点痕迹都没有。桶里已经空的时候返回空列表，不会重复归档。
        """
        try:
            return take_trace(run_id, "director")
        except Exception:  # noqa: BLE001 - 归档失败不能把失败原因本身搞丢
            return []

    record = db.get_run(run_id) or {}
    pending = record.get("pending") or {}
    # 只有「问过、也答过」才算续跑。只剩问题没有回答的（用户还没点），不该
    # 被当成续跑——那会带着空答案进管线。
    resume = pending if (pending.get("intake") and pending.get("answer")) else None

    if payload is None:
        options = ((record.get("brief") or {}).get("options") or {})
        try:
            payload = CreateRun(**options)
        except Exception as exc:  # noqa: BLE001 - 旧库没有这份存档
            db.update_run(
                run_id, status="failed",
                error=f"续跑失败：找不到这次任务的原始参数（{exc}）",
            )
            return
        # 重新占位，否则取消接口会把这个正在续跑的任务当成「没人跑的僵尸」。
        pf_exec.mark_active(run_id)

    try:
        # 回答已经吃进去了，清掉，避免重复应用；新的断点会重新写一份。
        db.update_run(run_id, status="running", pending={})

        async def on_event(event: dict[str, Any]) -> None:
            db.add_message(
                run_id,
                role="agent" if event["kind"] != "warning" else "system",
                agent=event["agent"],
                kind=event["kind"],
                content=event["content"],
                payload=event.get("payload") or {},
                # 清单这类「一直代表当前状态」的消息带 key，重复写入就是更新它自己。
                msg_key=str(event.get("msg_key") or ""),
            )

        result = await _run_pipeline(
            run_id=run_id,
            chapter_text=payload.chapter_text,
            target_seconds=payload.target_seconds,
            model_settings=models_config.load(),
            max_rework=payload.max_rework,
            review_mode=payload.review_mode,

            generate_images=payload.generate_images,
            generate_videos=payload.generate_videos,
            on_event=on_event,
            # Carried into the six-agent pipeline so the consistency stage can
            # bind a sample to a character. The legacy pipeline drops it.
            voice_uploads=list(payload.voice_uploads or []),
            resume=resume,
            **(
                {
                    "resume_state": retry.get("state") or {},
                    "retry_segment_ids": list(retry.get("segment_ids") or []),
                    "prev_approved": list(retry.get("prev_approved") or []),
                }
                if retry
                else {}
            ),
        )

        state = _jsonable(result.get("state") or {})
        pipeline_status = str(result.get("status") or "")
        # A cancelled run is neither success nor failure: it stopped because the
        # user asked it to, and reporting it as "failed" would look like a bug.
        if pipeline_status == "cancelled":
            status = "cancelled"
            error = ""
        elif result.get("status") == "cancelled":
            # The pipeline may finish its unwinding after the user cancelled;
            # a deliberate stop must not be reported as a failure.
            status = "cancelled"
            error = ""
        elif pipeline_status == "done":
            # Trust the pipeline. It knows things this layer does not: in prompt
            # mode there is deliberately no film, so deriving "done" from
            # `has_video` here made every such run report 未产出成片 even when the
            # critic approved.
            status = "done"
            error = ""
        elif result.get("has_video"):
            status = "done"
            error = ""
        else:
            status = "failed"
            error = str(result.get("reason") or "未产出成片")[:400]
        db.update_run(run_id, status=status, error=error, state=state)
        # 结束时落一行。失败有异常堆栈可查，成功只有这一行——但它回答了
        # 「这段到底跑完没有、用了多久」，而这条信息在库里只有时间戳。
        logger.info("任务结束 %s：%s %s", run_id, status, error or "（无错误）")
        logs.event("run.end", status=status, run_id=run_id, error=error)

        segments = result.get("segments") or []
        # 降级完成必须说出来。「9/9 段通过审查」和「9/9 通过、其中 1 段没做一致性
        # 校验」是两种交付质量，同一句话会让用户以为拿到了完整的东西。
        degraded = result.get("degraded_segments") or []
        degraded_note = (
            f"（其中 {len(degraded)} 段跳过了校验环节）" if degraded else ""
        )
        # 没通过的段要逐段说清「卡在哪一步、为什么」。「完成：1/2 段通过审查」
        # 只给了个数，用户得自己翻对话找原因；详情就写在同一条消息里，翻都不用翻。
        titles = _plan_segment_titles(run_id)
        details = _failure_details(segments, titles)
        content = (
            f"{'完成' if status == 'done' else '未完成'}："
            f"{result.get('approved_segments',0)}/{result.get('total_segments',0)} "
            f"段通过审查{degraded_note}"
            + ("；已合成成片" if result.get("has_video") else "")
        )
        if details:
            for item in details:
                name = f"「{item['title']}」" if item.get("title") else f"段 {item['segment_id']}"
                line = f"{name}未通过：卡在「{item['stage']}」"
                if item.get("reason"):
                    line += f"——{item['reason']}"
                if item.get("message") and item["message"] != item.get("reason"):
                    line += f"（{item['message'][:90]}）"
                content += f"\n· {line}"
        db.add_message(
            run_id,
            role="assistant",
            agent="director",
            kind="result",
            content=content,
            payload={
                "segments": segments,
                "has_video": result.get("has_video"),
                "degraded_segments": degraded,
                "failed_detail": details,
            },
        )
    except AskUser as ask:
        # 停下来等用户选一条路。任务没有失败，只是挂在「待你回答」上。
        #
        # `pending` 里存三样东西：问了什么（渲染那张卡）、当时的 intake（续跑
        # 时接着用）、以及回答位置（null）。run 已经退出，内存不在了，只能靠它。
        question = ask.question
        # 过程框不能跟着 run 一起消失：问卡是用户决定「换哪条路」的依据，而
        # 依据是它已经查过什么、怎么失败的。账本消息正常路径已经带走了一份，
        # 这里只归档剩下的（比如续跑时那次联网的抓取过程）。
        leftover = _archive_steps()
        if leftover:
            db.add_message(
                run_id,
                role="agent",
                agent="director",
                kind="text",
                content="停下来之前，刚才排查的过程：",
                payload={"steps": leftover},
            )
        db.update_run(
            run_id,
            status="waiting",
            error="",
            pending={
                "phase": str(question.get("phase") or "material"),
                "question": question,
                "intake": snapshot(ask.intake),
                "answer": None,
            },
        )
        db.add_message(
            run_id,
            role="agent",
            agent="director",
            kind="ask",
            content=str(question.get("title") or "需要你确认材料来源"),
            payload=question,
            # 每次提问一张卡：同一次运行可以问很多次，覆盖掉上一张会让历史变成
            # 「只记得最后一次问过什么」。
            msg_key=f"director:ask:{question.get('id')}",
        )
    except StopRun:
        db.update_run(run_id, status="cancelled", error="", pending={})
        stop_steps = _archive_steps()
        db.add_message(
            run_id, role="system", agent="director", kind="text",
            content="你选择了先不做，这次任务已结束。",
            payload={"steps": stop_steps} if stop_steps else None,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced through run status
        # 这一行是整个日志系统里最值钱的一行：`str(exc)[:400]` 只留下症状，堆栈
        # 才能指出崩在哪一行、由谁调进来。界面上那句话给用户看，这里给排错的人看。
        logger.exception("任务失败 %s：%s", run_id, type(exc).__name__)
        logs.event(
            "run.failed", level="error",
            error=f"{type(exc).__name__}: {exc}", run_id=run_id,
        )
        db.update_run(run_id, status="failed", error=str(exc)[:400])
        # 崩溃时最后那串过程框是排错的唯一线索（坏在哪一步、之前查到了什么），
        # 不挂上就等于把现场擦了。
        crash_steps = _archive_steps()
        db.add_message(
            run_id, role="system", agent="director", kind="error",
            content=str(exc)[:300],
            payload={"steps": crash_steps} if crash_steps else None,
        )
    finally:
        # The heartbeat outlives the pipeline only on a crash; a row that still
        # says "正在调用模型" after the run is over is a lie the poll would repeat
        # forever, so it is dropped here whatever the outcome.
        try:
            from ..doc.trace import clear_working

            clear_working(run_id)
        except Exception:  # noqa: BLE001 - cleanup must not mask the real outcome
            pass
        from ..doc import cancellation

        cancellation.clear(run_id)
        pf_exec.clear_active(run_id)


class AnswerRun(BaseModel):
    """用户对一张问卡的回答。

    `question_id` 不是必填，但填了就能挡住「两个窗口都点了提交」——后回答的那次
    会收到 409，而不是把已经跑了半分钟的任务再重放一遍。
    """

    question_id: str = Field(default="", description="那张问卡的 id")
    choice: str = Field(min_length=1, description="选项 id：provide / web / feishu / stop")
    text: str = Field(default="", description="choice=provide 时的正文")


@router.post("/api/doc/runs/{run_id}/answer")
async def answer_run(run_id: str, body: AnswerRun, background: BackgroundTasks) -> dict:
    """回答「停在等你选」的那张卡，然后从断点续跑。

    这里只做三件事：校验回答属于这次提问、把回答存进 run、重新入队。真正应用回答
    的逻辑在管线里（`lookup.resolve`）——放在这里就会与首次执行的路径分叉，而
    分叉的那一半迟早没人维护。
    """
    record = db.get_run(run_id)
    if record is None:
        raise HTTPException(404, "运行不存在")
    if record["status"] != "waiting":
        raise HTTPException(409, f"这次任务不在等待回答（当前状态：{record['status']}）")

    pending = record.get("pending") or {}
    question = pending.get("question") or {}
    known = {str(item.get("id")) for item in (question.get("options") or [])}
    if body.choice not in known:
        raise HTTPException(400, f"选项「{body.choice}」不在这次给出的选项里")
    if body.question_id and question.get("id") and body.question_id != question.get("id"):
        raise HTTPException(409, "这张问题已经过期（可能是另一个窗口先回答了）")
    if body.choice == "provide" and not body.text.strip():
        raise HTTPException(400, "你选择了「我给你正文」，但正文是空的")

    answer = {"choice": body.choice, "text": body.text[:20000]}
    pending["answer"] = answer

    # 那张卡自己也记下答案。不更新的话，历史里会留着一张仍然在问你问题的卡，
    # 而任务其实早就跑完了。
    db.add_message(
        run_id,
        role="agent",
        agent="director",
        kind="ask",
        content=str(question.get("title") or "需要你确认材料来源"),
        payload={**question, "answer": answer},
        msg_key=f"director:ask:{question.get('id')}",
    )

    from ..doc import lookup

    if body.choice == lookup.STOP:
        # 「先不做了」不需要跑任何东西。让它照常入队的话，管线会在抛 StopRun 之前
        # 先把能力报告、素材提示重发一遍——用户选了结束，却看到两条刚出现的通知。
        db.update_run(run_id, status="cancelled", pending={})
        db.add_message(
            run_id, role="system", agent="director", kind="text",
            content="你选择了先不做，这次任务已结束。",
        )
        return {"status": "cancelled", "choice": body.choice}

    db.update_run(run_id, status="queued", pending=pending)
    background.add_task(_execute, run_id)
    return {"status": "queued", "choice": body.choice}


@router.post("/api/doc/runs/{run_id}/retry")
async def retry_run(run_id: str, background: BackgroundTasks) -> dict:
    """继续生成：只重跑没完成的段，已通过审查的段原样保留。

    依据全部来自上一轮留下的两份存档：持久化的 run state（分段计划、需求、
    已生成产物都在里面）和清单卡上每段的状态。清单上不是「完成」的段就是要
    重跑的段——包括阶段抛异常停下的，也包括审查没通过的。

    超时、限流这类失败重跑大概率就过；审查没通过的段重跑会带上满额返工预算。
    """
    record = db.get_run(run_id)
    if record is None:
        raise HTTPException(404, "运行不存在")
    if record["status"] not in ("done", "failed"):
        raise HTTPException(
            409, f"这次任务还没跑完，不能继续生成（当前状态：{record['status']}）"
        )
    state = record.get("state") or {}
    if not isinstance(state, dict) or not state.get("segments"):
        raise HTTPException(409, "找不到上一轮的分段存档，无法断点重跑")

    plan = db.get_message_by_key(run_id, "director:plan")
    produce = {}
    for step in ((plan or {}).get("payload") or {}).get("steps") or []:
        if step.get("id") == "produce":
            produce = step
            break
    segs = produce.get("segments") or []
    retry_ids = [str(s["id"]) for s in segs if s.get("status") != "done" and s.get("id")]
    prev_approved = [str(s["id"]) for s in segs if s.get("status") == "done" and s.get("id")]
    if not retry_ids:
        # 清单上没有失败段：可能上一轮其实全过了。如实拒绝——重跑全部段
        # 不能管它叫「继续生成」。
        raise HTTPException(409, "清单上没有未完成的段，没有需要继续生成的东西")

    db.add_message(
        run_id,
        role="system",
        agent="director",
        kind="text",
        content=(
            f"继续生成：重跑 {'、'.join(retry_ids)}，已通过的 {len(prev_approved)} 段不重做。"
        ),
    )
    db.update_run(run_id, status="running", pending={})
    pf_exec.mark_active(run_id)
    background.add_task(
        _execute,
        run_id,
        None,
        {
            "state": state,
            "segment_ids": retry_ids,
            "prev_approved": prev_approved,
        },
    )
    return {"status": "queued", "retry_segments": retry_ids}


# 终态的界面名字，取消被拒时用来说人话，而不是把 slug 直接抛给用户。
_TERMINAL_LABEL = {
    "done": "已完成",
    "partial": "部分完成",
    "failed": "失败",
    "cancelled": "已取消",
}


@router.post("/api/doc/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict:
    """Stop a running pipeline immediately.

    The wait inside the video poll loop is sliced, so it observes the flag every
    0.25 s and aborts; the segment gather cancels sibling tasks the same way. The
    status is written to the database here rather than waiting for the pipeline to
    unwind, so the UI sees the change on its next poll instead of looking stuck.

    `waiting` is cancellable too, but it is not an *interruption*: the pipeline has
    already returned, so there is nothing to interrupt. Cancelling it means giving
    up the parked task — the question it is holding on is dropped and the record
    closes as `cancelled`. Without this branch the button was visible in the UI
    (waiting is a live status there) but the endpoint answered "无需取消", which
    left the user stuck on a task they explicitly no longer wanted.
    """
    record = db.get_run(run_id)
    if record is None:
        raise HTTPException(404, "运行不存在")

    status = str(record["status"])
    if status == "waiting":
        db.update_run(run_id, status="cancelled", error="", pending={})
        db.add_message(
            run_id, role="user", agent="human", kind="cancel",
            content="用户取消了任务（当时正停在等你回答）", payload={},
        )
        return {
            "ok": True,
            "run_id": run_id,
            "note": "已结束这次任务；停在等待回答的那一步不会再继续",
        }

    if status not in {"queued", "running"}:
        return {
            "ok": False,
            "reason": f"这次任务已经结束（{_TERMINAL_LABEL.get(status, status)}），不用取消",
        }

    # A non-terminal run with no live owner is already dead, and "cancelling" it would
    # only rewrite a status. Reporting it as cancelled hides the real outcome: the
    # process died mid-run. Close it out instead, so the record reads as what it is.
    if not pf_exec.is_running(run_id):
        db.reconcile_interrupted_runs(only=[run_id])
        return {
            "ok": True,
            "run_id": run_id,
            "note": "该任务已中断（执行进程已不存在），已标记为失败",
        }

    from ..doc import cancellation

    cancellation.request(run_id)
    # Reflect it in the database immediately. The pipeline unwinds over the next
    # fraction of a second, but the UI polls once per second — leaving the status
    # as `running` makes the button look broken.
    db.update_run(run_id, status="cancelled")
    db.add_message(
        run_id, role="user", agent="human", kind="cancel",
        content="用户取消了任务", payload={},
    )
    return {
        "ok": True,
        "run_id": run_id,
        "note": "已停止；正在生成中的视频会在四分之一秒内中断",
    }


@router.get("/api/doc/asset/{path:path}")
async def get_asset(path: str):
    """Serve a generated artifact (portrait, clip, film).

    Path traversal is blocked by resolving the final location and confirming it
    stays under the data directory: the path comes from a URL, and a `../` would
    otherwise expose any file on the machine.
    """
    from pathlib import Path as _Path

    from fastapi.responses import FileResponse as _File

    from ..config import DATA_DIR

    root = _Path(DATA_DIR).resolve()
    target = (root / path).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise HTTPException(403, "非法的路径") from None

    if not target.is_file():
        raise HTTPException(404, "文件不存在")

    suffix = target.suffix.lower()
    media = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".mp4": "video/mp4", ".mp3": "audio/mpeg",
        ".wav": "audio/wav", ".webm": "video/webm",
    }.get(suffix, "application/octet-stream")
    return _File(target, media_type=media)


@router.get("/api/doc/runs")
async def list_runs(limit: int = 20, status: str = "") -> dict:
    """History list for the panel.

    Returns a lightweight summary: the panel needs a title, a status and a time,
    not the whole state. Loading state for twenty runs would be megabytes for text
    nobody reads until they click one.

    `status` filters by run status; the UI defaults to `done` because a first screen
    of failed debug attempts is noise rather than history.
    """
    return db.list_runs_summary(limit=max(1, min(100, limit)), status=status or None)


@router.delete("/api/doc/runs/{run_id}")
async def delete_run(run_id: str) -> dict:
    """Delete one history entry, its messages and its generated files.

    A hard delete, as confirmed: the user asked to remove a conversation, and
    keeping a soft-deleted copy would retain their text while hiding it — the
    opposite of what deletion is for.

    Refuses only while the run is *actually executing*. The status column is a claim,
    and a claim outlives the process that made it: a crash, a reload or a machine
    restart left rows marked `running` that nothing was working on, and those became
    conversations that no delete could ever remove. The guard therefore asks the
    executor whether this run is live, and treats a non-terminal status with no live
    owner as a dead run — which is deletable.

    The live check must stay exact: deleting rows a live pipeline is still writing to
    would corrupt the run and orphan its files.
    """
    record = db.get_run(run_id)
    if record is None:
        raise HTTPException(404, "运行不存在")

    # Ask the executor, not the column. `status` says what a process *claimed*;
    # `is_running` says whether one is actually holding this run right now. A live
    # run is refused whatever its status says, because it can still be writing
    # rows: cancel writes `cancelled` immediately and then unwinds for a moment.
    if pf_exec.is_running(run_id):
        raise HTTPException(409, "任务运行中，不能删除；请先取消")

    if record["status"] in {"queued", "running"}:
        # Non-terminal but nothing owns it: close it out, then fall through to the
        # delete. Leaving the row as `running` is what made it undeletable forever.
        db.reconcile_interrupted_runs(only=[run_id])
        record = db.get_run(run_id)
        if record is None:
            return {"ok": True, "deleted": {"studio_runs": 1, "note": "已清理并删除"}}
        if record["status"] in {"queued", "running"}:
            raise HTTPException(409, "任务运行中，不能删除；请先取消")

    return db.delete_run(run_id)


def _local_media_index(run_id: str) -> dict[str, str]:
    """file name → `/api/doc/asset/...` for everything this run saved.

    Keys are bare file names because that is the only part of a renderer's link we
    can trust to still be there: the gateway names its output `gw_img_…png` and we
    save it under that name, while the host and query of its link are exactly what
    stops working (401 once the key is not sent).
    """
    from ..config import ASSET_DIR

    index: dict[str, str] = {}
    root = ASSET_DIR / run_id
    if not root.is_dir():
        return index
    from ..doc.media_url import served_url as to_url

    for path in root.rglob("*"):
        if path.is_file():
            index.setdefault(path.name, to_url(path))
    return index


def _localise_media(value: Any, index: dict[str, str]) -> Any:
    """Rewrite renderer links in a stored payload to the copy we serve.

    Runs recorded before the renderers started saving their own copy carry the
    gateway's URL, which answers 401 to a browser. The file is on disk under its own
    name, so the response can point at that instead — the stored payload is left
    untouched, and this only rewrites what we can actually serve.
    """
    from urllib.parse import unquote, urlparse

    if isinstance(value, dict):
        return {k: _localise_media(v, index) for k, v in value.items()}
    if isinstance(value, list):
        return [_localise_media(v, index) for v in value]
    if isinstance(value, str) and value.startswith("http"):
        name = unquote(urlparse(value).path.rsplit("/", 1)[-1])
        # The gateway prefixes its download path with a job hash
        # (`274d8c1622-gw_img_…png`) but saves the file as `gw_img_…png`, so the
        # bare name does not match: try the name after the prefix, then the suffix.
        local = index.get(name) or index.get(name.split("-", 1)[-1])
        if not local:
            local = next((url for key, url in index.items() if name.endswith(key)), None)
        if local:
            return local
    return value


@router.get("/api/doc/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    record = db.get_run(run_id)
    if record is None:
        raise HTTPException(404, "运行不存在")
    state = record.get("state") or {}
    messages = db.list_messages(run_id)
    media = _local_media_index(run_id)
    workers = _working(run_id, record["status"])
    return {
        "run_id": run_id,
        "status": record["status"],
        "error": record.get("error", ""),
        "has_video": bool(state.get("has_video")),
        "video_url": f"/api/doc/runs/{run_id}/video" if state.get("has_video") else "",
        "messages": [
            {
                "role": m["role"], "agent": m["agent"], "kind": m["kind"],
                "content": m["content"],
                "payload": _localise_media(m.get("payload") or {}, media),
            }
            for m in messages
        ],
        # Only the fields the UI renders: the full state holds portraits, shots,
        # voice clips and constraints, which would make every poll megabytes.
        "summary": {
            "segments": len(state.get("segments") or []),
            "shots": sum(len((b or {}).get("shots") or []) for b in (state.get("storyboards") or {}).values()),
            "portraits": len(state.get("portraits") or []),
            "shots_generated": len([a for a in (state.get("shot_assets") or []) if a.get("ok")]),
            "voiced": len([v for v in (state.get("voice_clips") or []) if v.get("ok")]),
            "critic": state.get("critic_summary", ""),
        },
        "material_check": state.get("material_check") or {},
        "final_cut": state.get("final_cut") or {},
        # Which shot is being rendered right now, if any. The UI turns this into
        # "正在生成视频 3/12" on the shot card: without it a 70-second render is
        # indistinguishable from a shot that failed, because both show "未生成".
        "render_progress": state.get("render_progress") or {},
        # What this run cannot produce (no video model, no audio capability, …).
        # Reported every poll so a UI that reloaded mid-run still shows it.
        "capabilities": state.get("capability_report") or {},
        "audio_plan": state.get("audio_plan") or {},
        # Live traces of agents still working. This is what makes the reasoning
        # appear *while it is being written* rather than after the stage ends:
        # the message list only gains an entry when a stage completes.
        "live": _live_traces(run_id, record["status"]),
        # The switch itself, so the UI can name the right reason for a quiet
        # stage instead of guessing from the absence of live steps.
        "streaming": _streaming_on(),
        # Who is working right now and for how long. Separate from `live` on
        # purpose: with streaming off there are no new steps to show, and this is
        # the only thing that tells a waiting user the run is alive.
        "working": workers[0] if workers else {},
        # Every agent busy right now. Segments run concurrently, so the big screen
        # needs the whole list; `working` stays for the conversation UI, which
        # only ever shows one row.
        "workers": workers,
    }


def _working(run_id: str, status: str) -> list[dict[str, Any]]:
    """The in-flight agents and their elapsed seconds; empty when nothing runs.

    Guarded by status so a finished run never reports a worker: the heartbeat is
    per-process, and a run that ended (cancelled, failed, killed) must not leave
    the UI showing a timer that will never stop.
    """
    if status not in {"queued", "running"}:
        return []
    try:
        from ..doc.trace import peek_workers

        return peek_workers(run_id)
    except Exception:  # noqa: BLE001 - diagnostics must never break the status call
        return []


def _live_traces(run_id: str, status: str) -> dict[str, list[dict[str, Any]]]:
    """Return in-flight traces, trimmed to what the UI renders.

    Only for a running run: once finished, the traces have been drained into the
    messages, and re-reading them would duplicate the content.
    """
    if status not in {"queued", "running"}:
        return {}
    try:
        from ..doc.trace import peek_all

        return peek_all(run_id)
    except Exception:  # noqa: BLE001 - diagnostics must never break the status call
        return {}


def _streaming_on() -> bool:
    """Whether model output is being streamed, per the provider config.

    The UI needs this to explain a silent stage. It used to infer the switch from
    "no live steps exist", which is wrong in a specific and common case: streaming
    is on, but the model returns no `reasoning_content` (any non-reasoning model),
    so no deltas are ever written to the trace. The result was a stage that said
    "streaming is off" while streaming was on.

    Read per call rather than cached: it is a small JSON lookup, and toggling the
    setting in 设置 should change the wording on the next poll instead of the next
    restart.
    """
    try:
        from ..agents.base import stream_enabled

        return bool(stream_enabled())
    except Exception:  # noqa: BLE001 - a broken config must not break status
        return False


@router.get("/api/doc/runs/{run_id}/video")
async def get_video(run_id: str):
    record = db.get_run(run_id)
    if record is None:
        raise HTTPException(404, "运行不存在")
    path = (record.get("state") or {}).get("video_path") or ""
    from pathlib import Path

    if not path or not Path(path).is_file():
        raise HTTPException(404, "该运行还没有成片")
    return FileResponse(path, media_type="video/mp4", filename=f"{run_id}.mp4")


def _jsonable(state: dict[str, Any]) -> dict[str, Any]:
    import json

    out: dict[str, Any] = {}
    for key, value in state.items():
        if key.startswith("__"):
            continue
        try:
            json.dumps(value)
            out[key] = value
        except (TypeError, ValueError):
            out[key] = str(value)[:400]
    return out
