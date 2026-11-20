"""四趟推进：资产提示词 → 出图 → 全片视频提示词 → 出视频。

这不是实现细节，而是用户能观察到的行为：

  · 提示词阶段不读「出图/出视频」开关 —— 不勾也照样产出；
  · 全片提示词写完、审过，才开始出图；
  · 图片全部出完后，先收齐并审查全片视频提示词，才开始出视频；
  · 出图一条一条提交 —— 两张同时在飞正是限流的来源。

测试用桩 Agent + 桩生成器，不碰网络，断言的是**调用顺序**。
"""
from __future__ import annotations

import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, Mock, patch

from app import models_config  # noqa: F401  (imported for side effects of config load)
from app.agents.base import AgentContext, _coerce_object
from app.doc import asset_queue, intake, media_mode, media_steps, pipeline_six
from app.doc import agents_six_impl  # noqa: F401


class OrderRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.inflight = 0
        self.peak = 0

    async def render_image(self, prompt: str, out_dir, size: str):
        from types import SimpleNamespace

        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        self.events.append(("image", prompt[:12]))
        import asyncio

        await asyncio.sleep(0.01)
        self.inflight -= 1
        out_dir.mkdir(parents=True, exist_ok=True)
        local = out_dir / f"fake-{len(self.events)}.png"
        local.write_bytes(b"fake-image")
        return SimpleNamespace(
            url=f"http://img/{len(self.events)}.png",
            local_path=str(local),
        )


class ThreePassTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.recorder = OrderRecorder()
        self.events: list[dict] = []

        self.stack.enter_context(patch.object(intake, "_config", return_value=object()))
        self.stack.enter_context(
            patch.object(
                intake.llm_tools,
                "chat_json",
                new_callable=AsyncMock,
                return_value={
                    "intent": "material", "topic": "", "desired_seconds": 0, "reason": "test",
                },
            )
        )
        self.stack.enter_context(patch.object(media_mode, "prompt_only", return_value=False))
        # The queue asks the media layer for a generator; this is the seam that
        # lets the test see submissions without a provider.
        self.stack.enter_context(
            patch.object(media_steps, "_generator", side_effect=lambda ctx, cap="": self.recorder)
        )
        self.stack.enter_context(
            patch.object(pipeline_six, "_assemble", new_callable=AsyncMock, return_value="")
        )
        for name, value in [
            ("_video_capabilities", {}),
            ("_image_capabilities", {}),
            ("_image_model_name", "img-model"),
            ("_video_model_name", "vid-model"),
        ]:
            self.stack.enter_context(patch.object(pipeline_six, name, return_value=value))

    async def collect(self, event: dict) -> None:
        self.events.append(event)
        artifacts = (event.get("payload") or {}).get("artifacts") or {}
        if (
            event.get("agent") == "video_gen"
            and (event.get("payload") or {}).get("prompts_ready")
            and artifacts.get("prompts")
        ):
            self.recorder.events.append(
                ("video_prompt_shown", str((event.get("payload") or {}).get("segment") or ""))
            )
        if event.get("agent") == "consistency" and artifacts.get("images_pending"):
            self.recorder.events.append(
                ("image_prompt_shown", str((event.get("payload") or {}).get("segment") or ""))
            )

    def _resolve(self):
        recorder = self.recorder

        def resolve(name):
            async def run(ctx: AgentContext, definition, options):
                index = int(options.get("segment_index") or 0)
                seg_id = f"seg{index + 1}"
                if name == "director":
                    return {
                        "segments": [
                            {"id": "seg1", "title": "一", "order": 0},
                            {"id": "seg2", "title": "二", "order": 1},
                        ],
                        "review_mode": "NORMAL",
                    }
                if name == "consistency":
                    recorder.events.append(("prompts", seg_id))
                    manifest = {
                        "segment_id": seg_id,
                        "character_assets": [{
                            "character_id": f"CHAR-{index}", "name": "孙中山",
                            "four_view_prompt": f"四视图 {seg_id}",
                            "negative_prompt": "",
                        }],
                        "scene_assets": [{
                            "scene_id": f"SCENE-{index}", "name": "石库门客厅",
                            "prompt": "夜晚的旧式客厅",
                        }],
                        "prop_assets": [],
                        "image_generation_tasks": [
                            {"asset_id": f"CHAR-{index}", "asset_type": "character_four_view",
                             "image_prompt": f"四视图 {seg_id}", "aspect_ratio": "1:1",
                             "status": "pending"},
                            {"asset_id": f"SCENE-{index}", "asset_type": "scene",
                             "image_prompt": "夜晚的旧式客厅", "aspect_ratio": "16:9",
                             "status": "pending"},
                        ],
                    }
                    index_map = ctx.state.setdefault("asset_index", {})
                    index_map[seg_id] = manifest
                    asset_queue.link(ctx, manifest, segment_index=index)
                    return {"asset_index": index_map}
                if name == "video_gen":
                    prepared = {
                        "video_generation_tasks": [{
                            "shot_id": f"SB-{index + 1:03d}",
                            "duration": 5,
                            "final_video_prompt": f"视频提示词 {seg_id}",
                            "status": "prompt_ready",
                        }],
                        "shot_assets": [],
                        "video_summary": "视频提示词已展示",
                    }
                    callback = options.get("on_prompts_ready")
                    if options.get("prepare_only") and callable(callback):
                        await callback(prepared)
                    if options.get("prepare_only"):
                        recorder.events.append(("video_prompt_prepared", seg_id))
                    else:
                        recorder.events.append(("video", seg_id))
                    return prepared
                if name == "quality_review":
                    return {"critic_verdict": {"approved": True}}
                return {}

            return None, run

        return resolve

    async def test_prompts_then_images_then_video(self):
        self.stack.enter_context(
            patch.object(pipeline_six.registry, "catalog", return_value=[{"id": "script_adaptor"}])
        )
        self.stack.enter_context(
            patch.object(pipeline_six.registry, "resolve", side_effect=self._resolve())
        )

        await pipeline_six.run_pipeline_six(
            run_id="test-three-pass",
            chapter_text="1924 年 1 月，国民党一大在广州召开，第一次国共合作正式形成。",
            target_seconds=30,
            model_settings=None,
            max_rework=0,
            generate_images=True,
            generate_videos=True,
            on_event=self.collect,
        )

        order = [kind for kind, _ in self.recorder.events]
        prompts = [i for i, k in enumerate(order) if k == "prompts"]
        images = [i for i, k in enumerate(order) if k == "image"]
        videos = [i for i, k in enumerate(order) if k == "video"]

        self.assertEqual(len(prompts), 2, "两段都要写提示词")
        self.assertEqual(len(images), 2, "两个人物同名、两个场景同名：全片只画两张")
        self.assertEqual(len(videos), 2, "两段都要出视频")
        self.assertLess(max(prompts), min(images), "提示词没写完就开始出图")
        image_prompts_shown = [i for i, k in enumerate(order) if k == "image_prompt_shown"]
        self.assertEqual(len(image_prompts_shown), 2, "每段出图提示词都必须先展示")
        self.assertLess(max(image_prompts_shown), min(images), "出图提示词还不可见就提交了图片")
        self.assertLess(max(images), min(videos), "图片没出完就开始出视频")
        shown = [i for i, k in enumerate(order) if k == "video_prompt_shown"]
        self.assertEqual(len(shown), 2, "每段视频提示词都必须先展示")
        self.assertLess(
            max(shown), min(videos),
            "必须收齐并展示全片所有视频提示词后，才能提交第一个视频",
        )
        for segment in ("seg1", "seg2"):
            shown_at = self.recorder.events.index(("video_prompt_shown", segment))
            rendered_at = self.recorder.events.index(("video", segment))
            self.assertLess(shown_at, rendered_at, f"{segment} 提示词还没展示就调用了视频生成")
        self.assertEqual(self.recorder.peak, 1, "出图必须一条一条来")
        self.assertEqual(
            [seg for kind, seg in self.recorder.events if kind == "video"], ["seg1", "seg2"],
            "视频按段序串行",
        )

    async def test_switch_off_still_writes_prompts(self):
        self.stack.enter_context(
            patch.object(pipeline_six.registry, "catalog", return_value=[{"id": "script_adaptor"}])
        )
        self.stack.enter_context(
            patch.object(pipeline_six.registry, "resolve", side_effect=self._resolve())
        )

        await pipeline_six.run_pipeline_six(
            run_id="test-prompts-only",
            chapter_text="1924 年 1 月，国民党一大在广州召开。",
            target_seconds=30,
            model_settings=None,
            max_rework=0,
            generate_images=False,
            generate_videos=False,
            on_event=self.collect,
        )

        kinds = [k for k, _ in self.recorder.events]
        self.assertEqual(kinds.count("prompts"), 2, "不勾任何开关，提示词照样要写")
        self.assertEqual(
            kinds.count("video_prompt_prepared"), 2,
            "不勾出视频时仍要产出并审查全片镜头提示词",
        )
        self.assertEqual(kinds.count("video"), 0, "开关关闭时不得提交视频生成")
        self.assertNotIn("image", kinds, "不勾出图，不能提交任何一张")
        # `image` 是本测试里唯一的“真实提交”记录点；视频阶段的桩不会提交，
        # 所以这里断言的是“没有任何渲染提交”。
        self.assertEqual(kinds.count("image"), 0)

    async def test_asset_rejection_retries_consistency_not_content_adaptor(self):
        calls: list[str] = []
        asset_reviews = 0

        def resolve(name):
            async def run(ctx: AgentContext, definition, options):
                nonlocal asset_reviews
                if name == "director":
                    return {
                        "segments": [{"id": "seg1", "title": "一", "order": 0}],
                        "review_mode": "NORMAL",
                    }
                if name == "quality_review":
                    stage = str(options.get("review_stage") or "")
                    if stage == "ASSET_REVIEW":
                        asset_reviews += 1
                        if asset_reviews == 1:
                            return {
                                "critic_summary": "一致性提示词需重做",
                                "critic_verdict": {
                                    "approved": False,
                                    # Simulate the bad model output that used to
                                    # rewind the pipeline to 内容改编.
                                    "rework_target": "script_adaptor",
                                    "blocking_issues": [{
                                        "severity": "P1",
                                        "responsible_agent": "script_adaptor",
                                        "description": "人物设定不一致",
                                        "repair_instruction": "重新生成一致性设定",
                                    }],
                                },
                            }
                    return {"critic_verdict": {"approved": True}}
                calls.append(name)
                if name == "video_gen":
                    prepared = {
                        "video_generation_tasks": [],
                        "shot_assets": [],
                        "video_summary": "提示词完成",
                    }
                    callback = options.get("on_prompts_ready")
                    if callable(callback):
                        await callback(prepared)
                    return prepared
                return {}

            return None, run

        self.stack.enter_context(
            patch.object(pipeline_six.registry, "catalog", return_value=[{"id": "script_adaptor"}])
        )
        self.stack.enter_context(patch.object(pipeline_six.registry, "resolve", side_effect=resolve))

        await pipeline_six.run_pipeline_six(
            run_id="test-local-asset-rework",
            chapter_text="一致性审查失败后应当只重做一致性阶段。",
            target_seconds=10,
            model_settings=None,
            max_rework=1,
            generate_images=False,
            generate_videos=False,
            on_event=self.collect,
        )

        self.assertEqual(calls.count("script_adaptor"), 1)
        self.assertEqual(calls.count("storyboard"), 1)
        self.assertEqual(calls.count("consistency"), 2)
        reworks = [e for e in self.events if e.get("kind") == "rework"]
        self.assertTrue(reworks)
        self.assertEqual((reworks[0].get("payload") or {}).get("target"), "consistency")

    async def test_incomplete_prompt_set_blocks_all_image_and_video_generation(self):
        calls: list[tuple[str, str]] = []

        def resolve(name):
            async def run(ctx: AgentContext, definition, options):
                index = int(options.get("segment_index") or 0)
                sid = f"seg{index + 1}"
                if name == "director":
                    return {
                        "segments": [
                            {"id": "seg1", "title": "一", "order": 0},
                            {"id": "seg2", "title": "二", "order": 1},
                        ],
                        "review_mode": "NORMAL",
                    }
                if name == "quality_review":
                    return {"critic_verdict": {"approved": True}}
                if name == "consistency":
                    calls.append(("prompt", sid))
                    if sid == "seg2":
                        raise ValueError("图片提示词未完成")
                    manifest = {
                        "segment_id": sid,
                        "character_assets": [{
                            "character_id": "CHAR-001", "name": "人物",
                            "four_view_prompt": "完整人物提示词",
                        }],
                        "scene_assets": [],
                        "prop_assets": [],
                        "image_generation_tasks": [{
                            "asset_id": "CHAR-001", "asset_type": "character_four_view",
                            "image_prompt": "完整人物提示词", "aspect_ratio": "1:1",
                            "status": "pending",
                        }],
                    }
                    ctx.state.setdefault("asset_index", {})[sid] = manifest
                    asset_queue.link(ctx, manifest, segment_index=index)
                    return {"asset_index": ctx.state["asset_index"]}
                if name == "video_gen":
                    calls.append(("video", sid))
                return {}

            return None, run

        self.stack.enter_context(
            patch.object(pipeline_six.registry, "catalog", return_value=[{"id": "script_adaptor"}])
        )
        self.stack.enter_context(
            patch.object(pipeline_six.registry, "resolve", side_effect=resolve)
        )

        await pipeline_six.run_pipeline_six(
            run_id="test-incomplete-prompts-stop-media",
            chapter_text="任何一段提示词失败，都不能提前开始媒体生成。",
            target_seconds=20,
            model_settings=None,
            max_rework=0,
            generate_images=True,
            generate_videos=True,
            on_event=self.collect,
        )

        self.assertEqual([c for c in calls if c[0] == "prompt"], [
            ("prompt", "seg1"), ("prompt", "seg2"), ("prompt", "seg2"),
        ], "失败分段只原地重试一次，成功分段不能重跑")
        self.assertNotIn("image", [kind for kind, _ in self.recorder.events])
        self.assertFalse([c for c in calls if c[0] == "video"])
        warnings = [e for e in self.events if e.get("kind") == "warning"]
        self.assertTrue(any("提示词未全部完成" in e.get("content", "") for e in warnings))

    async def test_retryable_storyboard_failure_retries_only_storyboard(self):
        calls: list[str] = []
        storyboard_calls = 0

        def resolve(name):
            async def run(ctx: AgentContext, definition, options):
                nonlocal storyboard_calls
                if name == "director":
                    return {
                        "segments": [{"id": "seg1", "title": "一", "order": 0}],
                        "review_mode": "FAST",
                    }
                if name == "quality_review":
                    return {"critic_verdict": {"approved": True}}

                calls.append(name)
                if name == "storyboard":
                    storyboard_calls += 1
                    if storyboard_calls == 1:
                        raise ValueError("模型连续两次没有返回可用分镜")
                    ctx.state["storyboards"] = {
                        "seg1": {
                            "segment_id": "seg1", "total_seconds": 5,
                            "shots": [{
                                "shot_id": "SB-001", "duration": 5,
                                "video_prompt": "完整提示词", "visual_description": "会场",
                            }],
                        }
                    }
                    return {"shot_count": 1, "total_seconds": 5}
                if name == "consistency":
                    manifest = {
                        "segment_id": "seg1", "character_assets": [],
                        "scene_assets": [], "prop_assets": [],
                        "image_generation_tasks": [],
                    }
                    ctx.state.setdefault("asset_index", {})["seg1"] = manifest
                    return {"asset_index": ctx.state["asset_index"]}
                if name == "video_gen":
                    prepared = {
                        "video_generation_tasks": [], "shot_assets": [],
                        "video_summary": "提示词完成",
                    }
                    callback = options.get("on_prompts_ready")
                    if callable(callback):
                        await callback(prepared)
                    return prepared
                return {}

            return None, run

        self.stack.enter_context(
            patch.object(pipeline_six.registry, "catalog", return_value=[{"id": "script_adaptor"}])
        )
        self.stack.enter_context(
            patch.object(pipeline_six.registry, "resolve", side_effect=resolve)
        )

        result = await pipeline_six.run_pipeline_six(
            run_id="test-local-stage-retry",
            chapter_text="分镜格式失败只能重试分镜，不能重写剧本。",
            target_seconds=10,
            model_settings=None,
            max_rework=1,
            generate_images=False,
            generate_videos=False,
            on_event=self.collect,
        )

        self.assertEqual(calls.count("script_adaptor"), 1)
        self.assertEqual(calls.count("storyboard"), 2)
        self.assertEqual(calls.count("consistency"), 1)
        self.assertEqual(result["approved_segments"], 1)


class VideoPersistenceBarrierTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_local_video_file_is_rejected(self):
        from types import SimpleNamespace

        class MissingFileVideoGenerator:
            async def render_video(self, **kwargs):
                out_dir = kwargs["out_dir"]
                return SimpleNamespace(
                    url="http://video/result.mp4",
                    local_path=str(out_dir / "not-created.mp4"),
                )

        ctx = AgentContext(run_id="test-video-local-file", brief={})
        ctx.trace_step = lambda *a, **k: None
        with self.assertRaisesRegex(RuntimeError, "未成功保存到本地"):
            await agents_six_impl._render_shot(
                ctx,
                MissingFileVideoGenerator(),
                {"shot_id": "SB-001", "duration": 5, "link_type": "CUT"},
                "视频提示词",
                [],
                "",
                supports_first_frame=False,
            )

    async def test_video_failure_stops_the_remaining_queue(self):
        def task(number: int) -> dict:
            return {
                "shot_id": f"SB-{number:03d}", "status": "prompt_ready",
                "video_uri": None, "local_path": "", "duration": 5,
                "reference_images": [], "error": None,
            }

        tasks = [task(1), task(2)]
        prepared = {
            "segment_id": "seg1",
            "plans": [
                (tasks[0], {"shot_id": "SB-001", "scene_id": "S1"}, "p1", []),
                (tasks[1], {"shot_id": "SB-002", "scene_id": "S1"}, "p2", []),
            ],
            "model_name": "fake-video", "caps": {}, "audio": {},
        }
        ctx = AgentContext(run_id="test-video-stop", brief={})
        ctx.trace_step = lambda *a, **k: None
        fake_generator = Mock()

        with (
            patch.object(media_steps, "_generator", return_value=fake_generator),
            patch.object(
                agents_six_impl, "_render_shot",
                new_callable=AsyncMock,
                side_effect=RuntimeError("本地保存失败"),
            ) as render,
        ):
            result = await agents_six_impl._render_video_plan(ctx, prepared)

        self.assertEqual(render.await_count, 1, "当前视频未落盘，不得提交下一条")
        self.assertEqual(tasks[0]["status"], "failed")
        self.assertEqual(tasks[1]["status"], "prompt_ready")
        self.assertIn("0/2", result["video_summary"])


class ReviewDigestCompletenessTests(unittest.TestCase):
    def test_review_receives_last_script_section_and_last_storyboard_shot(self):
        ctx = AgentContext(run_id="test-review-complete-input", brief={})
        sections = [
            {
                "section_title": f"第{i}节",
                "narrative_purpose": f"目的{i}",
                "scene_description": f"场景内容 SECTION-{i}",
                "narration": [f"第{i}节旁白"],
            }
            for i in range(1, 7)
        ]
        shots = [
            {
                "shot_id": f"SB-{i:03d}",
                "duration": 5,
                "shot_size": "中景",
                "requirement_ids": [f"REQ-{i:03d}"],
                "visual_description": f"第{i}镜画面",
                "narration": "最后必现内容" if i == 9 else f"第{i}镜旁白",
                "dialogue": [],
            }
            for i in range(1, 10)
        ]
        board = {
            "segment_id": "seg1", "shots": shots, "total_seconds": 45,
            "duration_analysis": {"narration_duration": 40, "dialogue_duration": 0},
        }
        ctx.state.update({
            "segments": [{"id": "seg1"}],
            "adapted_script": {"seg1": {"mode": "MODE_B", "sections": sections}},
            "constraints": {"seg1": {"requirements": [
                {
                    "requirement_id": f"REQ-{i:03d}",
                    "requirement": f"必现内容{i}",
                    "acceptance": "旁白出现",
                }
                for i in range(1, 10)
            ]}},
        })

        script_digest = agents_six_impl._review_digest(
            ctx, "SCRIPT_REVIEW", board, []
        )
        storyboard_digest = agents_six_impl._review_digest(
            ctx, "STORYBOARD_REVIEW", board, []
        )

        self.assertIn("SECTION-6", script_digest, "审查不能只收到前四节剧本")
        self.assertIn("SB-009", storyboard_digest, "审查不能只收到前八镜")
        self.assertIn("最后必现内容", storyboard_digest)

    def test_final_note_ignores_a_failure_superseded_by_success(self):
        outcome = pipeline_six.SegmentOutcome(segment_id="seg1", approved=False, attempts=2)
        outcome.stages = [
            {"agent": "storyboard", "ok": False, "error": "旧格式错误"},
            {"agent": "storyboard", "ok": True, "summary": "9 个镜头"},
        ]
        outcome.issues = [{"description": "最终审查发现的真实问题"}]

        note = pipeline_six._fail_note(outcome)

        self.assertNotIn("旧格式错误", note)
        self.assertIn("最终审查发现的真实问题", note)


class ConsistencySinglePromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_people_scenes_and_props_are_requested_in_one_model_call(self):
        ctx = AgentContext(run_id="test-one-consistency-call", brief={})
        ctx.trace_step = lambda *args, **kwargs: None
        ctx.state.update({
            "segments": [{"id": "seg1", "title": "一"}],
            "storyboards": {
                "seg1": {
                    "segment_id": "seg1",
                    "shots": [{
                        "shot_id": "SB-001",
                        "visual_description": "孙中山在旧式旅馆阅读报纸",
                        "characters": ["孙中山"],
                    }],
                }
            },
            "adapted_script": {},
            "constraints": {},
            "generate_images": False,
        })
        ctx.agent_ask = AsyncMock(return_value={
            "style_guide": {
                "style_prompt": "photorealistic cinematic film still",
                "negative_prompt": "anime",
                "palette": "暗褐、靛青",
            },
            "character_assets": [{
                "name": "孙中山", "appearance": "长脸，短发，西装",
                "four_view_prompt": "same person, front side back and face close-up",
                "negative_prompt": "modern clothes", "is_main": True,
            }],
            "scene_assets": [{
                "name": "旧式旅馆", "time_of_day": "夜晚", "location": "海外",
                "layout": "木桌靠窗", "lighting": "油灯侧光",
                "prompt": "old hotel room at night, cinematic lighting",
            }],
            "prop_assets": [{
                "name": "旧报纸", "material": "泛黄纸", "shape": "竖排报纸",
                "era_note": "民国初年", "prompt": "an aged newspaper on a wooden desk",
            }],
            "continuity_rules": ["人物服装保持一致"],
            "missing_assets": [],
        })

        result = await agents_six_impl.consistency_agent(
            ctx, None, {"segment_index": 0}
        )

        self.assertEqual(ctx.agent_ask.await_count, 1)
        request = str(ctx.agent_ask.await_args.kwargs.get("user") or "")
        self.assertIn("同时返回人物、场景和道具", request)
        self.assertNotIn("第 1/2 步", request)
        self.assertNotIn("第 2/2 步", request)
        manifest = result["asset_manifest"]
        self.assertEqual(len(manifest["character_assets"]), 1)
        self.assertEqual(len(manifest["scene_assets"]), 1)
        self.assertEqual(len(manifest["prop_assets"]), 1)


class VideoPromptVisibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_video_agent_publishes_prompts_before_switch_and_render(self):
        order: list[str] = []
        ctx = AgentContext(run_id="test-video-prompt-order", brief={})
        ctx.trace_step = lambda *args, **kwargs: None
        ctx.state.update({
            "segments": [{"id": "seg1"}],
            "storyboards": {
                "seg1": {
                    "segment_id": "seg1",
                    "shots": [{
                        "shot_id": "SB-001",
                        "duration": 5,
                        "video_prompt": "中景，人物走入房间",
                        "scene_id": "SCENE-001",
                        "character_ids": [],
                        "link_type": "SCENE_TRANSITION",
                    }],
                }
            },
            "asset_index": {"seg1": {
                "character_assets": [], "scene_assets": [], "prop_assets": [],
            }},
            "constraints": {},
            "art_bible": {},
            "generate_videos": True,
            "video_model_capabilities": {},
        })

        class FakeVideoGenerator:
            video_model = "fake-video"

        async def shown(result):
            self.assertTrue(result["video_generation_tasks"][0]["final_video_prompt"])
            order.append("shown")

        async def render(*args, **kwargs):
            order.append("render")
            return ("http://video/1.mp4", "C:/tmp/1.mp4", "none", "")

        def choose_generator(*args, **kwargs):
            order.append("switch")
            return FakeVideoGenerator()

        with patch.object(agents_six_impl, "_check_prompts", new_callable=AsyncMock), \
             patch.object(
                 agents_six_impl, "_adapt_prompts", new_callable=AsyncMock,
                 return_value={"SB-001": "适配后的视频提示词"},
             ), \
             patch.object(agents_six_impl, "_video_model", return_value="fake-video"), \
             patch.object(agents_six_impl, "_render_shot", side_effect=render), \
             patch.object(media_steps, "_generator", side_effect=choose_generator):
            await agents_six_impl.video_gen_agent(
                ctx, None, {"segment_index": 0, "on_prompts_ready": shown}
            )

        self.assertEqual(order, ["shown", "switch", "render"])


class StoryboardShapeRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def context(self) -> AgentContext:
        ctx = AgentContext(run_id="test-storyboard-shape", brief={})
        ctx.trace_step = lambda *args, **kwargs: None
        ctx.state.update({
            "segments": [{"id": "seg1", "title": "南湖续会"}],
            "target_seconds": 20,
            "adapted_script": {
                "seg1": {
                    "title": "南湖续会",
                    "sections": [{
                        "section_id": "SEC-001",
                        "section_title": "游船续会",
                        "scene_description": "代表们在南湖游船中继续会议。",
                        "narration": ["会议转移到嘉兴南湖游船上继续举行。"],
                    }],
                }
            },
            "constraints": {"seg1": {"requirements": []}},
            "video_model_capabilities": {"max_seconds_per_shot": 10},
        })
        return ctx

    @staticmethod
    def shot(number: int) -> dict:
        return {
            "shot_id": f"SB-{number:03d}",
            "section_id": "SEC-001",
            "duration": 5,
            "scene_id": "SCENE-001",
            "character_ids": [],
            "characters": ["一大代表"],
            "requirement_ids": [],
            "visual_description": "游船舱内的会议现场",
            "narration": "会议在南湖游船上继续举行。",
            "video_prompt": "中景，平视。0–5秒：代表们围桌讨论。",
            "start_state": "代表们围桌而坐",
            "end_state": "代表们举手表决",
            "link_type": "CONTINUOUS_ACTION",
        }

    async def test_bare_array_keeps_all_shots(self):
        ctx = self.context()
        parsed = _coerce_object([self.shot(1), self.shot(2)])
        self.assertEqual(len(parsed["items"]), 2, "公共解析层不能只保留数组第一项")
        ctx.agent_ask = AsyncMock(return_value=parsed)

        with patch.object(agents_six_impl, "_video_model", return_value=""):
            result = await agents_six_impl.storyboard_six_agent(
                ctx, None, {"segment_index": 0, "target_seconds": 20}
            )

        self.assertEqual(result["shot_count"], 2)
        self.assertEqual(len(ctx.state["storyboards"]["seg1"]["shots"]), 2)

    async def test_skeleton_reply_retries_locally_and_accepts_shots_key(self):
        ctx = self.context()
        ctx.agent_ask = AsyncMock(side_effect=[
            {"storyboard": [{"shot_id": "SB-001"}]},
            {"shots": [self.shot(1)]},
        ])

        with patch.object(agents_six_impl, "_video_model", return_value=""):
            result = await agents_six_impl.storyboard_six_agent(
                ctx, None, {"segment_index": 0, "target_seconds": 20}
            )

        self.assertEqual(ctx.agent_ask.await_count, 2)
        self.assertTrue(all(
            call.kwargs.get("stream") is True
            and call.kwargs.get("live_trace") is False
            and call.kwargs.get("thinking") is False
            for call in ctx.agent_ask.await_args_list
        ), "分镜必须流式保活、静默展示，并关闭隐藏深度思考")
        self.assertEqual(result["shot_count"], 1)

    async def test_two_skeleton_replies_fail_instead_of_showing_default_ten_seconds(self):
        ctx = self.context()
        skeleton = {"storyboard": [
            {"shot_id": "SB-001"},
            {"shot_id": "SB-002"},
        ]}
        ctx.agent_ask = AsyncMock(side_effect=[skeleton, skeleton])

        with patch.object(agents_six_impl, "_video_model", return_value=""), \
             self.assertRaisesRegex(ValueError, "缺时长、视频提示词"):
            await agents_six_impl.storyboard_six_agent(
                ctx, None, {"segment_index": 0, "target_seconds": 20}
            )

        self.assertNotIn("storyboards", ctx.state)


class AgentStreamingOverrideTests(unittest.IsolatedAsyncioTestCase):
    async def test_per_call_stream_false_uses_complete_response_path(self):
        from types import SimpleNamespace

        ctx = AgentContext(run_id="", brief={})
        ctx.model_settings = SimpleNamespace(stream=True)
        ctx.config_for = lambda *args, **kwargs: SimpleNamespace(model="fake-model")
        ctx.trace_step = lambda *args, **kwargs: None
        ctx._ask_streaming = AsyncMock(return_value={"wrong": "streamed"})

        with patch(
            "app.tools.llm.chat_json", new_callable=AsyncMock,
            return_value={"storyboard": [{"shot_id": "SB-001"}]},
        ) as complete_call:
            result = await ctx._agent_ask_bound(
                "storyboard", system="system", user="user", stream=False,
            )

        ctx._ask_streaming.assert_not_awaited()
        complete_call.assert_awaited_once()
        self.assertEqual(result["storyboard"][0]["shot_id"], "SB-001")

    async def test_silent_stream_buffers_reasoning_until_completion(self):
        import time
        from types import SimpleNamespace

        from app.tools.llm import LLMResult

        ctx = AgentContext(run_id="", brief={})
        ctx.config_for = lambda *args, **kwargs: SimpleNamespace(model="fake-model")
        ctx.trace_step = lambda *args, **kwargs: None
        ctx._trace_reasoning = Mock()

        async def fake_stream(*args, **kwargs):
            self.assertFalse(kwargs["thinking"])
            await kwargs["on_delta"]("reasoning", "先分析完整剧本")
            await kwargs["on_delta"]("content", '{"storyboard": []}')
            self.assertEqual(ctx._trace_reasoning.call_count, 0)
            return LLMResult(
                text='{"storyboard": []}', raw={}, model="fake-model",
                reasoning="先分析完整剧本", finish_reason="stop",
            )

        with patch("app.tools.llm_stream.stream_chat", side_effect=fake_stream):
            result = await ctx._ask_streaming(
                "storyboard", system="system", user="user", images=None,
                max_tokens=2000, independent_from="", started=time.perf_counter(),
                live_trace=False, thinking=False,
            )

        self.assertEqual(result, {"storyboard": []})
        ctx._trace_reasoning.assert_called_once()
        self.assertEqual(ctx._trace_reasoning.call_args.args[0], "模型思考过程")
        self.assertEqual(ctx._trace_reasoning.call_args.args[1], "先分析完整剧本")


if __name__ == "__main__":
    unittest.main()
