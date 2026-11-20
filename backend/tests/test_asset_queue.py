"""全片资产表与串行出图队列。

三条规则，每条都对应一种曾经发生过的浪费：

  · 同一个人/场景/道具在全片只登记一次 —— 否则两段各画一张，两张脸不一样，
    钱也付两次；
  · 队列一次只提交一条 —— 三条同时在飞正是图片接口限流的来源；
  · 提示词为空不提交 —— 空串照样计费，还会画出与内容无关的图。

用桩生成器跑，不碰网络。
"""
from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.agents.base import AgentContext
from app.doc import asset_queue, media_mode
from app.doc.generator import Generator
from app.models_config import ModelSettings, ProviderProfile


class FakeGenerator:
    """Records concurrency and returns a fake file. Optionally fails once."""

    image_model = "fake-image"

    def __init__(self, fail_on: tuple[int, ...] = ()) -> None:
        self.inflight = 0
        self.peak = 0
        self.prompts: list[str] = []
        self.fail_on = set(fail_on)

    async def render_image(self, prompt: str, out_dir, size: str):
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        self.prompts.append(prompt)
        await asyncio.sleep(0.01)
        self.inflight -= 1
        n = len(self.prompts)
        if n in self.fail_on:
            raise RuntimeError(f"上游 500（第 {n} 张）")
        out_dir.mkdir(parents=True, exist_ok=True)
        local = out_dir / f"fake-{id(self)}-{n}.png"
        local.write_bytes(b"fake-image")
        return SimpleNamespace(url=f"http://img/{n}.png", local_path=str(local))


def manifest(segment_id: str, *, character_prompt: str) -> dict:
    """One character, one scene, one task per asset."""
    return {
        "segment_id": segment_id,
        "character_assets": [
            {"character_id": f"CHAR-{segment_id}", "name": "孙中山",
             "four_view_prompt": character_prompt, "negative_prompt": ""},
        ],
        "scene_assets": [
            {"scene_id": f"SCENE-{segment_id}", "name": "石库门客厅",
             "prompt": "夜晚的旧式客厅，长桌、油灯"},
        ],
        "prop_assets": [],
        "image_generation_tasks": [
            {"asset_id": f"CHAR-{segment_id}", "asset_type": "character_four_view",
             "image_prompt": character_prompt, "aspect_ratio": "1:1", "status": "pending"},
            {"asset_id": f"SCENE-{segment_id}", "asset_type": "scene",
             "image_prompt": "夜晚的旧式客厅", "aspect_ratio": "16:9", "status": "pending"},
        ],
    }


def context(*manifests: dict) -> AgentContext:
    ctx = AgentContext(run_id="test-assets", brief={})
    ctx.state.update({
        "generate_images": True,
        "segments": [{"id": m["segment_id"]} for m in manifests],
        "asset_index": {},
    })
    ctx.trace_step = lambda *a, **k: None
    return ctx


class LinkTests(unittest.TestCase):
    def test_second_segment_adopts_the_first_prompt(self):
        ctx = context()
        ctx.state["asset_index"]["seg1"] = manifest(
            "seg1", character_prompt="四视图 v1：长脸、短发、黑色中山装"
        )
        asset_queue.link(ctx, ctx.state["asset_index"]["seg1"], segment_index=0)

        second = manifest("seg2", character_prompt="四视图 v2：另一个人")
        ctx.state["asset_index"]["seg2"] = second
        counts = asset_queue.link(ctx, second, segment_index=1)

        self.assertEqual(
            second["character_assets"][0]["four_view_prompt"],
            "四视图 v1：长脸、短发、黑色中山装",
            "同一个人在两段里必须共用一套提示词，否则会画出两张脸",
        )
        self.assertEqual(counts["reused_prompt"], 1)
        # 两段各有一条场景，名称相同 —— 全片只有一条登记。
        self.assertEqual(len(ctx.state["asset_registry"]), 2)

    def test_registry_is_keyed_by_canonical_name(self):
        ctx = context()
        first = manifest("seg1", character_prompt="p")
        first["character_assets"][0]["name"] = "孙中山（特写）"
        asset_queue.link(ctx, first, segment_index=0)
        second = manifest("seg2", character_prompt="p2")
        second["character_assets"][0]["name"] = "孙中山的手"
        asset_queue.link(ctx, second, segment_index=1)
        self.assertEqual(len(ctx.state["asset_registry"]), 2, "人物一条 + 场景一条")


class RenderQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_submission_at_a_time_and_one_picture_per_asset(self):
        ctx = context()
        for index, sid in enumerate(("seg1", "seg2")):
            m = manifest(sid, character_prompt=f"四视图 {sid}")
            ctx.state["asset_index"][sid] = m
            asset_queue.link(ctx, m, segment_index=index)

        gen = FakeGenerator()
        ctx.state["__generator"] = gen
        with patch.object(media_mode, "prompt_only", return_value=False):
            result = await asset_queue.render_all(ctx)

        # 两个人物同名 → 一张图；两个场景同名 → 一张图。共 2 条，不是 4 条。
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["rendered"], 2)
        self.assertEqual(gen.peak, 1, "队列必须串行：同时只有一条在飞")
        for sid in ("seg1", "seg2"):
            m = ctx.state["asset_index"][sid]
            self.assertTrue(m["character_assets"][0]["image_path"], sid)
            self.assertTrue(m["scene_assets"][0]["image_path"], sid)
            self.assertTrue(all(t["status"] == "done" for t in m["image_generation_tasks"]))

    async def test_failure_is_recorded_and_stops_later_submissions(self):
        ctx = context()
        m = manifest("seg1", character_prompt="p")
        m["prop_assets"] = [{"prop_id": "PROP-1", "name": "茶碗", "prompt": "白瓷茶碗"}]
        m["image_generation_tasks"].append(
            {"asset_id": "PROP-1", "asset_type": "prop", "image_prompt": "白瓷茶碗",
             "aspect_ratio": "1:1", "status": "pending"}
        )
        ctx.state["asset_index"]["seg1"] = m
        asset_queue.link(ctx, m, segment_index=0)

        gen = FakeGenerator(fail_on=(1,))
        ctx.state["__generator"] = gen
        with patch.object(media_mode, "prompt_only", return_value=False):
            result = await asset_queue.render_all(ctx)

        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["rendered"], 0)
        self.assertEqual(len(gen.prompts), 1, "当前图片未落盘，不能提交下一条")
        failed = [t for t in m["image_generation_tasks"] if t["status"] == "failed"]
        self.assertEqual(len(failed), 1)
        self.assertIn("500", failed[0]["error"])

    async def test_empty_prompt_is_never_submitted(self):
        ctx = context()
        m = manifest("seg1", character_prompt="p")
        m["scene_assets"][0]["prompt"] = ""
        ctx.state["asset_index"]["seg1"] = m
        asset_queue.link(ctx, m, segment_index=0)

        gen = FakeGenerator()
        ctx.state["__generator"] = gen
        with patch.object(media_mode, "prompt_only", return_value=False):
            result = await asset_queue.render_all(ctx)

        self.assertEqual(result["failed"], 1)
        self.assertEqual(len(gen.prompts), 1, "空提示词不能提交给上游或越过它继续")
        self.assertTrue(all(p.strip() for p in gen.prompts))

    async def test_missing_local_file_is_failure_before_next_submission(self):
        class MissingFileGenerator(FakeGenerator):
            async def render_image(self, prompt: str, out_dir, size: str):
                self.prompts.append(prompt)
                return SimpleNamespace(
                    url="http://img/result.png",
                    local_path=str(out_dir / "not-created.png"),
                )

        ctx = context()
        m = manifest("seg1", character_prompt="人物提示词")
        ctx.state["asset_index"]["seg1"] = m
        asset_queue.link(ctx, m, segment_index=0)
        gen = MissingFileGenerator()
        ctx.state["__generator"] = gen

        with patch.object(media_mode, "prompt_only", return_value=False):
            result = await asset_queue.render_all(ctx)

        self.assertEqual(result["rendered"], 0)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(len(gen.prompts), 1, "未形成本地文件时必须停止队列")
        self.assertEqual(m["image_generation_tasks"][0]["status"], "failed")
        self.assertEqual(m["image_generation_tasks"][1]["status"], "pending")

    async def test_switch_off_keeps_prompts_and_draws_nothing(self):
        ctx = context()
        m = manifest("seg1", character_prompt="p")
        ctx.state["asset_index"]["seg1"] = m
        ctx.state["generate_images"] = False
        asset_queue.link(ctx, m, segment_index=0)

        gen = FakeGenerator()
        ctx.state["__generator"] = gen
        with patch.object(media_mode, "prompt_only", return_value=False):
            result = await asset_queue.render_all(ctx)

        self.assertEqual(gen.prompts, [], "未勾选出图时不能提交任何一条")
        self.assertEqual(result["skipped"], 2)
        self.assertTrue(m["character_assets"][0]["four_view_prompt"], "提示词必须保留")


class IndependentMediaSwitchTests(unittest.IsolatedAsyncioTestCase):
    def test_api_allows_video_only_request(self):
        from app.api import doc as api_doc

        api_doc._check_media_switches(
            api_doc.CreateRun(
                chapter_text="1924 年 1 月，国民党一大在广州召开。",
                target_seconds=30,
                generate_images=False,
                generate_videos=True,
            )
        )

        # 只勾出图、或两个都勾，都是合法组合。
        api_doc._check_media_switches(
            api_doc.CreateRun(
                chapter_text="x", target_seconds=30,
                generate_images=True, generate_videos=False,
            )
        )
        api_doc._check_media_switches(
            api_doc.CreateRun(
                chapter_text="x", target_seconds=30,
                generate_images=True, generate_videos=True,
            )
        )


class AgentMediaBindingTests(unittest.TestCase):
    def test_media_model_pin_does_not_hijack_prompt_writing_provider(self):
        text = ProviderProfile(
            id="text", label="文本模型", base_url="https://text.test/v1",
            api_key="key-text-model", model="text-chat",
        )
        media = ProviderProfile(
            id="media", label="媒体网关", base_url="http://10.0.0.8:8190/v1",
            api_key="key-media-gateway", model="gateway-default",
            vision_model="vision-one", image_model="image-one", video_model="video-one",
        )
        settings = ModelSettings(
            providers=[text, media],
            assignments={
                "consistency": "media:image-one",
                "video_gen": "media:video-one",
                "quality_review": "media:vision-one",
            },
            default_provider="text",
        )

        self.assertIs(settings.provider_for("consistency"), text)
        self.assertIs(settings.provider_for("video_gen"), text)
        self.assertIs(settings.provider_for("quality_review"), text)

    def test_explicit_text_model_pin_still_selects_assigned_provider(self):
        default = ProviderProfile(
            id="default", label="默认", base_url="https://default.test/v1",
            api_key="key-default-model", model="default-chat",
        )
        assigned = ProviderProfile(
            id="assigned", label="指定", base_url="https://assigned.test/v1",
            api_key="key-assigned-model", model="assigned-chat",
            image_model="assigned-image",
        )
        settings = ModelSettings(
            providers=[default, assigned],
            assignments={"consistency": "assigned:assigned-chat"},
            default_provider="default",
        )

        self.assertIs(settings.provider_for("consistency"), assigned)

    def test_consistency_uses_its_assigned_image_model(self):
        first = ProviderProfile(
            id="first", label="一号", base_url="https://one.test/v1", api_key="key-one",
            image_model="image-one", video_model="video-one",
        )
        second = ProviderProfile(
            id="second", label="二号", base_url="https://two.test/v1", api_key="key-two",
            image_model="image-two", video_model="video-two",
        )
        settings = ModelSettings(
            providers=[first, second],
            assignments={
                "consistency": "second:image-two",
                "video_gen": "first:video-one",
            },
            default_provider="first",
        )

        generator = Generator.from_settings(settings)

        self.assertEqual(generator.image_model, "image-two")
        self.assertEqual(generator.api_key, "key-two")
        self.assertEqual(generator.video_model, "video-one")
        self.assertEqual(generator.video_api_key, "key-one")


if __name__ == "__main__":
    unittest.main()
