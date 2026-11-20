"""阶段失败不再「跳过并继续」：这一段就是没完成。

以前一致性被审核拦下时，处理是跳过它、本段继续出片，并在界面上写「降级完成」。
理由听起来成立——剧本和分镜都在，缺的只是统一参考资产。但跳过在界面上和做完
长得一样：用户拿到的是一段没做过一致性校验的片，而清单上那一步是绿的。修不动
的问题被伪装成了没事。

现在的规则：这一段停下，其余分段跑完，清单上标「未完成」，并且说清是谁的问题。
"""
from __future__ import annotations

import json
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

from app.doc import intake, media_mode, pipeline_six
from app import models_config
from app.models_config import ModelSettings, ProviderProfile
from app.tools.llm import LLMError


class StageFailureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.events: list[dict] = []
        # 输入识别要挡掉：这一段测的是分段生产，不该为了跑通去调真实模型。
        self.stack.enter_context(patch.object(intake, "_config", return_value=object()))
        self.stack.enter_context(
            patch.object(
                intake.llm_tools,
                "chat_json",
                new_callable=AsyncMock,
                return_value={"intent": "material", "topic": "", "desired_seconds": 0, "reason": "test"},
            )
        )

    async def collect(self, event):
        self.events.append(event)

    def mock_agents(self, *, failing: str = "consistency", failed_segments=("seg1",)):
        def resolve(name):
            async def run(ctx, definition, options):
                index = int(options.get("segment_index") or 0)
                segment = f"seg{index + 1}"
                if name == "director":
                    return {
                        "segments": [
                            {"id": "seg1", "title": "一", "order": 0},
                            {"id": "seg2", "title": "二", "order": 1},
                        ],
                        "review_mode": "NORMAL",
                        "director_summary": "现成素材 / 工作流 default：切 2 段",
                    }
                if name == failing and segment in failed_segments:
                    raise LLMError(
                        "内容审核连续拦截了 3 次（glm-4.5，HTTP 400）：提示词被内容安全策略判为敏感",
                        status=400,
                        error_type="moderation_in",
                    )
                if name == "quality_review":
                    return {"critic_verdict": {"approved": True}}
                return {}

            return None, run

        self.stack.enter_context(
            patch.object(pipeline_six.registry, "catalog", return_value=[{"id": "script_adaptor"}])
        )
        self.stack.enter_context(patch.object(pipeline_six.registry, "resolve", side_effect=resolve))
        self.stack.enter_context(patch.object(media_mode, "prompt_only", return_value=True))
        self.stack.enter_context(
            patch.object(pipeline_six, "_assemble", new_callable=AsyncMock, return_value="")
        )
        for name, value in [
            ("_video_capabilities", {}),
            ("_image_capabilities", {}),
            ("_image_model_name", ""),
            ("_video_model_name", ""),
        ]:
            self.stack.enter_context(patch.object(pipeline_six, name, return_value=value))

    async def run_pipeline(self):
        return await pipeline_six.run_pipeline_six(
            run_id="test-no-skip",
            chapter_text=(
                "1924 年 1 月，国民党一大在广州召开，实行联俄、联共、扶助农工三大政策。"
                "大会通过了新的党章，标志着第一次国共合作正式形成。"
            ),
            target_seconds=30,
            model_settings=None,
            max_rework=0,
            on_event=self.collect,
        )

    def errors(self) -> list[dict]:
        return [e for e in self.events if (e.get("payload") or {}).get("mcp_error")]

    def plans(self) -> list[dict]:
        return [e for e in self.events if e.get("kind") == "plan"]

    async def test_failed_stage_leaves_the_segment_unfinished(self):
        self.mock_agents()
        result = await self.run_pipeline()

        self.assertEqual(result["approved_segments"], 1, "只有没失败的 seg2 算通过")
        self.assertEqual(result["degraded_segments"], [], "不再有「降级完成」这种东西")
        seg1 = next(s for s in result["segments"] if s["segment_id"] == "seg1")
        self.assertFalse(seg1["approved"])
        self.assertEqual(seg1["degraded"], [])

    async def test_failure_says_the_segment_did_not_finish(self):
        self.mock_agents()
        await self.run_pipeline()

        failures = self.errors()
        self.assertTrue(failures)
        error = failures[0]["payload"]["mcp_error"]
        self.assertEqual(error["error_type"], "moderation_in")
        # 说「这一段没有完成」，不说「已跳过」。跳过是上一版的说法，指的是
        # 交付了一段缺校验的片；现在交付的是没有这一段。
        self.assertIn("这一段没有完成", error["impact"])
        self.assertNotIn("已跳过", error["impact"])
        self.assertFalse(error["retryable"], "同一个提示词重试一定还是同一个 400")

    async def test_summary_reports_unfinished_not_degraded(self):
        self.mock_agents()
        await self.run_pipeline()

        text = "\n".join(str(e.get("content") or "") for e in self.events)
        self.assertNotIn("降级完成", text)
        self.assertIn("1 段未完成", text)

    async def test_plan_is_posted_before_any_production(self):
        """清单要在开工前贴出来——事后补的只是流水账。"""
        self.mock_agents()
        await self.run_pipeline()

        plans = self.plans()
        self.assertTrue(plans, "没有发出任务清单")
        first_plan = self.events.index(plans[0])
        produced = [
            i for i, e in enumerate(self.events)
            if e.get("agent") in ("script_adaptor", "storyboard", "consistency", "video_gen")
        ]
        self.assertTrue(produced)
        self.assertLess(first_plan, min(produced), "清单跑到生产后面去了")

        # 同一条清单，写多少次都只更新它自己。
        self.assertEqual({e.get("msg_key") for e in plans}, {"director:plan"})
        self.assertEqual([s["title"] for s in plans[0]["payload"]["steps"]],
                         ["取材", "分段", "逐段生产"])

    async def test_plan_records_the_failed_segment(self):
        self.mock_agents()
        await self.run_pipeline()

        final = self.plans()[-1]["payload"]
        segments = next(s for s in final["steps"] if s["id"] == "produce")["segments"]
        by_id = {s["id"]: s for s in segments}
        self.assertEqual(by_id["seg1"]["status"], "failed")
        self.assertEqual(by_id["seg2"]["status"], "done")
        self.assertIn("一致性", by_id["seg1"]["note"])
        # 收尾之后清单不能还有一步在转圈。
        self.assertTrue(all(s["status"] not in ("pending", "running") for s in final["steps"]))

    async def test_plan_records_the_material_route(self):
        self.mock_agents()
        await self.run_pipeline()

        first = self.plans()[0]["payload"]
        material = first["steps"][0]
        # 这条测试输入是现成素材，不该被写成「查了知识库」。
        self.assertEqual(material["status"], "not_needed")
        self.assertIn("直接用你给的原文", material["summary"])

    async def test_transient_failure_is_retried_in_place(self):
        """超时这类瞬时故障应该原地重试，而不是把一段判成未完成。

        用户的网关抖一下就丢一整段，而用户什么都没做错——同一请求重发大概率
        就过。审核拦截不在此列（原样重发一定还是 400），由别的用例卡着。
        """
        calls = {"storyboard": 0}

        def resolve(name):
            async def run(ctx, definition, options):
                index = int(options.get("segment_index") or 0)
                segment = f"seg{index + 1}"
                if name == "director":
                    return {
                        "segments": [
                            {"id": "seg1", "title": "一", "order": 0},
                            {"id": "seg2", "title": "二", "order": 1},
                        ],
                        "review_mode": "NORMAL",
                    }
                if name == "storyboard" and segment == "seg1":
                    calls["storyboard"] += 1
                    if calls["storyboard"] == 1:
                        raise LLMError(
                            "LLM 请求超时：60 秒内没有等到响应",
                            error_type="timeout",
                        )
                if name == "quality_review":
                    return {"critic_verdict": {"approved": True}}
                return {}

            return None, run

        self.stack.enter_context(
            patch.object(pipeline_six.registry, "catalog", return_value=[{"id": "script_adaptor"}])
        )
        self.stack.enter_context(patch.object(pipeline_six.registry, "resolve", side_effect=resolve))
        self.stack.enter_context(patch.object(media_mode, "prompt_only", return_value=True))
        self.stack.enter_context(
            patch.object(pipeline_six, "_assemble", new_callable=AsyncMock, return_value="")
        )
        for name, value in [
            ("_video_capabilities", {}),
            ("_image_capabilities", {}),
            ("_image_model_name", ""),
            ("_video_model_name", ""),
        ]:
            self.stack.enter_context(patch.object(pipeline_six, name, return_value=value))

        result = await self.run_pipeline()

        self.assertEqual(calls["storyboard"], 2, "第一次超时，第二次应该重试")
        self.assertEqual(result["approved_segments"], 2, "重试成功后两段都算通过")
        warnings = [
            e for e in self.events
            if e.get("kind") == "warning" and "自动重试" in str(e.get("content") or "")
        ]
        self.assertTrue(warnings, "重试这件事要说出来，不能悄悄重发")
        self.assertEqual(len(warnings), 1, "只重试一次就成功，不该出现两条重试通知")

    async def test_timeout_retry_escalates_the_provider_timeout(self):
        """超时重试必须先把超时上限翻倍，再发下一次。

        超时是确定性失败：同样的上限、同样的输入，原样重发几乎必然再挂——
        「自动重试」只是在原地把失败演三遍。对只有单一慢模型的用户，
        「下次多等一倍」是他们唯一的自救路径，不依赖换模型。
        """
        calls = {"storyboard": 0}

        def resolve(name):
            async def run(ctx, definition, options):
                index = int(options.get("segment_index") or 0)
                segment = f"seg{index + 1}"
                if name == "director":
                    return {
                        "segments": [
                            {"id": "seg1", "title": "一", "order": 0},
                            {"id": "seg2", "title": "二", "order": 1},
                        ],
                        "review_mode": "NORMAL",
                    }
                if name == "storyboard" and segment == "seg1":
                    calls["storyboard"] += 1
                    if calls["storyboard"] == 1:
                        raise LLMError("LLM 请求超时", error_type="timeout")
                if name == "quality_review":
                    return {"critic_verdict": {"approved": True}}
                return {}

            return None, run

        self.stack.enter_context(
            patch.object(pipeline_six.registry, "catalog", return_value=[{"id": "script_adaptor"}])
        )
        self.stack.enter_context(patch.object(pipeline_six.registry, "resolve", side_effect=resolve))
        self.stack.enter_context(patch.object(media_mode, "prompt_only", return_value=True))
        self.stack.enter_context(
            patch.object(pipeline_six, "_assemble", new_callable=AsyncMock, return_value="")
        )
        for name, value in [
            ("_video_capabilities", {}),
            ("_image_capabilities", {}),
            ("_image_model_name", ""),
            ("_video_model_name", ""),
        ]:
            self.stack.enter_context(patch.object(pipeline_six, name, return_value=value))

        provider = ProviderProfile(
            id="slow", base_url="http://gw/v1", api_key="k", model="m", timeout=60.0
        )
        settings = ModelSettings(
            providers=[provider], assignments={"storyboard": "slow"}, default_provider="slow"
        )

        result = await pipeline_six.run_pipeline_six(
            run_id="test-timeout-escalation",
            chapter_text="1924 年 1 月，国民党一大在广州召开。",
            target_seconds=30,
            model_settings=settings,
            max_rework=0,
            on_event=self.collect,
        )

        self.assertEqual(calls["storyboard"], 2)
        self.assertEqual(result["approved_segments"], 2)
        self.assertEqual(provider.timeout, 120.0, "超时重试后，供应商的上限应该翻倍")
        bumped = [
            e for e in self.events
            if "超时上限已从 60 秒升到 120 秒" in str(e.get("content") or "")
        ]
        self.assertTrue(bumped, "升了上限要告诉用户，不能悄悄改等待时长")

    async def test_resume_state_reruns_only_the_failed_segment(self):
        """断点重跑：上一轮的 state 恢复后，识别和导演都不该再跑。

        重跑改变的是「哪些段还没完成」，不是「片子切成几段」——重新切段会给出
        不同的段，已生成视频就对不上号了。
        """
        self.mock_agents()  # seg1 的 consistency 失败（审核拦截，不可重试）
        first = await self.run_pipeline()
        self.assertEqual(first["approved_segments"], 1)

        import json as _json

        state = _json.loads(_json.dumps(first["state"], ensure_ascii=False, default=str))
        classify_calls = intake.llm_tools.chat_json.await_count

        director_calls = {"n": 0}

        def resolve(name):
            async def run(ctx, definition, options):
                if name == "director":
                    director_calls["n"] += 1
                    raise AssertionError("重跑不应该再走导演切段")
                if name == "quality_review":
                    return {"critic_verdict": {"approved": True}}
                return {}

            return None, run

        self.stack.enter_context(
            patch.object(pipeline_six.registry, "catalog", return_value=[{"id": "script_adaptor"}])
        )
        self.stack.enter_context(patch.object(pipeline_six.registry, "resolve", side_effect=resolve))
        self.stack.enter_context(patch.object(media_mode, "prompt_only", return_value=True))
        self.stack.enter_context(
            patch.object(pipeline_six, "_assemble", new_callable=AsyncMock, return_value="")
        )
        for name, value in [
            ("_video_capabilities", {}),
            ("_image_capabilities", {}),
            ("_image_model_name", ""),
            ("_video_model_name", ""),
        ]:
            self.stack.enter_context(patch.object(pipeline_six, name, return_value=value))

        retry = await pipeline_six.run_pipeline_six(
            run_id="test-no-skip",
            chapter_text="1924 年 1 月，国民党一大在广州召开。",
            target_seconds=30,
            model_settings=None,
            max_rework=0,
            on_event=self.collect,
            resume_state=state,
            retry_segment_ids=["seg1"],
            prev_approved=["seg2"],
        )

        self.assertEqual(director_calls["n"], 0, "导演被重跑了")
        self.assertEqual(intake.llm_tools.chat_json.await_count, classify_calls, "输入识别被重跑了")
        self.assertEqual(retry["approved_segments"], 2, "seg1 重跑通过 + seg2 桩")
        self.assertEqual(retry["total_segments"], 2)
        # 清单沿用上一轮的：seg2 保持完成，只有 seg1 回到生产。
        plans = self.plans()
        final = plans[-1]["payload"]
        produce = next(s for s in final["steps"] if s["id"] == "produce")
        by_id = {s["id"]: s for s in produce["segments"]}
        self.assertEqual(by_id["seg2"]["status"], "done", "已通过的段不该被重置")
        self.assertEqual(by_id["seg1"]["status"], "done")
        self.assertIn("继续生成", "".join(final.get("notes") or []))


class EscalateTimeoutTests(unittest.TestCase):
    """超时重试前的自动升上限：翻倍、封顶、找不到供应商要安全退让。"""

    def _settings(self, timeout: float = 60.0):
        provider = ProviderProfile(
            id="only", base_url="http://gw/v1", api_key="k", model="m", timeout=timeout
        )
        return provider, ModelSettings(
            providers=[provider], assignments={"storyboard": "only"}, default_provider="only"
        )

    def test_timeout_doubles_per_escalation(self):
        provider, settings = self._settings(60.0)
        got = models_config.escalate_timeout(settings, "storyboard")
        self.assertEqual(got, (60.0, 120.0))
        self.assertEqual(provider.timeout, 120.0)
        got = models_config.escalate_timeout(settings, "storyboard")
        self.assertEqual(got, (120.0, 240.0))

    def test_ceiling_stops_the_escalation(self):
        provider, settings = self._settings(1200.0)
        self.assertEqual(models_config.escalate_timeout(settings, "storyboard"), (1200.0, 1800.0))
        self.assertIsNone(
            models_config.escalate_timeout(settings, "storyboard"),
            "到顶后再升必须拒绝，否则上限会被翻成天文数字",
        )

    def test_missing_settings_or_provider_is_a_safe_no_op(self):
        self.assertIsNone(models_config.escalate_timeout(None, "storyboard"))
        # 没有任何供应商时 resolve 不出对象，升级必须安全退让而不是抛异常。
        settings = ModelSettings(providers=[], assignments={}, default_provider="")
        self.assertIsNone(models_config.escalate_timeout(settings, "storyboard"))
        # 两个可用供应商、没有绑定也没有默认——选择是真歧义，不猜。
        two = ModelSettings(
            providers=[
                ProviderProfile(id="a", base_url="http://a", api_key="k", model="m", timeout=60.0),
                ProviderProfile(id="b", base_url="http://b", api_key="k", model="m", timeout=60.0),
            ],
        )
        self.assertIsNone(models_config.escalate_timeout(two, "storyboard"))

    def test_escalation_is_memory_only_not_persisted(self):
        import tempfile
        from pathlib import Path

        provider, settings = self._settings(60.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "providers.json"
            path.write_text(
                json.dumps({"providers": [{"id": "only", "timeout": 60.0}]}), encoding="utf-8"
            )
            with patch.object(models_config, "PROVIDERS_PATH", path):
                models_config.escalate_timeout(settings, "storyboard")
                raw = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            float(raw["providers"][0]["timeout"]), 60.0, "自救不该写进用户的配置文件"
        )


if __name__ == "__main__":
    unittest.main()
