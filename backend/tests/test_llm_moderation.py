"""审核类失败的两种方向、四级处置。

锁定的是一次真实故障：一段国民党一大的分镜里写着「会场悬挂青天白日满地红旗」，
一致性 agent 的整段提示词被判敏感直接 400，整个阶段失败；而更隐蔽的另一半是
出向——模型自己写出敏感词，HTTP 200 + `finish_reason=content_filter`，正文为空，
此前落到 `extract_json` 的「不是合法 JSON」分支，用户按那句提示永远修不好。
"""
from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

import httpx

from app.config import LLMConfig
from app.platform import llm_errors, tool_feedback
from app.tools import llm


def config(**kw) -> LLMConfig:
    base = dict(
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key="sk-test-0000000000000000000000000000000000000000",
        model="glm-4.5",
        timeout_seconds=30.0,
    )
    base.update(kw)
    return LLMConfig(**base)


def response(payload: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json=payload,
        request=httpx.Request("POST", "https://example.test/chat/completions"),
    )


def ok(text: str = '{"a": 1}', finish: str = "stop") -> httpx.Response:
    return response({"choices": [{"message": {"content": text}, "finish_reason": finish}]})


def filtered_out() -> httpx.Response:
    """出向拦截：200，但没有正文。"""
    return response(
        {"choices": [{"message": {"content": ""}, "finish_reason": "content_filter"}]}
    )


def blocked_in() -> httpx.Response:
    return response(
        {"error": {"message": "系统检测到输入或生成内容可能包含不安全或敏感内容", "code": "1301"}},
        status=400,
    )


class OutboundModerationTests(unittest.IsolatedAsyncioTestCase):
    """出向拦截：正文被撤回，不是提示词的问题。"""

    async def test_filtered_reply_is_retried_with_softer_wording(self):
        seen: list[dict] = []

        async def fake_post(_self, _url, *, json, **_kwargs):
            seen.append(json)
            return filtered_out() if len(seen) == 1 else ok()

        with patch.object(httpx.AsyncClient, "post", fake_post):
            result = await llm.chat(config(), system="s", user="战场的场面", max_tokens=16)

        self.assertEqual(result.text, '{"a": 1}')
        self.assertEqual(len(seen), 2, "出向拦截也要重发一次，否则这一半永远兜不住")
        self.assertEqual(result.moderation_plan, "soften")
        # 第二版必须有降调要求，否则重发的是同一个必然被拦的请求。
        self.assertIn("中性", seen[1]["messages"][0]["content"])

    async def test_repeated_filtering_reports_moderation_not_json(self):
        """连续出向拦截报的是审核，不是「无法解析 JSON」——后者用户修不好。"""
        with patch.object(
            httpx.AsyncClient, "post", AsyncMock(side_effect=lambda *_a, **_k: filtered_out())
        ):
            with self.assertRaises(llm.LLMError) as ctx:
                await llm.chat(config(), system="s", user="战场的场面", max_tokens=16)

        self.assertEqual(ctx.exception.error_type, "moderation_out")
        self.assertIn("内容审核", str(ctx.exception))
        self.assertNotIn("JSON", str(ctx.exception))

    async def test_nonstream_can_disable_hidden_thinking(self):
        seen: list[dict] = []

        async def fake_post(_self, _url, *, json, **_kwargs):
            seen.append(json)
            return ok()

        with patch.object(httpx.AsyncClient, "post", fake_post):
            await llm.chat(
                config(), system="s", user="u", max_tokens=16, thinking=False,
            )

        self.assertEqual(seen[0]["thinking"], {"type": "disabled"})


class InboundModerationTests(unittest.IsolatedAsyncioTestCase):
    """入向拦截：提示词撞了词表，按「换词 → 整句降调」逐级加力。"""

    async def test_two_levels_are_tried_before_giving_up(self):
        seen: list[dict] = []

        async def fake_post(_self, _url, *, json, **_kwargs):
            seen.append(json)
            return blocked_in() if len(seen) < 3 else ok()

        with patch.object(httpx.AsyncClient, "post", fake_post):
            result = await llm.chat(
                config(), system="s", user="会场悬挂青天白日旗。", max_tokens=16
            )

        self.assertEqual(len(seen), 3, "换词之后再失败，还有「整句降调」这一级")
        users = [s["messages"][1]["content"] for s in seen]
        self.assertIn("青天白日旗", users[0])
        self.assertNotIn("青天白日", users[1])
        self.assertIn("民国时期的旗帜", users[1])
        # 第三级必须带上改写到表述本身的要求，而不只是同样的替换再发一次。
        self.assertIn("表述要求", seen[2]["messages"][0]["content"])
        self.assertEqual(result.moderation_plan, "soften")

    async def test_all_levels_failing_is_classified_as_moderation(self):
        with patch.object(
            httpx.AsyncClient, "post", AsyncMock(side_effect=lambda *_a, **_k: blocked_in())
        ):
            with self.assertRaises(llm.LLMError) as ctx:
                await llm.chat(config(), system="s", user="青天白日旗", max_tokens=16)

        self.assertEqual(ctx.exception.error_type, "moderation_in")
        # 三级都试过了，不能再建议「重试」——重试得到的是同一个结果。
        self.assertIn("改写", str(ctx.exception))

    async def test_prompt_without_known_terms_skips_the_pointless_level(self):
        """没命中已知词时「换词」等于什么都没改，跳过它，别发一次一模一样的请求。"""
        seen: list[dict] = []

        async def fake_post(_self, _url, *, json, **_kwargs):
            seen.append(json)
            return blocked_in() if len(seen) == 1 else ok()

        with patch.object(httpx.AsyncClient, "post", fake_post):
            await llm.chat(config(), system="s", user="完全中性的描述", max_tokens=16)

        self.assertEqual(len(seen), 2)


class ModerationRecordTests(unittest.IsolatedAsyncioTestCase):
    async def test_take_moderation_reports_what_was_rewritten(self):
        """改写要能被上层读到：界面上得告诉用户这句话被换过。"""
        llm.take_moderation()
        seen: list[dict] = []

        async def fake_post(_self, _url, *, json, **_kwargs):
            seen.append(json)
            return blocked_in() if len(seen) == 1 else ok()

        with patch.object(httpx.AsyncClient, "post", fake_post):
            await llm.chat(config(), system="s", user="会场悬挂青天白日旗", max_tokens=16)

        notes = llm.take_moderation()
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["plan"], "neutralize")
        self.assertIn("青天白日旗", notes[0]["terms"])
        self.assertEqual(llm.take_moderation(), [], "取过一次就要清空，否则下一段会重复报")


class FeedbackContractTests(unittest.TestCase):
    """模型失败要落到和 MCP 工具同一套契约上，界面才说得清下一步。"""

    def test_moderation_is_not_retryable_and_says_what_to_do(self):
        error, text = llm_errors.message_for(
            llm.LLMError("内容审核拦截", error_type="moderation_in"), stage="一致性"
        )
        self.assertFalse(error["retryable"], "原样重试一定还是同一个 400")
        self.assertIn("改写", error["user_action"])
        self.assertIn("任务影响", text)
        self.assertIn("一致性", text)

    def test_moderation_impact_talks_about_one_segment_not_the_whole_run(self):
        _, text = llm_errors.message_for(
            llm.LLMError("内容审核拦截", error_type="moderation_out"), stage="一致性"
        )
        self.assertIn("这一段", text)
        self.assertNotIn("已在生成前暂停", text)

    def test_transient_errors_stay_retryable(self):
        error, _ = llm_errors.message_for(
            llm.LLMError("LLM 请求超时", error_type="timeout"), stage="分镜设计"
        )
        self.assertTrue(error["retryable"])
        self.assertEqual(error["error_type"], "timeout")

    def test_every_error_type_has_a_default_action(self):
        """新增分类却忘了给处置，界面就会显示一句默认的「请重试」。"""
        for kind in llm.ERROR_TYPES:
            if kind in ("not_configured",):
                continue
            self.assertIn(kind, tool_feedback._DEFAULTS, f"{kind} 没有默认处置")

    def test_fallback_policy_reads_the_classification_not_the_wording(self):
        """换不换模型由分类决定，不由中文文案决定——文案一改就会漏掉一整类。"""
        from app import free_models

        # 文案里不含任何关键词，分类照样生效。
        self.assertFalse(free_models.should_fallback("随便一句话", "moderation_in"))
        self.assertFalse(free_models.should_fallback("随便一句话", "balance"))
        self.assertTrue(free_models.should_fallback("随便一句话", "auth"))
        self.assertTrue(free_models.should_fallback("随便一句话", "timeout"))
        # 没有分类时退回文案判断，老调用点不受影响。
        self.assertFalse(free_models.should_fallback("内容含敏感信息"))


if __name__ == "__main__":
    unittest.main()
