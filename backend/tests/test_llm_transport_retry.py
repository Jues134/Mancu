"""传输层重试回归：不访问真实模型。

这些用例锁定的是一次真实故障：分镜调用先安静地思考 100 多秒，连接在这个
窗口里被中间设备回收，`chat()` 立刻把整个阶段判死，界面显示「无法连接 LLM
服务」——而地址和 key 都是对的。重试和更准确的错误分类是本修复的两个部分。
"""
from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

import httpx

from app import free_models
from app.config import LLMConfig
from app.tools import llm


def config(**kw) -> LLMConfig:
    base = dict(
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key="1cee2c9909fc4c1badeb71c385f645e4.8LcZPlz2n1xIxsKo",
        model="glm-4.5",
        timeout_seconds=420.0,
    )
    base.update(kw)
    return LLMConfig(**base)


def ok_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": '{"a": 1}'}, "finish_reason": "stop"}
            ]
        },
        request=httpx.Request("POST", "https://example.test/chat/completions"),
    )


class TransportRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_dropped_connection_is_retried_and_succeeds(self):
        """一次 ReadError 之后重试成功，阶段不应失败。"""
        responses = [httpx.ReadError(""), ok_response()]

        async def fake_post(*_args, **_kwargs):
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with patch.object(httpx.AsyncClient, "post", AsyncMock(side_effect=fake_post)), \
                patch.object(llm.asyncio, "sleep", AsyncMock()):
            result = await llm.chat(config(), system="s", user="u", max_tokens=16)

        self.assertEqual(result.text, '{"a": 1}')
        self.assertEqual(responses, [], "重试后应当把第二次响应消费掉")

    async def test_permanent_connect_failure_still_fails(self):
        """连不上就是连不上：重试用尽后仍然报错，且消息点明 ConnectError。"""
        with patch.object(
            httpx.AsyncClient, "post", AsyncMock(side_effect=httpx.ConnectError("refused"))
        ), patch.object(llm.asyncio, "sleep", AsyncMock()):
            with self.assertRaises(llm.LLMError) as ctx:
                await llm.chat(config(), system="s", user="u", max_tokens=16)

        message = str(ctx.exception)
        self.assertIn("无法连接 LLM 服务", message)
        self.assertIn("ConnectError", message)

    def test_mid_response_drop_is_not_described_as_unreachable(self):
        """中断与不可达必须说成两件事，否则用户会去查一个没问题的地址。"""
        friendly = llm._friendly_error(
            httpx.ReadError(""), config()
        )

        message = str(friendly)
        self.assertIn("中断", message)
        # 关键区别：不能把「连上了但被掐断」说成地址不可达。
        self.assertNotIn("请确认地址可达", message)
        self.assertIn("ReadError", friendly.detail)

    def test_retry_policy_treats_interruption_as_recoverable(self):
        """中断要能走 fallback，否则换供应商这层兜底形同虚设。"""
        friendly = llm._friendly_error(httpx.RemoteProtocolError("peer closed"), config())

        self.assertTrue(free_models.should_fallback(str(friendly)))
        self.assertFalse(free_models.is_terminal(str(friendly)))

    def test_balance_error_stays_terminal(self):
        """余额问题不能被新文案误判成可重试，否则会一直重试到超时。"""
        self.assertTrue(free_models.is_terminal("账户余额或额度不足：请充值"))
        self.assertFalse(free_models.should_fallback("账户余额或额度不足：请充值"))


def moderation_400(text: str = "系统检测到输入或生成内容可能包含不安全或敏感内容") -> httpx.Response:
    return httpx.Response(
        400,
        json={"error": {"message": text, "code": "1301"}},
        request=httpx.Request("POST", "https://example.test/chat/completions"),
    )


class ModerationRetryTests(unittest.IsolatedAsyncioTestCase):
    """内容审核拒了提示词时，换中性表述重发一次。

    真实故障：一段国民党一大的分镜里写着「会场悬挂青天白日满地红旗」，
    一致性 agent 的整段提示词被智谱判成敏感内容直接 400，整个阶段失败。
    实测同一句里换掉那一个短语就能通过，其余「国民党」「蒋介石」都不受影响。
    """

    async def test_blocked_prompt_is_reworded_and_retried(self):
        seen: list[dict] = []

        async def fake_post(_self, _url, *, json, **_kwargs):
            seen.append(json)
            if len(seen) == 1:
                return moderation_400()
            return ok_response()

        with patch.object(httpx.AsyncClient, "post", fake_post):
            result = await llm.chat(
                config(), system="s", user="会场悬挂青天白日满地红旗。", max_tokens=16
            )

        self.assertEqual(result.text, '{"a": 1}')
        self.assertEqual(len(seen), 2, "被审核拒掉之后必须且只重发一次")
        first = seen[0]["messages"][1]["content"]
        second = seen[1]["messages"][1]["content"]
        self.assertIn("青天白日满地红旗", first)
        self.assertNotIn("青天白日", second, "重发时那句话必须被换掉")
        self.assertIn("民国时期的旗帜", second)
        # 除那一个短语之外的内容不能被动过。
        self.assertIn("会场悬挂", second)

    async def test_plain_400_is_not_retried(self):
        """参数错误的 400 重发一次只是再失败一次，不该伪装成审核问题。"""

        async def fake_post(*_args, **_kwargs):
            return httpx.Response(
                400,
                json={"error": {"message": "invalid temperature"}},
                request=httpx.Request("POST", "https://example.test/chat/completions"),
            )

        with patch.object(httpx.AsyncClient, "post", AsyncMock(side_effect=fake_post)):
            with self.assertRaises(llm.LLMError) as ctx:
                await llm.chat(config(), system="s", user="青天白日满地红旗", max_tokens=16)
        self.assertEqual(ctx.exception.status, 400)

    async def test_moderation_retry_failing_again_reports_the_second_error(self):
        async def fake_post(*_args, **_kwargs):
            return moderation_400()

        with patch.object(httpx.AsyncClient, "post", AsyncMock(side_effect=fake_post)):
            with self.assertRaises(llm.LLMError) as ctx:
                await llm.chat(config(), system="s", user="青天白日旗", max_tokens=16)
        self.assertIn("400", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
