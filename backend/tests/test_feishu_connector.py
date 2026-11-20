"""飞书 OAuth 测试：全部用假的 HTTP 响应，不需要真实账号。"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlparse

from app.connectors import feishu


class FeishuConnectorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path_patch = patch.object(feishu, "_CONFIG_PATH", Path(self.temp.name) / "feishu.json")
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)
        self.state_patch = patch.object(feishu, "_STATE_PATH", Path(self.temp.name) / "feishu_oauth.json")
        self.state_patch.start()
        self.addCleanup(self.state_patch.stop)

    def pending_states(self):
        return feishu._read_states()

    def save(self):
        return feishu.save_settings(
            "cli_test123456", "secret-value",
            "http://127.0.0.1:8000/api/mcp/feishu/callback",
        )

    def connect_config(self):
        self.save()
        data = json.loads(feishu._CONFIG_PATH.read_text(encoding="utf-8"))
        data.update(access_token="u-test", refresh_token="r-test", expires_at=time.time() + 3600)
        feishu._write_config(data)

    def test_authorization_url_is_browser_oauth_with_minimum_scopes(self):
        self.save()
        started = feishu.start_login()
        parsed = urlparse(started["authorization_url"])
        query = parse_qs(parsed.query)
        self.assertEqual(parsed.netloc, "accounts.feishu.cn")
        self.assertEqual(query["client_id"], ["cli_test123456"])
        self.assertEqual(query["response_type"], ["code"])
        self.assertIn("search:docs:read", query["scope"][0])
        self.assertIn(query["state"][0], self.pending_states())
        self.assertNotIn("secret-value", started["authorization_url"])

    async def test_callback_exchanges_code_and_never_returns_tokens(self):
        self.save()
        started = feishu.start_login()
        state = parse_qs(urlparse(started["authorization_url"]).query)["state"][0]
        fake = AsyncMock(side_effect=[
            {"access_token": "u-private", "refresh_token": "r-private", "expires_in": 7200},
            {"name": "测试账号", "open_id": "ou_test"},
        ])
        with patch.object(feishu, "_json_request", fake):
            result = await feishu.complete_login("one-time-code", state)
        self.assertTrue(result["connected"])
        self.assertEqual(result["account"], "测试账号")
        self.assertNotIn("u-private", str(result))
        stored = json.loads(feishu._CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(stored["access_token"], "u-private")

    async def test_search_normalises_and_rejects_non_feishu_urls(self):
        self.connect_config()
        payload = {"results": [
            {"title": "<em>产品</em>方案", "entity_type": "docx", "result_meta": {"url": "https://acme.feishu.cn/docx/abc"}},
            {"title": "恶意结果", "result_meta": {"url": "https://example.com/steal"}},
        ]}
        with patch.object(feishu, "_json_request", new_callable=AsyncMock, return_value=payload) as request:
            result = await feishu.search_documents("产品方案")
        self.assertTrue(result["ok"])
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["results"][0]["title"], "产品方案")
        self.assertIn("search/v2/doc_wiki/search", request.await_args.args[1])

    async def test_fetch_reads_docx_and_rejects_non_feishu_url(self):
        self.connect_config()
        with patch.object(feishu, "_json_request", new_callable=AsyncMock, return_value={"content": "# 标题\n正文"}) as request:
            result = await feishu.fetch_document("https://acme.feishu.cn/docx/abc")
            refused = await feishu.fetch_document("https://example.com/docx/abc")
        self.assertTrue(result["material_ready"])
        self.assertIn("正文", result["text"])
        self.assertFalse(refused["ok"])
        self.assertEqual(request.await_count, 1)

    def test_error_redacts_oauth_secrets(self):
        safe = feishu._safe_error("access_token=token-value client_secret:abcdef")
        self.assertNotIn("token-value", safe)
        self.assertNotIn("abcdef", safe)

    async def test_pending_state_survives_a_restart(self):
        """后端重启不能把正在进行的扫码作废。

        以前 state 只放内存字典，uvicorn --reload 或 start.ps1 重启端口就丢了，
        用户扫完码回来正好被告知「授权已经过期」。
        """
        self.save()
        started = feishu.start_login()
        state = parse_qs(urlparse(started["authorization_url"]).query)["state"][0]
        # 模拟重启：内存字典没了，只剩文件。
        self.assertIn(state, self.pending_states())
        fake = AsyncMock(side_effect=[
            {"access_token": "u-private", "expires_in": 7200},
            {"name": "重启后账号"},
        ])
        with patch.object(feishu, "_json_request", fake):
            result = await feishu.complete_login("one-time-code", state)
        self.assertTrue(result["connected"])
        self.assertEqual(result["account"], "重启后账号")
        self.assertNotIn(state, self.pending_states())

    async def test_cancel_login_stops_the_panel_waiting(self):
        """没授权就关弹窗，必须结束这次等待。"""
        self.save()
        started = feishu.start_login()
        state = parse_qs(urlparse(started["authorization_url"]).query)["state"][0]
        self.assertEqual((await feishu.status())["state"], "authorizing")
        cancelled = feishu.cancel_login()
        self.assertEqual(cancelled["message"], "已取消本次飞书登录")
        self.assertEqual((await feishu.status())["state"], "not_connected")
        # state 真的没了，迟到的回调也补不上。
        with self.assertRaises(feishu.FeishuError):
            await feishu.complete_login("one-time-code", state)

    def test_expired_states_are_dropped_on_read(self):
        feishu._write_states({"stale": time.time() - 5, "fresh": time.time() + 60})
        states = feishu._read_states()
        self.assertNotIn("stale", states)
        self.assertIn("fresh", states)

    def test_a_new_login_supersedes_the_previous_attempt(self):
        """废弃的尝试不能越积越多，把界面一直卡在「授权中」。

        以前每点一次登录就多一个待授权 state，关掉再点还留着，
        面板要等全部过期才恢复正常。
        """
        self.save()
        first = feishu.start_login()
        second = feishu.start_login()
        self.assertEqual(len(feishu._read_states()), 1)
        first_state = parse_qs(urlparse(first["authorization_url"]).query)["state"][0]
        second_state = parse_qs(urlparse(second["authorization_url"]).query)["state"][0]
        self.assertNotIn(first_state, feishu._read_states())
        self.assertIn(second_state, feishu._read_states())

    async def test_login_reports_account_and_granted_permissions(self):
        """登录结果要带账号和权限清单。

        「连接成功」不足以让人相信真的连上了，也不解释为什么某篇文档读不到。
        权限取令牌响应里实际返回的 scope，而不是我们申请的清单。
        """
        self.save()
        started = feishu.start_login()
        state = parse_qs(urlparse(started["authorization_url"]).query)["state"][0]
        fake = AsyncMock(side_effect=[
            {
                "access_token": "u-private",
                "expires_in": 7200,
                "scope": "docx:document:readonly search:docs:read",
            },
            {"name": "张三", "email": "zhangsan@example.com", "tenant_name": "示例公司"},
        ])
        with patch.object(feishu, "_json_request", fake):
            result = await feishu.complete_login("one-time-code", state)
        self.assertTrue(result["connected"])
        self.assertEqual(result["account"], "张三")
        self.assertEqual(result["account_detail"]["email"], "zhangsan@example.com")
        labels = [item["label"] for item in result["permissions"]]
        self.assertIn("读取新版文档正文", labels)
        self.assertIn("搜索云文档", labels)
        # 申请了但没被授予的权限不该出现在清单里。
        self.assertNotIn("读取知识库", labels)

    async def test_document_list_normalises_and_drops_entries_without_token(self):
        self.connect_config()
        payload = {"files": [
            {"token": "docx1", "name": "产品方案 v2", "type": "docx", "owner_name": "李四"},
            {"name": "没有 token 的脏数据", "type": "docx"},
            {"token": "sheet1", "name": "预算表", "type": "sheet"},
        ]}
        with patch.object(feishu, "_json_request", new_callable=AsyncMock, return_value=payload):
            result = await feishu.list_documents(20)
        self.assertTrue(result["ok"])
        self.assertEqual(result["count"], 2)
        self.assertEqual(result["documents"][0]["title"], "产品方案 v2")
        self.assertEqual(result["documents"][1]["type"], "sheet")
        self.assertIn("sheets", result["documents"][1]["url"])

    async def test_document_list_degrades_when_scope_is_missing(self):
        """缺 drive 权限不是登录失败，只是拿不到清单。

        很多租户只开文档读取。这时如实说「直接搜索」，而不是把
        整个已完成的登录判成错误。
        """
        self.connect_config()
        with patch.object(feishu, "_json_request", new_callable=AsyncMock,
                          side_effect=feishu.FeishuError("飞书错误 99991672：无权限")):
            result = await feishu.list_documents(20)
        self.assertFalse(result["ok"])
        self.assertEqual(result["documents"], [])
        self.assertIn("搜索", result["reason"])
        # 连接本身没受影响。
        self.assertTrue((await feishu.status())["connected"])


if __name__ == "__main__":
    unittest.main()
