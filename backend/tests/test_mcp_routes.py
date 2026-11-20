"""通过真实的 FastAPI 应用检查 MCP 路由，不另外启动服务。"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient

from app.connectors import feishu
from app.main import app


class MCPRoutesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        for name, value in (
            ("_CONFIG_PATH", base / "feishu.json"),
            ("_STATE_PATH", base / "feishu_oauth.json"),
        ):
            p = patch.object(feishu, name, value)
            p.start()
            self.addCleanup(p.stop)
        feishu.save_settings(
            "cli_test123456", "secret-value",
            "http://127.0.0.1:8100/api/mcp/feishu/callback",
        )
        self.client = TestClient(app)

    def test_cancel_route_clears_the_waiting_state(self):
        started = self.client.post("/api/mcp/feishu/connect").json()
        self.assertEqual(started["state"], "authorizing")

        state = parse_qs(urlparse(started["authorization_url"]).query)["state"][0]
        self.assertTrue(feishu._read_states())

        cancelled = self.client.post("/api/mcp/feishu/cancel")
        self.assertEqual(cancelled.status_code, 200, cancelled.text)
        self.assertEqual(cancelled.json()["message"], "已取消本次飞书登录")
        self.assertEqual(feishu._read_states(), {})
        self.assertEqual(
            self.client.get("/api/mcp/feishu/status").json()["state"], "not_connected"
        )
        # 被取消的 state 不能被迟到的回调复用。
        callback = self.client.get(
            "/api/mcp/feishu/callback", params={"code": "late-code", "state": state}
        )
        self.assertIn("已经过期", callback.text)

    def test_callback_page_shows_account_and_stays_open(self):
        """回调页在系统浏览器里，必须自己把自己说清楚。

        这条链路已经改成「默认浏览器整页授权」，页面不再是被 opener 关掉的弹窗：
        它得显示登录的是谁、授予了什么、能读哪些文档，并且**不自动关闭**——
        用户开的是一个真实标签页，替他关掉会像登录失败。
        """
        started = self.client.post("/api/mcp/feishu/connect").json()
        state = parse_qs(urlparse(started["authorization_url"]).query)["state"][0]
        fake = AsyncMock(side_effect=[
            {"access_token": "u-private", "expires_in": 7200,
             "scope": "docx:document:readonly search:docs:read"},
            {"name": "测试账号", "open_id": "ou_test"},
            # list_documents 用的云空间列表响应
            {"files": [{"token": "docx-1", "name": "示例文档", "type": "docx"}]},
        ])
        with patch.object(feishu, "_json_request", fake):
            page = self.client.get(
                "/api/mcp/feishu/callback", params={"code": "one-time", "state": state}
            )
        self.assertEqual(page.status_code, 200)
        self.assertIn("飞书登录完成", page.text)
        self.assertIn("测试账号", page.text)
        self.assertIn("postMessage", page.text)
        # 账号、权限、文档三件事都在页面上。
        self.assertIn("搜索云文档", page.text)
        self.assertIn("示例文档", page.text)
        # 关键行为：不再自动关闭。
        self.assertNotIn("window.close", page.text)
        self.assertNotIn("setTimeout", page.text)
        # 凭证不会出现在浏览器里。
        self.assertNotIn("u-private", page.text)

    def test_callback_failure_is_reported_but_not_a_500(self):
        page = self.client.get(
            "/api/mcp/feishu/callback", params={"error": "access_denied", "error_description": "用户拒绝"}
        )
        self.assertEqual(page.status_code, 200)
        self.assertIn("登录未完成", page.text)
        self.assertIn("用户拒绝", page.text)


if __name__ == "__main__":
    unittest.main()
