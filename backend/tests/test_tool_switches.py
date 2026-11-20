from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.platform import mcp_tools, tool_switches


class ToolSwitchPersistenceTests(unittest.TestCase):
    def test_defaults_are_enabled_and_changes_persist(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            tool_switches, "_PATH", Path(directory) / "mcp_tools.json"
        ):
            self.assertEqual(tool_switches.load(), {"knowledge": True, "feishu": True})

            tool_switches.set_enabled("feishu", False)

            self.assertTrue(tool_switches.enabled("knowledge"))
            self.assertFalse(tool_switches.enabled("feishu"))

    def test_unknown_service_is_rejected(self):
        with self.assertRaises(KeyError):
            tool_switches.set_enabled("unknown", False)


class ToolSwitchGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_knowledge_service_stops_before_research(self):
        server = mcp_tools.build_outbound_server()
        with patch.object(tool_switches, "enabled", return_value=False):
            result = await server.call("search_knowledge", {"query": "测试主题"})

        self.assertEqual(result["error_type"], "tool_disabled")
        self.assertFalse(result["material_ready"])

    async def test_disabled_feishu_service_keeps_account_but_blocks_calls(self):
        server = mcp_tools.build_outbound_server()
        with patch.object(tool_switches, "enabled", return_value=False):
            result = await server.call("search_feishu_docs", {"query": "项目资料"})

        self.assertEqual(result["error_type"], "tool_disabled")
        self.assertIn("仍保持登录", result["message"])


if __name__ == "__main__":
    unittest.main()
