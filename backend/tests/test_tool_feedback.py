from __future__ import annotations

import unittest

from app.platform.tool_feedback import classify_error, failure, normalize_failure, user_message


class ToolFeedbackTests(unittest.TestCase):
    def test_permission_error_has_user_action_and_cost_impact(self):
        result = failure(
            provider="飞书", tool="fetch_feishu_doc", stage="读取正文",
            message="飞书错误 99991672：permission denied",
        )
        self.assertEqual(result["error_type"], "permission_denied")
        self.assertFalse(result["retryable"])
        self.assertIn("权限", result["user_action"])
        self.assertIn("费用", result["impact"])

    def test_no_result_is_not_reported_as_connection_failure(self):
        result = normalize_failure(
            {"reason": "没有找到匹配文档", "error_type": "no_results"},
            provider="飞书", tool="search_feishu_docs", stage="搜索文档",
            fallback_message="搜索失败",
        )
        self.assertEqual(result["error_type"], "no_results")
        self.assertNotEqual(classify_error(result["message"]), "connection_error")

    def test_user_message_names_stage_impact_and_action(self):
        result = failure(
            provider="外部资料库", tool="search", stage="连接服务",
            message="连接超时",
        )
        message = user_message(result)
        self.assertIn("连接服务", message)
        self.assertIn("任务影响", message)
        self.assertIn("建议", message)


if __name__ == "__main__":
    unittest.main()
