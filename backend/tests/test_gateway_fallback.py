import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from app.config import LLMConfig
from app.tools import llm, llm_stream


class GatewayFallbackTests(unittest.IsolatedAsyncioTestCase):
    def config(self):
        return LLMConfig(base_url='https://gateway.test/v1', api_key='test-long-secret-value', model='test-model')

    async def test_gateway_failure_recovers_via_stream(self):
        failure = llm.LLMError('upstream failed', status=502, detail='gateway timeout')
        success = llm.LLMResult(text='{"ok":true}', raw={}, model='test-model', finish_reason='stop')
        with patch.object(llm, '_post_with_moderation', AsyncMock(side_effect=failure)), \
             patch.object(llm_stream, 'stream_chat', AsyncMock(return_value=success)) as stream:
            result = await llm.chat_json(self.config(), system='system', user='material')
        self.assertEqual(result, {'ok': True})
        stream.assert_awaited_once()
        self.assertEqual(stream.call_args.kwargs['user'], 'material')
        self.assertTrue(stream.call_args.kwargs['json_mode'])

    async def test_auth_error_does_not_retry_or_change_transport(self):
        with patch.object(llm, '_post_with_moderation', AsyncMock(side_effect=llm.LLMError('auth', status=401))), \
             patch.object(llm_stream, 'stream_chat', AsyncMock()) as stream:
            with self.assertRaises(llm.LLMError):
                await llm.chat(self.config(), system='s', user='u')
        stream.assert_not_awaited()

    async def test_both_failures_remain_visible_and_retry_is_bounded(self):
        with patch.object(llm, '_post_with_moderation', AsyncMock(side_effect=llm.LLMError('first', status=502, detail='first body'))), \
             patch.object(llm_stream, 'stream_chat', AsyncMock(side_effect=llm.LLMError('second', status=503, detail='second body'))) as stream:
            with self.assertRaises(llm.LLMError) as caught:
                await llm.chat(self.config(), system='s', user='u')
        self.assertEqual(caught.exception.status, 503)
        self.assertIn('first body', caught.exception.detail)
        self.assertIn('second body', caught.exception.detail)
        stream.assert_awaited_once()

    async def test_stream_recognizes_gateway_reasoning(self):
        async def handle(request):
            return httpx.Response(200, text='data: {"choices":[{"delta":{"reasoning":"planning"}}]}\n\ndata: {"choices":[{"delta":{"content":"{}"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        with patch.object(llm_stream.httpx, 'AsyncClient', return_value=client):
            result = await llm_stream.stream_chat(self.config(), system='s', user='u')
        self.assertEqual(result.reasoning, 'planning')
        self.assertEqual(result.text, '{}')

    async def test_stream_can_disable_hidden_thinking_for_structured_output(self):
        seen = {}

        async def handle(request):
            seen.update(json.loads(request.content.decode('utf-8')))
            return httpx.Response(
                200,
                text='data: {"choices":[{"delta":{"content":"{}"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        with patch.object(llm_stream.httpx, 'AsyncClient', return_value=client):
            await llm_stream.stream_chat(
                self.config(), system='s', user='u', thinking=False,
            )

        self.assertEqual(seen['thinking'], {'type': 'disabled'})

    async def test_stream_error_is_not_silently_ignored(self):
        async def handle(request):
            return httpx.Response(200, text='data: {"error":{"message":"upstream disconnected"}}\n\n')
        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        with patch.object(llm_stream.httpx, 'AsyncClient', return_value=client):
            with self.assertRaises(llm.LLMError) as caught:
                await llm_stream.stream_chat(self.config(), system='s', user='u')
        self.assertIn('upstream disconnected', str(caught.exception))
