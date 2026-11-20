"""供应商按钮必须报告上游事实，而不是制造一个本地 400/502。"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from app import model_scan
from app.api import settings


class _Client:
    def __init__(self, response: httpx.Response):
        self.response = response
        self.payload = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def post(self, _url, *, json, headers):
        self.url = _url
        self.headers = headers
        self.payload = json
        return self.response


class ProviderProbeTests(unittest.IsolatedAsyncioTestCase):
    async def test_saved_key_does_not_override_visible_draft(self):
        profile = SimpleNamespace(id='saved', base_url='https://old.test/v1',
                                  api_key='saved-secret', model='old-model', timeout=60)
        response = httpx.Response(200, json={'choices':[{'message':{'content':'OK'}}]})
        client = _Client(response)
        with patch.object(settings.models_config, 'load', return_value=SimpleNamespace(providers=[profile])), \
             patch.object(settings.httpx, 'AsyncClient', return_value=client):
            result = await settings.test_provider(settings.TestProviderPayload(
                provider_id='saved', base_url='https://new.test/v1', model='new-model'))
        self.assertTrue(result['ok'])
        self.assertEqual(client.payload['model'], 'new-model')
        self.assertEqual(client.url, 'https://new.test/v1/chat/completions')
        self.assertEqual(client.headers['Authorization'], 'Bearer saved-secret')

    async def test_upstream_502_is_data_not_a_local_http_exception(self):
        response = httpx.Response(
            502,
            text="gateway is restarting",
            request=httpx.Request("POST", "http://provider/v1/chat/completions"),
        )
        client = _Client(response)
        payload = settings.TestProviderPayload(
            base_url="http://provider/v1", api_key="secret", model="model-a", timeout=9
        )
        with patch.object(settings.httpx, "AsyncClient", return_value=client):
            result = await settings.test_provider(payload)

        self.assertFalse(result["ok"])
        self.assertEqual(result["source"], "upstream")
        self.assertEqual(result["upstream_status"], 502)
        self.assertIn("上游模型服务返回 HTTP 502", result["error"])

    async def test_connection_probe_sends_only_openai_compatible_fields(self):
        response = httpx.Response(
            200,
            json={"model": "model-a", "choices": [{"message": {"content": "OK"}}]},
            request=httpx.Request("POST", "http://provider/v1/chat/completions"),
        )
        client = _Client(response)
        payload = settings.TestProviderPayload(
            base_url="http://provider/v1", api_key="secret", model="model-a"
        )
        with patch.object(settings.httpx, "AsyncClient", return_value=client):
            result = await settings.test_provider(payload)

        self.assertTrue(result["ok"])
        self.assertEqual(result["reply"], "OK")
        self.assertNotIn("thinking", client.payload)
        self.assertNotIn("response_format", client.payload)


class ModelScanTests(unittest.TestCase):
    def test_scan_preserves_the_upstream_status_and_body(self):
        response = httpx.Response(
            400,
            text='{"error":{"message":"unsupported route"}}',
            request=httpx.Request("GET", "http://provider/v1/models"),
        )
        with patch.object(model_scan.httpx, "get", return_value=response):
            catalogue = model_scan.scan("http://provider/v1", "secret")

        self.assertEqual(catalogue.error_source, "upstream")
        self.assertEqual(catalogue.upstream_status, 400)
        self.assertIn("unsupported route", catalogue.upstream_body)
        self.assertIn("上游模型服务返回 HTTP 400", catalogue.error)


if __name__ == "__main__":
    unittest.main()
