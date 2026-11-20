import unittest
from app.tools.provider_network import provider_trust_env

class ProviderNetworkTests(unittest.TestCase):
    def test_lan_gateways_bypass_system_proxy(self):
        for url in ('http://10.0.0.5:18902/v1', 'http://127.0.0.1:8000',
                    'http://192.168.1.3/v1', 'http://172.16.2.3', 'http://localhost',
                    'http://[::1]', 'http://[::ffff:10.1.2.3]'):
            self.assertFalse(provider_trust_env(url), url)

    def test_public_providers_keep_proxy_support(self):
        for url in ('https://api.deepseek.com/v1', 'http://8.8.8.8',
                    'http://172.32.1.1', 'https://localhost.example.com'):
            self.assertTrue(provider_trust_env(url), url)
