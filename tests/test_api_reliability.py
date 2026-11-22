import os
import unittest
from unittest.mock import patch

import httpx
from openai import BadRequestError

from utils.provider_utils import model_api_request_options


class RequestReliabilityTests(unittest.TestCase):
    def test_opt_in_validation_keeps_default_profile_unchanged(self):
        self.assertEqual(model_api_request_options({}), {})
        self.assertEqual(model_api_request_options({'OPENAI_API_MAX_RETRIES': '10'}), {'max_retries': 10})
        for values in ({'OPENAI_API_CONNECT_TIMEOUT_SECONDS': '-1'},
                       {'OPENAI_API_CONNECT_TIMEOUT_SECONDS': 'nan'}, {'OPENAI_API_MAX_RETRIES': '11'}):
            with self.assertRaises(ValueError):
                model_api_request_options(values)

    def test_real_mem0_sdk_recovers_connect_timeout_but_does_not_retry_content_filter(self):
        import utils.agent  # Register vendored source paths.
        import mem0.embeddings.openai as embedding_module
        from mem0.configs.embeddings.base import BaseEmbedderConfig
        real_client = httpx.Client
        calls = []
        policy_error = [False]

        def handler(request):
            calls.append(request)
            self.assertEqual(request.extensions['timeout']['connect'], 30)
            if policy_error[0]:
                return httpx.Response(400, json={'error': {'message': 'policy rejected', 'code': 'content_filter'}})
            if len(calls) <= 3:
                raise httpx.ConnectTimeout('simulated TLS handshake timeout', request=request)
            return httpx.Response(200, json={'object': 'list', 'model': 'offline',
                'data': [{'object': 'embedding', 'index': 0, 'embedding': [1.0, 0.0, 0.0]}],
                'usage': {'prompt_tokens': 1, 'total_tokens': 1}})

        class FixtureClient(real_client):
            def __init__(self, **kwargs):
                super().__init__(transport=httpx.MockTransport(handler), **kwargs)

        with patch.dict(os.environ, {'OPENAI_API_CONNECT_TIMEOUT_SECONDS': '30', 'OPENAI_API_MAX_RETRIES': '4'}), \
             patch.object(embedding_module.httpx, 'Client', FixtureClient), \
             patch('openai._base_client.time.sleep'):
            embedder = embedding_module.OpenAIEmbedding(BaseEmbedderConfig(model='offline', embedding_dims=3,
                api_key='offline-test-key', openai_base_url='https://offline.invalid/v1'))
            try:
                self.assertEqual(embedder.embed('frozen text'), [1.0, 0.0, 0.0])
                self.assertEqual(len(calls), 4)
                policy_error[0] = True
                with self.assertRaises(BadRequestError):
                    embedder.embed('unchanged policy-blocked text')
                self.assertEqual(len(calls), 5)
            finally:
                embedder.client.close()
