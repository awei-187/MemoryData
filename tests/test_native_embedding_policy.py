"""Run in the Hindsight environment against its installed native engine; no network."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SOURCE = Path('E:/AML/hindsight/hindsight-api-slim')
if SOURCE.is_dir():
    sys.path.insert(0, str(SOURCE))

import httpx
import openai
from openai import BadRequestError
try:
    from hindsight_api.engine.embeddings import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None
from methods.hindsight.api_config import configure_native_embeddings_request_policy, server_environment


@unittest.skipIf(OpenAIEmbeddings is None, 'Run this native-engine integration test in the hindsight environment')
class NativeEmbeddingPolicyTests(unittest.IsolatedAsyncioTestCase):
    def test_native_llm_reads_mapped_connect_timeout_and_preserves_request_budget(self):
        from hindsight_api.config import HindsightConfig
        from hindsight_api.engine import llm_transport
        mapped = server_environment({}, {'OPENAI_API_KEY': 'offline',
            'OPENAI_BASE_URL': 'https://offline.invalid/v1', 'OPENAI_API_CONNECT_TIMEOUT_SECONDS': '30',
            'HINDSIGHT_API_EMBEDDINGS_PROVIDER': 'openai', 'HINDSIGHT_API_RERANKER_PROVIDER': 'rrf'})
        with patch.dict(os.environ, mapped):
            native = HindsightConfig.from_env()
        with patch.object(llm_transport, 'get_config', return_value=native):
            timeout = llm_transport.build_sdk_timeout(120)
        self.assertEqual(timeout.connect, 30)
        self.assertEqual(timeout.read, 120)
        self.assertEqual(timeout.write, 120)

    async def test_native_client_retries_transient_handshake_without_retrying_policy_rejection(self):
        calls = []
        blocked = [False]
        original_accessor = OpenAIEmbeddings._loop_client
        real_sdk = openai.AsyncOpenAI

        async def handler(request):
            calls.append(request)
            self.assertEqual(request.extensions['timeout']['connect'], 30)
            if blocked[0]:
                return httpx.Response(400, json={'error': {'message': 'policy rejected', 'code': 'content_filter'}})
            if len(calls) <= 4:
                raise httpx.ConnectTimeout('simulated TLS delay', request=request)
            return httpx.Response(200, json={'object': 'list', 'model': 'offline',
                'data': [{'index': 0, 'object': 'embedding', 'embedding': [1.0, 0.0, 0.0]}],
                'usage': {'prompt_tokens': 1, 'total_tokens': 1}})

        class OfflineSDK(real_sdk):
            def __init__(self, **kwargs):
                super().__init__(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False), **kwargs)

        try:
            with patch.dict(os.environ, {'OPENAI_API_CONNECT_TIMEOUT_SECONDS': '30', 'OPENAI_API_MAX_RETRIES': '4'}), \
                 patch.object(openai, 'AsyncOpenAI', OfflineSDK):
                configure_native_embeddings_request_policy()
                embedder = OpenAIEmbeddings(api_key='offline', model='text-embedding-3-small',
                                             base_url='https://offline.invalid/v1', dimensions=3)
                await embedder.initialize()
                vectors = await embedder.encode(['unchanged input'])
                self.assertEqual(list(vectors[0]), [1.0, 0.0, 0.0])
                self.assertEqual(len(calls), 5)
                blocked[0] = True
                with self.assertRaises(Exception):
                    await embedder.encode(['unchanged blocked input'])
                self.assertEqual(len(calls), 6)
                await embedder._loop_client().close()
        finally:
            OpenAIEmbeddings._loop_client = original_accessor
            if hasattr(OpenAIEmbeddings, '_memorydata_request_policy_applied'):
                delattr(OpenAIEmbeddings, '_memorydata_request_policy_applied')
