"""Shared API configuration contracts; no external model calls."""
import json
import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import yaml

from utils.provider_utils import model_api_trust_env, resolve_model_config

ROOT = Path(__file__).resolve().parents[1]


class SharedModelTests(unittest.TestCase):
    def configs(self):
        return [(p.name, yaml.safe_load(p.read_text(encoding="utf-8")))
                for p in (ROOT / "config").glob("*.yaml")]

    def test_every_preset_uses_shared_remote_models_and_no_local_model_endpoint(self):
        configs = self.configs()
        self.assertEqual(len(configs), 25)  # + REMem, TSM
        for name, config in configs:
            with self.subTest(preset=name):
                resolved = resolve_model_config(config, {})
                self.assertEqual(resolved["model"], "gpt-4o-mini")
                self.assertEqual(config["model_env"], "OPENAI_MODEL")
                self.assertEqual(config["base_url_env"], "OPENAI_BASE_URL")
                self.assertFalse(config.get("base_url"))
                self.assertEqual(config["api_key_env"], "OPENAI_API_KEY")
                self.assertEqual(config["tokenizer_encoding"], "o200k_base")
                if config.get("embedding_model_env"):
                    self.assertEqual(resolved["embedding_model"], "text-embedding-3-small")
                    self.assertEqual(resolved["embedding_dim"], 1536)
                    self.assertEqual(config["embedding_base_url_env"], "OPENAI_BASE_URL")
                    self.assertFalse(config.get("embedding_base_url"))
                self.assertNotIn(":990", json.dumps(config))
                self.assertNotIn(":991", json.dumps(config))
                self.assertNotIn(":9009", json.dumps(config))
                self.assertNotIn(":9010", json.dumps(config))

    def test_environment_changes_reach_all_native_model_aliases_without_secrets(self):
        env = {"OPENAI_MODEL": "test-chat", "OPENAI_EMBEDDING_MODEL": "test-embedding",
               "OPENAI_EMBEDDING_DIMENSIONS": "1536", "OPENAI_API_TRUST_ENV": "false",
               "OPENAI_BASE_URL": "https://model.invalid/v1", "OPENAI_API_KEY": "never-serialize-this"}
        for name, config in self.configs():
            with self.subTest(preset=name):
                original = dict(config)
                resolved = resolve_model_config(config, env)
                self.assertEqual(config, original)
                self.assertEqual(resolved, resolve_model_config(resolved, env))
                self.assertEqual(resolved["model"], "test-chat")
                self.assertFalse(resolved["model_api_trust_env"])
                for alias in ("zep_local_small_model", "zep_local_answer_model", "hindsight_llm_model",
                              "remem_llm_model", "tsm_llm_model"):
                    if alias in resolved:
                        self.assertEqual(resolved[alias], "test-chat")
                for alias in ("text_embedding", "mem0_embedder_model", "embedder_model",
                              "a_mem_embedding_model", "lightmem_embedding_model", "hindsight_embeddings_model",
                              "remem_embedding_model", "tsm_embedding_model"):
                    if alias in resolved:
                        self.assertEqual(resolved[alias], "test-embedding")
                for alias in ("a_mem_base_url", "simplemem_base_url", "lightmem_base_url", "zep_local_answer_base_url"):
                    if alias in resolved:
                        self.assertEqual(resolved[alias], env["OPENAI_BASE_URL"])
                self.assertNotIn(env["OPENAI_API_KEY"], json.dumps(resolved))
                self.assertEqual(resolved.get("retrieve_num"), original.get("retrieve_num"))
                self.assertEqual(resolved["temperature"], original["temperature"])

    def test_legacy_configs_do_not_opt_in_to_model_overrides(self):
        legacy = {"model": "custom-model", "embedding_model": "custom-embedding", "base_url": "http://localhost/v1"}
        self.assertEqual(resolve_model_config(legacy, {"OPENAI_MODEL": "other"}), legacy)

    def test_ignored_method_endpoint_variables_are_reported(self):
        from utils import provider_utils

        config = {"model": "m", "model_env": "OPENAI_MODEL", "lightmem_base_url": "", "simplemem_base_url": ""}
        env = {"OPENAI_MODEL": "gpt-4o-mini", "OPENAI_BASE_URL": "http://shared/v1",
               "LIGHTMEM_BASE_URL": "http://other/v1"}
        with patch.object(provider_utils, "_WARNED_IGNORED_ENV", set()),                 self.assertLogs("utils.provider_utils", "WARNING") as logs:
            resolved = resolve_model_config(config, env)
            resolve_model_config(config, env)  # reported once
        self.assertEqual(resolved["lightmem_base_url"], "http://shared/v1")
        self.assertEqual(len(logs.output), 1)
        self.assertIn("LIGHTMEM_BASE_URL is set but ignored", logs.output[0])
        explicit = dict(config, lightmem_base_url="http://yaml/v1")
        with patch.object(provider_utils, "_WARNED_IGNORED_ENV", set()),                 patch.object(provider_utils.logger, "warning") as warning:
            self.assertEqual(resolve_model_config(explicit, env)["lightmem_base_url"], "http://yaml/v1")
        warning.assert_not_called()

    def test_invalid_dimensions_and_proxy_flag_fail_early(self):
        config = {"embedding_dim_env": "DIMS"}
        for value in ("0", "-1", "not-a-number"):
            with self.assertRaises(ValueError):
                resolve_model_config(config, {"DIMS": value})
        self.assertFalse(model_api_trust_env("false"))
        self.assertTrue(model_api_trust_env("true"))
        with self.assertRaises(ValueError):
            model_api_trust_env("maybe")

    def test_hindsight_server_uses_same_shared_environment(self):
        from methods.hindsight.api_config import server_environment
        config = yaml.safe_load((ROOT / "config/hybrid_hindsight.yaml").read_text())
        env = server_environment(config, {"OPENAI_MODEL": "test-chat", "OPENAI_EMBEDDING_MODEL": "test-embedding",
            "OPENAI_EMBEDDING_DIMENSIONS": "1536", "OPENAI_API_KEY": "test-secret", "OPENAI_BASE_URL": "https://example.invalid/v1"})
        self.assertEqual(env["HINDSIGHT_API_LLM_MODEL"], "test-chat")
        self.assertEqual(env["HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL"], "test-embedding")
        self.assertEqual(env["HINDSIGHT_API_EMBEDDINGS_OPENAI_DIMENSIONS"], "1536")
        self.assertNotIn('HINDSIGHT_API_LLM_CONNECT_TIMEOUT', env)
        shared = {"OPENAI_API_KEY": "offline", "OPENAI_BASE_URL": "https://offline.invalid/v1",
                  "OPENAI_API_CONNECT_TIMEOUT_SECONDS": "30"}
        self.assertEqual(float(server_environment(config, shared)['HINDSIGHT_API_LLM_CONNECT_TIMEOUT']), 30)
        explicit = dict(shared, HINDSIGHT_API_LLM_CONNECT_TIMEOUT='17')
        self.assertEqual(server_environment(config, explicit)['HINDSIGHT_API_LLM_CONNECT_TIMEOUT'], '17')

    def test_memagent_api_tokenizer_does_not_download_huggingface_model(self):
        from methods.memagent.recurrent import MemAgentRecurrentRunner, RecurrentConfig
        with patch("tiktoken.get_encoding", return_value=object()) as encoding, \
             patch("methods.memagent.recurrent.AutoTokenizer.from_pretrained") as hf:
            MemAgentRecurrentRunner(RecurrentConfig(model="gpt-4o-mini", base_url="https://example.invalid/v1",
                                                   tokenizer_encoding="o200k_base"))
        encoding.assert_called_once_with("o200k_base")
        hf.assert_not_called()

    def test_memorag_api_only_rejected_before_model_initialization(self):
        from utils.agent import AgentWrapper
        with self.assertRaisesRegex(ValueError, "requires a local memory model"):
            AgentWrapper({"agent_name": "Structure_rag_memo_rag", "api_only": True}, {}, "unused")

    def test_custom_remote_http_clients_follow_proxy_environment(self):
        import methods.embedding_rag.embedding_retriever as rag
        with patch.dict(os.environ, {"OPENAI_API_TRUST_ENV": "true", "OPENAI_API_KEY": "test-key"}), \
             patch.object(rag.httpx, "Client") as client, \
             patch.object(rag, "OpenAI"), patch.object(rag, "OpenAIEmbeddings"):
            rag._create_llm_client(provider="openai_compatible", base_url="https://example.invalid/v1")
            rag._create_embedding_model_client(model="text-embedding-3-small", provider="openai_compatible",
                                               base_url="https://example.invalid/v1")
        self.assertEqual(client.call_count, 2)
        self.assertTrue(all(call.kwargs["trust_env"] for call in client.call_args_list))

    def test_yaml_is_read_as_utf8_on_windows(self):
        from utils.initialization import _load_yaml_config
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.yaml"
            path.write_text("# 中文配置\nmodel: gpt-4o-mini\n", encoding="utf-8")
            self.assertEqual(_load_yaml_config(path), {"model": "gpt-4o-mini"})

    def test_mem0_native_clients_follow_proxy_and_explicit_api_settings(self):
        import utils.agent  # Registers the repository's vendored source paths.
        import mem0.llms.openai as llms
        import mem0.embeddings.openai as embeddings
        from mem0.configs.llms.base import BaseLlmConfig
        from mem0.configs.embeddings.base import BaseEmbedderConfig
        with patch.dict(os.environ, {"OPENAI_API_TRUST_ENV": "true", "OPENROUTER_API_KEY": "unrelated-key"}), \
             patch.object(llms.httpx, "Client") as client, \
             patch.object(llms, "OpenAI") as chat, patch.object(embeddings, "OpenAI") as embed:
            llm = llms.OpenAILLM(BaseLlmConfig(model="gpt-4o-mini", api_key="shared-key",
                                             openai_base_url="https://example.invalid/v1"))
            embeddings.OpenAIEmbedding(BaseEmbedderConfig(model="text-embedding-3-small", api_key="shared-key",
                                                         openai_base_url="https://example.invalid/v1", embedding_dims=1536))
        self.assertFalse(llm._use_openrouter)
        self.assertEqual(chat.call_args.kwargs["base_url"], "https://example.invalid/v1")
        self.assertEqual(embed.call_args.kwargs["base_url"], "https://example.invalid/v1")
        self.assertEqual(client.call_count, 2)
        self.assertTrue(all(call.kwargs["trust_env"] for call in client.call_args_list))


if __name__ == "__main__":
    unittest.main()
