"""Offline adapter and real MemoryData/HTTP pipeline checks; no model inference."""
import importlib
import hashlib
import shutil
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import httpx
import yaml

from methods.hindsight.hindsight_adapter import HindsightAdapter, STATE_FILE

ROOT = Path(__file__).resolve().parents[1]


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8", errors="ignore")


class Backend:
    """Small Hindsight wire-contract fixture, also exposed over real loopback HTTP."""
    def __init__(self):
        self.banks = {}
        self.calls = []
        self.recall_results = None
        self.stats = {"pending_operations": 0, "failed_operations": 0,
                      "pending_consolidation": 0, "operations_by_status": {}}
        self.retain_reply = None
        self.retain_status = 200
        self.profile_status = 200

    def respond(self, method, path, body):
        self.calls.append((method, path, body))
        if path == "/v1/chat/completions":
            return 200, {"id": "offline", "object": "chat.completion", "created": 0, "model": "offline-model",
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": "Tea"}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 20, "completion_tokens": 1, "total_tokens": 21}}
        parts = path.split("?")[0].split("/")
        bank = parts[4]
        suffix = "/".join(parts[5:])
        if method == "PUT" and not suffix:
            self.banks[bank] = {}
            return 200, {"bank_id": bank, "name": "test", "disposition": {}, "mission": ""}
        if method == "DELETE" and not suffix:
            if bank not in self.banks:
                return 404, {"detail": "bank missing"}
            del self.banks[bank]
            return 200, {"success": True}
        if method == "GET" and suffix == "profile" and self.profile_status != 200:
            return self.profile_status, {"detail": "profile endpoint unavailable"}
        if bank not in self.banks:
            return 404, {"detail": "bank missing"}
        if method == "GET" and suffix == "profile":
            return 200, {"bank_id": bank}
        if method == "GET" and suffix == "config":
            return 200, {"bank_id": bank, "config": {}, "overrides": {}}
        if method == "GET" and suffix == "stats":
            return 200, dict(self.stats)
        if method == "POST" and suffix == "memories":
            for item in body["items"]:
                self.banks[bank][item["document_id"]] = item
            return self.retain_status, self.retain_reply or {"success": True, "async": False, "items_count": 1, "bank_id": bank}
        if method == "POST" and suffix == "memories/recall":
            return 200, self.recall_results or {"results": [
                {"id": str(i), "text": item["content"], "document_id": document}
                for i, (document, item) in enumerate(self.banks[bank].items())]}
        raise AssertionError((method, path, body))

    def transport(self, request):
        status, body = self.respond(request.method, request.url.raw_path.decode(), json.loads(request.content) if request.content else None)
        return httpx.Response(status, json=body, request=request)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.backend = Backend()
        self.config = {"hindsight_base_url": "http://offline.invalid", "retrieve_num": 2}
        self.dataset = {"dataset": "LoCoMo", "sub_dataset": "locomo_qa"}
        self.environment = patch.dict(os.environ, {"HINDSIGHT_API_URL": "", "HINDSIGHT_API_KEY": "secret-not-on-disk"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def adapter(self, name="state", config=None):
        return HindsightAdapter(config or self.config, self.dataset, self.root / name,
                                transport=httpx.MockTransport(self.backend.transport))

    def ready(self):
        adapter = self.adapter()
        adapter.add_chunk("Alice drinks Tea", source_ids=["D1:1"], timestamp="2024-01-01T00:00:00Z")
        adapter.save()
        return adapter

    def test_retain_is_synchronous_and_manifest_has_no_secrets(self):
        adapter = self.ready()
        body = next(b for m, p, b in self.backend.calls if p.endswith("/memories") and m == "POST")
        self.assertIs(body["async"], False)
        self.assertEqual(body["items"][0]["update_mode"], "replace")
        self.assertEqual(body["items"][0]["timestamp"], "2024-01-01T00:00:00Z")
        self.assertEqual(adapter.retrieve("Drink?")[0]["source_ids"], ["D1:1"])
        self.assertNotIn("secret-not-on-disk", adapter.path.read_text())

    def test_missing_timestamp_is_unset_and_empty_history_does_not_write(self):
        a = self.adapter()
        a.add_chunk("  ")
        self.assertEqual(self.backend.calls, [])
        a.add_chunk("Timeless source")
        self.assertEqual(next(iter(self.backend.banks[a.state["bank_id"]].values()))["timestamp"], "unset")

    def test_top_k_native_budget_and_observation_provenance(self):
        a = self.ready()
        document = next(iter(a.state["documents"]))
        self.backend.recall_results = {"results": [{"id": "obs", "text": "Observation", "source_fact_ids": ["f"]},
                                                   {"id": "unknown", "text": "No provenance"}, {"id": "extra", "text": "Extra"}],
                                       "source_facts": {"f": {"document_id": document, "source_fact_ids": ["f"]}}}
        rows = a.retrieve("Drink?", query_timestamp="2024-02-01T00:00:00Z")
        self.assertEqual([r["source_ids"] for r in rows], [["D1:1"], []])
        request = self.backend.calls[-1][2]
        self.assertNotIn("top_k", request)
        self.assertEqual(request["max_tokens"], 4096)
        self.assertEqual(request["query_timestamp"], "2024-02-01T00:00:00Z")

    def test_completed_state_loads_without_retain(self):
        a = self.ready()
        before = len(self.backend.calls)
        resumed = self.adapter()
        resumed.load()
        self.assertEqual(a.state["bank_id"], resumed.state["bank_id"])
        self.assertEqual([m for m, _, _ in self.backend.calls[before:]], ["GET"])
        with self.assertRaisesRegex(RuntimeError, "append"):
            resumed.add_chunk("New history")

    def test_config_and_history_change_refuse_reuse(self):
        self.ready()
        for changes in [{"_hindsight_history_fingerprint": "changed"}, {"hindsight_bank_config": {"mission": "x"}},
                        {"hindsight_wait_for_consolidation": False}]:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "differs"):
                self.adapter(config={**self.config, **changes})

    def test_query_time_settings_reuse_the_same_bank(self):
        original = self.ready()
        state_before = original.path.read_bytes()
        changed = {**self.config, "retrieve_num": 1, "hindsight_recall_max_tokens": 1024,
                   "hindsight_recall_budget": "high", "hindsight_recall_types": ["world"],
                   "hindsight_settle_timeout_seconds": 36000}
        before = len(self.backend.calls)
        reused = self.adapter(config=changed)
        rows = reused.retrieve("Drink?")
        self.assertEqual(reused.state["bank_id"], original.state["bank_id"])
        self.assertEqual(len(rows), 1)
        self.assertNotIn("POST", [m for m, p, _ in self.backend.calls[before:] if p.endswith("/memories")])
        self.assertNotIn("DELETE", [m for m, _, _ in self.backend.calls[before:]])
        request = self.backend.calls[-1][2]
        self.assertEqual((request["max_tokens"], request["budget"], request["types"]), (1024, "high", ["world"]))
        self.assertEqual(original.path.read_bytes(), state_before)

    def _write_legacy_identity(self, adapter, recall, settle, location=False):
        """Rewrite a saved state's identity with the pre-identity_version-2 formula."""
        value = {'protocol': 'memorydata_hindsight_retain_recall_v1', 'base_url': 'http://offline.invalid',
                 'dataset': self.dataset, 'state_dir': str(adapter.path.parent.resolve()), 'bank_config': {},
                 'history': None, 'recall': recall, 'settle': settle}
        if location:
            from utils.repo_paths import portable
            value.update(dataset=portable(self.dataset), state_dir=portable(value['state_dir']), location='repo-relative')
        state = adapter.state | {'identity': hashlib.sha256(json.dumps(
            value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()}
        adapter.path.write_text(json.dumps(state))
        return adapter.path.read_bytes()

    def test_legacy_identity_with_other_query_settings_needs_declared_values(self):
        types_ = ['world', 'experience', 'observation']
        for location in (False, True):
            with self.subTest(repo_relative=location):
                self.backend = Backend()
                adapter = self.adapter(f'legacy-{location}')
                adapter.add_chunk('Alice drinks Tea')
                adapter.save()
                saved = self._write_legacy_identity(adapter, [2, 4096, 'mid', types_], [True, 7200.0], location)
                ablation = {**self.config, 'retrieve_num': 5}
                with self.assertRaisesRegex(ValueError, 'differs'):
                    self.adapter(f'legacy-{location}', config=ablation)
                declared = {**ablation, 'hindsight_legacy_identity_params': [
                    {'retrieve_num': 9}, {'retrieve_num': 2, 'hindsight_settle_timeout_seconds': 7200}]}
                reused = self.adapter(f'legacy-{location}', config=declared)
                reused.load()
                self.assertEqual(reused.state['bank_id'], adapter.state['bank_id'])
                self.assertEqual(reused.top_k, 5)
                self.assertEqual(adapter.path.read_bytes(), saved)
                with self.assertRaisesRegex(ValueError, 'differs'):
                    self.adapter(f'legacy-{location}', config={**declared, '_hindsight_history_fingerprint': 'other'})
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            self.adapter('legacy-False', config={**self.config, 'hindsight_legacy_identity_params': {'model': 'x'}})

    def test_expected_documents_match_acknowledged_retains(self):
        a = self.adapter()
        items = [("First", ["D1:1", "D1:1", ""]), ("   ", ["D1:2"]), ("Second", [])]
        expected = HindsightAdapter.expected_documents(items)
        self.assertFalse(a.awaits_consolidation_only(expected))
        a.add_chunk("First", source_ids=["D1:1", "D1:1", ""])
        self.assertFalse(a.awaits_consolidation_only(expected))
        for content, source_ids in items[1:]:
            a.add_chunk(content, source_ids=source_ids)
        self.assertEqual(a.state["documents"], expected)
        self.assertTrue(a.awaits_consolidation_only(expected))
        self.assertFalse(a.awaits_consolidation_only({}))
        a.save()
        self.assertFalse(a.awaits_consolidation_only(expected))

    def test_retired_profile_load_uses_config_and_still_checks_bank_exists(self):
        a = self.ready()
        self.backend.profile_status = 410
        before = len(self.backend.calls)
        self.adapter().load()
        calls = self.backend.calls[before:]
        self.assertEqual([(m, p.rsplit('/', 1)[-1]) for m, p, _ in calls],
                         [("GET", "profile"), ("GET", "config")])
        del self.backend.banks[a.state["bank_id"]]
        with self.assertRaisesRegex(RuntimeError, "no longer exists"):
            self.adapter().load()

    def test_profile_auth_error_does_not_fall_back(self):
        self.ready()
        self.backend.profile_status = 403
        before = len(self.backend.calls)
        with self.assertRaises(httpx.HTTPStatusError):
            self.adapter().load()
        self.assertEqual(len(self.backend.calls[before:]), 1)

    def test_partial_ingestion_rebuilds_only_owned_bank(self):
        self.backend.banks["unrelated"] = {"doc": "keep"}
        a = self.adapter()
        a.add_chunk("partial")
        resumed = self.adapter()
        with self.assertRaisesRegex(RuntimeError, "incomplete|complete"):
            resumed.retrieve("question")
        resumed.add_chunk("full")
        resumed.save()
        self.assertEqual([x["content"] for x in self.backend.banks[a.state["bank_id"]].values()], ["full"])
        self.assertIn("unrelated", self.backend.banks)

    def test_verified_snapshot_reuses_identity_but_still_checks_history_and_origin(self):
        original = self.ready()
        copied = self.root / 'fork'
        shutil.copytree(original.path.parent, copied)
        config = {**self.config, 'hindsight_snapshot_origin': {
            'state_dir': str(original.path.parent),
            'state_sha256': hashlib.sha256(original.path.read_bytes()).hexdigest()}}
        restored = self.adapter('fork', config=config)
        restored.load()
        self.assertEqual(restored.state['bank_id'], original.state['bank_id'])
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.adapter('fork', config={**config, '_hindsight_history_fingerprint': 'other-history'})
        original.path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'origin manifest changed'):
            self.adapter('fork', config=config)

    def _in_repo(self, repo):
        return patch('utils.repo_paths.REPO_ROOT', repo.resolve())

    def _repo_adapter(self, repo, dataset):
        return HindsightAdapter(self.config, dataset, repo / 'results' / 'agents' / 'exp_0',
                                transport=httpx.MockTransport(self.backend.transport))

    def test_moving_the_whole_repository_keeps_saved_state_reusable(self):
        old, new = self.root / 'old checkout', self.root / 'new'
        dataset = {**self.dataset, 'test_files': str(old / 'results' / 'input.json'), 'demo_files': ''}
        with self._in_repo(old):
            original = self._repo_adapter(old, dataset)
            original.add_chunk('Alice drinks Tea')
            original.save()
        shutil.move(str(old), str(new))
        moved = {**dataset, 'test_files': str(new / 'results' / 'input.json')}
        with self._in_repo(new):
            restored = self._repo_adapter(new, moved)
            restored.load()
            self.assertEqual(restored.state['bank_id'], original.state['bank_id'])
            # A state copied to another place inside the repository is still refused.
            shutil.copytree(new / 'results' / 'agents', new / 'results' / 'copied')
            with self.assertRaisesRegex(ValueError, 'differs'):
                HindsightAdapter(self.config, moved, new / 'results' / 'copied' / 'exp_0',
                                 transport=httpx.MockTransport(self.backend.transport))

    def test_absolute_path_legacy_state_needs_declared_former_root_and_is_not_rewritten(self):
        old, new = self.root / 'old checkout', self.root / 'new'
        dataset = {**self.dataset, 'test_files': str(old / 'results' / 'input.json')}
        with self._in_repo(old):
            adapter = self._repo_adapter(old, dataset)
            adapter.add_chunk('Alice drinks Tea')
            adapter.save()
        # Identity exactly as written before repository-relative fingerprints.
        legacy = adapter.state | {'identity': hashlib.sha256(json.dumps({
            'protocol': 'memorydata_hindsight_retain_recall_v1', 'base_url': 'http://offline.invalid',
            'dataset': dataset, 'state_dir': str(adapter.path.parent.resolve()), 'bank_config': {},
            'history': None, 'recall': [2, 4096, 'mid', ['world', 'experience', 'observation']],
            'settle': [True, 300.0]}, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()}
        adapter.path.write_text(json.dumps(legacy))
        with self._in_repo(old):
            self._repo_adapter(old, dataset).load()  # Old code's states stay valid where they are.
        shutil.move(str(old), str(new))
        state_file = new / 'results' / 'agents' / 'exp_0' / STATE_FILE
        before = state_file.read_bytes()
        moved = {**dataset, 'test_files': str(new / 'results' / 'input.json')}
        with self._in_repo(new):
            with self.assertRaisesRegex(ValueError, 'differs'):
                self._repo_adapter(new, moved)
            with patch.dict(os.environ, {'MEMORYDATA_LEGACY_REPO_ROOTS': str(old)}):
                restored = self._repo_adapter(new, moved)
                restored.load()
                with self.assertRaisesRegex(ValueError, 'differs'):
                    self._repo_adapter(new, {**moved, 'sub_dataset': 'other'})
        self.assertEqual(restored.state['bank_id'], adapter.state['bank_id'])
        self.assertEqual(state_file.read_bytes(), before)

    def test_verified_complete_ingestion_resumes_without_retain_or_delete(self):
        a = self.adapter()
        a.add_chunk('Complete history')
        self.backend.stats['total_documents'] = 1
        expected = dict(a.state['documents'])
        before = len(self.backend.calls)
        a.resume_completed_ingestion(expected)
        self.assertTrue(a.state['ready'])
        self.assertTrue(all(method == 'GET' for method, _, _ in self.backend.calls[before:]))

    def test_incomplete_resume_refuses_before_write(self):
        a = self.adapter()
        a.add_chunk('Partial history')
        before = len(self.backend.calls)
        with self.assertRaisesRegex(ValueError, 'complete history'):
            a.resume_completed_ingestion({'different-document': []})
        self.assertEqual(len(self.backend.calls), before)
        self.backend.stats['total_documents'] = 0
        with self.assertRaisesRegex(ValueError, 'native document count'):
            a.resume_completed_ingestion(dict(a.state['documents']))
        self.assertFalse(a.state['ready'])

    def test_contexts_have_distinct_banks(self):
        a, b = self.adapter("a"), self.adapter("b")
        a.add_chunk("Alpha")
        b.add_chunk("Beta")
        a.save()
        b.save()
        self.assertNotEqual(a.state["bank_id"], b.state["bank_id"])
        self.assertEqual([r["text"] for r in b.retrieve("all")], ["Beta"])

    def test_missing_remote_bank_fails_and_unsafe_reset_is_rejected(self):
        a = self.ready()
        del self.backend.banks[a.state["bank_id"]]
        with self.assertRaisesRegex(RuntimeError, "no longer exists"):
            self.adapter().load()
        state = dict(a.state, bank_id="unrelated")
        a.path.write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "not owned"):
            HindsightAdapter.reset_saved(self.config, a.path.parent)

    def test_retain_errors_never_mark_ready_or_retry_write(self):
        for response in [{"success": False}, {"success": True, "async": True, "items_count": 1}]:
            self.backend.retain_reply = response
            a = self.adapter(str(len(self.backend.calls)))
            with self.assertRaisesRegex(RuntimeError, "synchronous"):
                a.add_chunk("source")
            self.assertFalse(a.state["ready"])
        self.backend.retain_status = 500
        count = len(self.backend.calls)
        with self.assertRaises(httpx.HTTPStatusError):
            self.adapter("failure").add_chunk("source")
        self.assertEqual(sum(m == "POST" for m, _, _ in self.backend.calls[count:]), 1)

    def test_failed_or_timed_out_consolidation_does_not_mark_ready(self):
        self.backend.stats["failed_operations"] = 1
        a = self.adapter()
        a.add_chunk("source")
        with self.assertRaisesRegex(RuntimeError, "failed background"):
            a.save()
        self.assertFalse(a.state["ready"])
        persisted = json.loads(a.path.read_text())
        self.assertFalse(persisted['ready'])
        self.assertEqual(persisted['last_consolidation_stats']['failed_operations'], 1)
        self.backend.stats.update(failed_operations=0, pending_consolidation=1)
        with patch("methods.hindsight.hindsight_adapter.time.monotonic", side_effect=[0, 301]):
            with self.assertRaises(TimeoutError):
                a.save()
        self.assertFalse(a.state["ready"])
        self.assertEqual(json.loads(a.path.read_text())['last_consolidation_stats']['pending_consolidation'], 1)

    def test_malformed_recall_is_a_failure(self):
        a = self.ready()
        self.backend.recall_results = {"not_results": []}
        with self.assertRaisesRegex(ValueError, "results list"):
            a.retrieve("question")

    def test_memoryagentbench_date_is_sent_as_iso_without_invented_timezone(self):
        a = self.ready()
        source_date = '2023/05/27 (Sat) 22:54'
        a.retrieve('Question?', query_timestamp=source_date)
        body = self.backend.calls[-1][2]
        self.assertEqual(body['query_timestamp'], '2023-05-27T22:54:00')
        self.assertEqual(source_date, '2023/05/27 (Sat) 22:54')
        a.retrieve('Question?', query_timestamp='2023-05-27T22:54:00+08:00')
        self.assertEqual(self.backend.calls[-1][2]['query_timestamp'], '2023-05-27T22:54:00+08:00')

    def test_invalid_question_date_is_dropped_with_a_warning(self):
        # An unknown date format must not fail the question: recall runs without
        # a time anchor and the value is reported (strict parsing stays available
        # as normalize_query_timestamp).
        a = self.ready()
        with self.assertLogs('methods.hindsight.hindsight_adapter', 'WARNING') as logs:
            a.retrieve('Question?', query_timestamp='2023/99/27 (Sat) 22:54')
        self.assertIn('query_timestamp', logs.output[0])
        self.assertNotIn('query_timestamp', self.backend.calls[-1][2])
        from methods.hindsight.hindsight_adapter import normalize_query_timestamp
        with self.assertRaisesRegex(ValueError, 'query_timestamp'):
            normalize_query_timestamp('2023/99/27 (Sat) 22:54')

    def test_pending_consolidation_waits_before_marking_ready(self):
        a = self.adapter()
        a.add_chunk("source")
        pending = dict(self.backend.stats, pending_consolidation=1)
        with patch.object(a, "_request", side_effect=[pending, dict(self.backend.stats)]) as request, \
             patch("methods.hindsight.hindsight_adapter.time.sleep") as sleep:
            a.save()
        self.assertEqual(request.call_count, 2)
        sleep.assert_called_once()
        self.assertTrue(a.state["ready"])

    def test_invalid_configuration_and_endpoint_reset_fail_before_network(self):
        # A URL carrying (dummy) credentials must be rejected. Assembled at run
        # time so public-repository secret scanners do not flag a fake login.
        credential_url = "http://" + ":".join(["example-user", "example-secret"]) + "@localhost"
        for changes in [{"retrieve_num": 0}, {"hindsight_recall_budget": "invalid"},
                        {"hindsight_recall_types": ["unknown"]}, {"hindsight_base_url": credential_url}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.adapter(config={**self.config, **changes})
        a = self.ready()
        with self.assertRaisesRegex(ValueError, "endpoint changed"):
            HindsightAdapter.reset_saved({**self.config, "hindsight_base_url": "http://another.invalid"}, a.path.parent)

    def test_requests_match_local_upstream_generated_models(self):
        source = Path(os.environ.get("HINDSIGHT_SOURCE_PATH", "E:/AML/hindsight")) / "hindsight-clients/python/hindsight_client_api"
        if not source.exists():
            self.skipTest("Set HINDSIGHT_SOURCE_PATH to verify the local upstream SDK contract")
        # Load only generated request models; do not import the client runtime or server.
        namespaces = {}
        for name, directory in [("hindsight_client_api", source), ("hindsight_client_api.models", source / "models")]:
            module = types.ModuleType(name)
            module.__path__ = [str(directory)]
            namespaces[name] = module
        with patch.dict(sys.modules, namespaces):
            retain = importlib.import_module("hindsight_client_api.models.retain_request").RetainRequest
            recall = importlib.import_module("hindsight_client_api.models.recall_request").RecallRequest
            bank = importlib.import_module("hindsight_client_api.models.create_bank_request").CreateBankRequest
            a = self.ready()
            a.retrieve("question")
            for method, path, body in self.backend.calls:
                if method == "PUT": bank.from_dict(body)
                elif path.endswith("/memories/recall"): recall.from_dict(body)
                elif method == "POST":
                    parsed = retain.from_dict(body)
                    self.assertEqual(parsed.to_dict()["items"][0]["content"], body["items"][0]["content"])


class MainPipelineTests(unittest.TestCase):
    def test_main_real_http_resume_force_and_legacy_templates(self):
        from utils.agent import AgentWrapper
        from utils.initialization import initialize_and_memorize_agent, _apply_chunk_size_ablation
        from benchmark.memoryagentbench.prompts.benchmark_templates import get_template
        from utils.artifact_paths import generate_agent_save_folder_path
        import main

        backend = Backend()
        class Handler(BaseHTTPRequestHandler):
            def do_request(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                status, value = backend.respond(self.command, self.path, json.loads(body) if body else None)
                payload = json.dumps(value).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            do_GET = do_POST = do_PUT = do_DELETE = do_request
            def log_message(self, *args): pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
            endpoint = f"http://127.0.0.1:{server.server_port}"
            config = yaml.safe_load((ROOT / "config/hybrid_hindsight.yaml").read_text())
            config.update(base_url=endpoint + "/v1", hindsight_base_url=endpoint, model="offline-model", model_api_trust_env=False)
            dataset = yaml.safe_load((ROOT / "benchmark/locomo/config/Locomo_qa_4cat_600_dist.yaml").read_text())
            dataset.update(test_files=str(root / "data.json"), max_test_samples=2, chunk_size=4096)
            samples = [{"sample_id": f"sample-{i}", "conversation": {"speaker_a": "Alice", "speaker_b": "Bob",
                        "session_1_date_time": "1 January 2024", "session_1": [
                        {"speaker": "Alice", "dia_id": "D1:1", "text": "I drink Tea. " + "History detail. " * 160}]},
                        "qa": [{"question": "What drink?", "answer": "Tea", "category": 4, "evidence": ["D1:1"]},
                               {"question": "What secret?", "answer": "SECRET_GOLD_ONLY", "category": 4, "evidence": []}]} for i in range(2)]
            (root / "data.json").write_text(json.dumps(samples))
            for name, value in [("agent.yaml", config), ("dataset.yaml", dataset)]:
                (root / name).write_text(yaml.safe_dump(value))

            def run(*flags):
                args = ["main.py", "--agent_config", str(root / "agent.yaml"), "--dataset_config", str(root / "dataset.yaml"),
                        "--artifact_root", str(root / "runs"), *flags]
                with patch.object(sys, "argv", args): main.main()

            with patch.dict(os.environ, {"OPENAI_API_KEY": "offline", "HINDSIGHT_API_URL": ""}), \
                 patch.object(AgentWrapper, "_initialize_tokenizer", return_value=CharacterTokenizer()):
                run("--max_test_queries_ablation", "1")
                first_writes = sum(m == "POST" and p.endswith("/memories") for m, p, _ in backend.calls)
                run()
                writes = sum(m == "POST" and p.endswith("/memories") for m, p, _ in backend.calls)
                self.assertEqual(writes, first_writes * 2)
                banks_before = set(backend.banks)
                calls_before = len(backend.calls)
                run()
                self.assertEqual(len(backend.calls), calls_before, "Completed queries must not call APIs again")
                results = list((root / "runs" / "outputs").rglob("*_results.json"))
                self.assertEqual(len(results), 1)
                payload = json.loads(results[0].read_text())
                self.assertTrue({"agent_config", "dataset_config", "data", "metrics", "time_cost", "averaged_metrics"} <= payload.keys())
                self.assertEqual(len(payload["data"]), 4)
                self.assertTrue(all(row["output"] == "Tea" for row in payload["data"]))
                self.assertEqual(payload["data"][0]["retrieved_source_id_groups"], [["D1:1"]])
                # Restore validation rejects history changes instead of mixing memories.
                state_dir = next((root / "runs" / "agents").rglob(STATE_FILE)).parent
                with self.assertRaisesRegex(ValueError, "differs"):
                    initialize_and_memorize_agent(config, dataset, str(state_dir), ["changed history"], 0, 2)
                backend.banks["foreign-bank"] = {}
                run("--force")
                self.assertFalse(banks_before & set(backend.banks))
                self.assertIn("foreign-bank", backend.banks)
                self.assertEqual(len(backend.banks), 3)
                # A legacy baseline still uses the original CLI, loader and results flow.
                legacy = dict(config, agent_name="Long_context_agent_offline", output_dir=str(root / "legacy"))
                legacy.pop("agent_chunk_size")
                legacy.pop("agent_variant_tag")
                (root / "agent.yaml").write_text(yaml.safe_dump(legacy))
                before = len(backend.calls)
                run()
                self.assertEqual(len(list((root / "legacy").rglob("*_results.json"))), 1)
                self.assertTrue(all(p.endswith("/chat/completions") for _, p, _ in backend.calls[before:]))

                # Existing LongMemEval debug plumbing recognizes the new method.
                # The real sub-dataset name contains "*", which Windows paths reject.
                lme = dict(dataset, dataset="accurate_retrieval", sub_dataset="longmemeval_s*")
                debug_config = dict(config, _backfill_longmemeval_recall_debug=True,
                                    artifact_root=str(root / "runs"))
                debug_agent = AgentWrapper(debug_config, lme, str(root / "debug_state"))
                debug_agent._maybe_save_longmemeval_backfill_debug(query_id=0, context_id=0,
                    retrieved_paragraphs=["history"], query_text="question", response_text="Tea")
                debug_files = list((root / "runs").rglob("query_0_context_0.json"))
                self.assertTrue(debug_files)
                self.assertIn("longmemeval_s%2A", debug_files[0].parts)
                self.assertEqual(main._resolve_backfill_longmemeval_method(config["agent_name"]), ("hindsight", "Hindsight"))

            for method, path, body in backend.calls:
                if method == "POST" and path.endswith("/memories"):
                    self.assertNotIn("SECRET_GOLD_ONLY", json.dumps(body))
                    self.assertNotIn("LOCOMO_META", body["items"][0]["content"])
                if path.endswith("/chat/completions"):
                    self.assertNotIn("SECRET_GOLD_ONLY", json.dumps(body))
            for name in ["eventqa_qa", "locomo_qa", "longbench_mcq", "membench_simple"]:
                for template in ["system", "memorize", "query", "fact_extraction", "memory_answer"]:
                    self.assertTrue(get_template(name, template, config["agent_name"]))
            path = generate_agent_save_folder_path(config, dataset, 0, str(root / "runs"))
            self.assertIn("agents", Path(path).parts)
            self.assertIn("chunk4096", path)
            _apply_chunk_size_ablation(types.SimpleNamespace(chunk_size_ablation=2048), config, dataset)
            self.assertEqual((config["agent_chunk_size"], dataset["chunk_size"]), (2048, 2048))


def serve_over_loopback(testcase, backend):
    class Handler(BaseHTTPRequestHandler):
        def do_request(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            status, value = backend.respond(self.command, self.path, json.loads(body) if body else None)
            payload = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        do_GET = do_POST = do_PUT = do_DELETE = do_request
        def log_message(self, *args): pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    testcase.addCleanup(server.server_close)
    testcase.addCleanup(server.shutdown)
    return f"http://127.0.0.1:{server.server_port}"


class SettleTimeoutRecoveryTests(unittest.TestCase):
    def test_rerun_after_settle_timeout_waits_without_deleting_or_retaining(self):
        from utils.agent import AgentWrapper
        from utils.initialization import initialize_and_memorize_agent
        from utils.locomo_utils import build_locomo_storage_text

        backend = Backend()
        endpoint = serve_over_loopback(self, backend)
        config = yaml.safe_load((ROOT / "config/hybrid_hindsight.yaml").read_text())
        config.update(base_url=endpoint + "/v1", hindsight_base_url=endpoint, model="offline-model",
                      model_api_trust_env=False, hindsight_settle_timeout_seconds=0.01)
        dataset = yaml.safe_load((ROOT / "benchmark/locomo/config/Locomo_qa_4cat_600_dist.yaml").read_text())
        texts = ["Alice: I drink Tea.", "Bob: I drink coffee."]
        chunks = [{"text": text, "storage_text": build_locomo_storage_text(text, f"c{i}", [f"D1:{i + 1}"])}
                  for i, text in enumerate(texts)] + ["   "]
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {"OPENAI_API_KEY": "offline", "HINDSIGHT_API_URL": ""}), \
             patch.object(AgentWrapper, "_initialize_tokenizer", return_value=CharacterTokenizer()):
            folder = str(Path(directory) / "exp_0")
            backend.stats["pending_consolidation"] = 1
            with self.assertRaisesRegex(TimeoutError, "rerun the same command"):
                initialize_and_memorize_agent(config, dataset, folder, chunks, 0, 1)
            state = json.loads((Path(folder) / STATE_FILE).read_text())
            self.assertFalse(state["ready"])
            self.assertEqual(list(state["documents"].values()), [["D1:1"], ["D1:2"]])

            # Still settling: a rerun waits again and keeps the acknowledged bank.
            calls = len(backend.calls)
            backend.stats["total_documents"] = 2
            with self.assertRaises(TimeoutError):
                initialize_and_memorize_agent(config, dataset, folder, chunks, 0, 1)
            self.assertTrue(all(m == "GET" for m, _, _ in backend.calls[calls:]))

            # A larger wait budget is accepted, and readiness completes without new writes.
            backend.stats["pending_consolidation"] = 0
            calls = len(backend.calls)
            agent, _ = initialize_and_memorize_agent(
                dict(config, hindsight_settle_timeout_seconds=600), dataset, folder, chunks, 0, 1)
            self.assertTrue(agent.hindsight.state["ready"])
            self.assertEqual(agent.hindsight.state["bank_id"], state["bank_id"])
            self.assertTrue(all(m == "GET" for m, _, _ in backend.calls[calls:]))
            self.assertEqual(len(backend.banks[state["bank_id"]]), 2)
            self.assertEqual(agent.send_message("What drink?", memorizing=False, query_id=0, context_id=0)["output"], "Tea")

    def test_partial_history_is_still_rebuilt_after_restart(self):
        from utils.agent import AgentWrapper
        from utils.initialization import initialize_and_memorize_agent

        backend = Backend()
        endpoint = serve_over_loopback(self, backend)
        config = yaml.safe_load((ROOT / "config/hybrid_hindsight.yaml").read_text())
        config.update(base_url=endpoint + "/v1", hindsight_base_url=endpoint, model="offline-model",
                      model_api_trust_env=False)
        dataset = yaml.safe_load((ROOT / "benchmark/locomo/config/Locomo_qa_4cat_600_dist.yaml").read_text())
        chunks = ["Alice: I drink Tea.", "Bob: I drink coffee."]
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {"OPENAI_API_KEY": "offline", "HINDSIGHT_API_URL": ""}), \
             patch.object(AgentWrapper, "_initialize_tokenizer", return_value=CharacterTokenizer()):
            folder = str(Path(directory) / "exp_0")
            original = HindsightAdapter.add_chunk
            def fail_second(adapter, content, **kwargs):
                if adapter.chunk_index == 1:
                    raise RuntimeError("interrupted")
                return original(adapter, content, **kwargs)
            with patch.object(HindsightAdapter, "add_chunk", fail_second), self.assertRaisesRegex(RuntimeError, "interrupted"):
                initialize_and_memorize_agent(config, dataset, folder, chunks, 0, 1)
            calls = len(backend.calls)
            agent, _ = initialize_and_memorize_agent(config, dataset, folder, chunks, 0, 1)
            later = backend.calls[calls:]
            self.assertIn("DELETE", [m for m, _, _ in later])
            self.assertEqual(sum(m == "POST" and p.endswith("/memories") for m, p, _ in later), 2)
            self.assertTrue(agent.hindsight.state["ready"])


class APIConfigurationTests(unittest.TestCase):
    def test_copied_openai_settings_feed_native_hindsight_without_local_models(self):
        from methods.hindsight.api_config import server_environment
        env = server_environment({}, {"OPENAI_API_KEY": "test-secret", "OPENAI_BASE_URL": "https://example.invalid/v1/"})
        self.assertEqual(env["HINDSIGHT_API_LLM_MODEL"], "gpt-4o-mini")
        self.assertEqual(env["HINDSIGHT_API_LLM_PROVIDER"], "openai")
        self.assertEqual(env["HINDSIGHT_API_LLM_API_KEY"], "test-secret")
        self.assertEqual(env["HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY"], "test-secret")
        self.assertEqual(env["HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL"], "https://example.invalid/v1")
        self.assertEqual(env["HINDSIGHT_API_EMBEDDINGS_PROVIDER"], "openai")
        self.assertEqual(env["HINDSIGHT_API_RERANKER_PROVIDER"], "rrf")

    def test_missing_api_credentials_fail_before_server_start(self):
        from methods.hindsight.api_config import server_environment
        for env in [{}, {"OPENAI_BASE_URL": "https://example.invalid/v1"}]:
            with self.assertRaisesRegex(ValueError, "OPENAI_BASE_URL and OPENAI_API_KEY"):
                server_environment({}, env)

    def test_answer_model_environment_override_does_not_store_credentials(self):
        from methods.hindsight.api_config import resolve_answer_config
        config = {"model": "gpt-4o-mini", "model_env": "HINDSIGHT_ANSWER_MODEL", "api_key_env": "OPENAI_API_KEY"}
        with patch.dict(os.environ, {"HINDSIGHT_ANSWER_MODEL": "test-api-model", "OPENAI_API_KEY": "secret"}):
            result = resolve_answer_config(config)
        self.assertEqual(result["model"], "test-api-model")
        self.assertNotIn("secret", json.dumps(result))
        self.assertEqual(config["model"], "gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
