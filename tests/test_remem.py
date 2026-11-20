"""Offline REMem integration checks: adapter, worker glue, real worker subprocess, main pipeline.

No model, embedding service or REMem environment is used. The native worker is
replaced by tests/fixtures/fake_memory_worker.py (same protocol), and REMem's
own objects by small fakes.
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import yaml

from methods.remem import remem_adapter
from methods.remem.remem_adapter import RememAdapter, native_config
from methods.worker_bridge import WorkerError, WorkerProcess, close_all_workers, worker_environment

ROOT = Path(__file__).resolve().parents[1]
FAKE_WORKER = ROOT / "tests" / "fixtures" / "fake_memory_worker.py"
SECRET = "sk-secret-not-on-disk"


def load_worker_module():
    spec = importlib.util.spec_from_file_location("remem_worker_under_test", ROOT / "methods/remem/remem_worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeWorker:
    """In-process stand-in recording calls; replies like remem_worker.RememService."""

    def __init__(self, log):
        self.log = log
        self.docs = []
        self.closed = False

    def call(self, op, timeout=None, **args):
        self.log.append((op, args))
        if op == "open":
            self.docs = args["docs"]
            self.working_dir = Path(args["working_dir"])
            return {"opened": True}
        if op == "index":
            self.working_dir.mkdir(parents=True, exist_ok=True)
            (self.working_dir / "graph.pkl").write_bytes(b"fake")
            return {"graph_nodes": len(self.docs)}
        if op == "query":
            return {"answer": "native Tea", "passages": [
                {"node": "gists-1", "type": "gists", "text": "Alice drinks tea"},
                {"node": "entity-1", "type": "entity", "text": "Alice"},
                {"node": "verbatim-1", "type": "verbatim", "text": "  "}],
                "source_docs": [[d["doc_id"] for d in self.docs], [], [self.docs[0]["doc_id"]]]}
        raise AssertionError(op)

    def close(self, kill=False):
        self.closed = True


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(str(text).encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8", errors="ignore")


class AdapterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.calls = []
        self.config = {"model": "gpt-4o-mini", "embedding_model": "text-embedding-3-small", "retrieve_num": 5,
                       "_memory_history_fingerprint": "history-a"}
        self.dataset = {"dataset": "LoCoMo", "sub_dataset": "locomo_qa", "max_test_queries": 3}

    def adapter(self, config=None, name="state"):
        return RememAdapter(config or self.config, self.dataset, self.root / name,
                            worker_factory=lambda: FakeWorker(self.calls))

    def ready(self):
        adapter = self.adapter()
        adapter.add_chunk("Alice drinks tea", source_ids=["D1:1", "D1:1", ""], timestamp="2024-01-01T00:00:00")
        adapter.add_chunk("   ")
        adapter.add_chunk("Bob drinks coffee", source_ids=["D1:2"])
        adapter.save()
        return adapter

    def test_buffered_history_is_indexed_once_and_manifest_has_no_texts(self):
        adapter = self.ready()
        self.assertEqual([op for op, _ in self.calls], ["open", "index"])
        docs = self.calls[0][1]["docs"]
        self.assertEqual([d["text"] for d in docs], ["Alice drinks tea", "Bob drinks coffee"])
        self.assertEqual(docs[0]["date"], "2024-01-01T00:00:00")
        self.assertIsNone(docs[1]["date"])
        self.assertEqual(self.calls[0][1]["config"]["extract_method"], "episodic_gist")
        manifest = adapter.path.read_text(encoding="utf-8")
        self.assertNotIn("Alice drinks tea", manifest)
        self.assertTrue(adapter.state["ready"])
        self.assertEqual(adapter.state["documents"][0]["source_ids"], ["D1:1"])

    def test_retrieve_maps_visited_nodes_to_source_ids_and_drops_empty_text(self):
        adapter = self.ready()
        answer, passages = adapter.retrieve("What does Alice drink?", query_timestamp="2024/01/02 (Tue) 10:00")
        self.assertEqual(answer, "native Tea")
        self.assertEqual([p["source_ids"] for p in passages], [["D1:1", "D1:2"], []])
        self.assertEqual(self.calls[-1][1]["question_date"], "2024-01-02T10:00:00")
        self.assertEqual([op for op, _ in self.calls], ["open", "index", "query"])

    def test_saved_index_is_reopened_without_new_history(self):
        self.ready()
        adapter = self.adapter()
        adapter.load()
        adapter.retrieve("Drink?")
        ops = [op for op, _ in self.calls]
        self.assertEqual(ops, ["open", "index", "open", "index", "query"])
        self.assertEqual(len(self.calls[2][1]["docs"]), 2)
        with self.assertRaisesRegex(RuntimeError, "completed REMem index"):
            adapter.add_chunk("more history")

    def test_query_time_settings_reuse_but_index_settings_and_history_refuse(self):
        self.ready()
        self.adapter(dict(self.config, remem_native_config={"agent_max_steps": 5, "qa_top_k": 3})).load()
        for change in ({"remem_native_config": {"synonymy_edge_sim_threshold": 0.7}},
                       {"_memory_history_fingerprint": "history-b"},
                       {"embedding_model": "text-embedding-3-large"}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "differs"):
                self.adapter(dict(self.config, **change))

    def test_unfinished_ingestion_is_discarded_before_rebuilding(self):
        first = self.adapter()
        first.add_chunk("stale chunk", source_ids=["D9:9"])
        (first.native_dir / "leftover").mkdir(parents=True)
        second = self.adapter()
        self.assertFalse(second.state["ready"])
        second.add_chunk("fresh chunk")
        self.assertFalse(second.native_dir.exists())
        self.assertEqual([d["text"] for d in second.documents()], ["fresh chunk"])
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            second.load()

    def test_failed_index_never_marks_ready(self):
        adapter = self.adapter(dict(self.config))

        class Failing(FakeWorker):
            def call(self, op, timeout=None, **args):
                if op == "index":
                    raise WorkerError("native failure")
                return super().call(op, **args)

        adapter.worker_factory = lambda: Failing(self.calls)
        adapter.add_chunk("history")
        with self.assertRaises(WorkerError):
            adapter.save()
        self.assertFalse(RememAdapter.read_state(adapter.state_dir)["ready"])
        with self.assertRaisesRegex(RuntimeError, "no history"):
            self.adapter(name="empty").save()

    def test_invalid_configuration_fails_before_any_worker_starts(self):
        for change, message in (({"remem_native_config": {"unknown": 1}}, "Unsupported"),
                                ({"remem_native_config": {"extract_method": "openie"}}, "episodic_gist"),
                                ({"embedding_model": "nvidia/NV-Embed-v2"}, "text-embedding"),
                                ({"remem_answer_mode": "reflect"}, "answer_mode")):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                self.adapter(dict(self.config, **change))
        self.assertEqual(self.calls, [])

    def test_native_config_follows_author_example_defaults(self):
        config = native_config(self.config)
        self.assertEqual((config["agent_max_steps"], config["qa_top_k"], config["linking_top_k"]), (3, 10, 5))
        self.assertEqual(config["synonymy_edge_sim_threshold"], 0.8)
        self.assertEqual(config["llm_name"], "gpt-4o-mini")


class WorkerGlueTests(unittest.TestCase):
    def setUp(self):
        self.worker = load_worker_module()

    def test_session_format_keeps_date_header_and_document_metadata(self):
        session = self.worker.format_session({"doc_id": "md-doc-00000001", "text": "Alice: hi", "date": "2024-01-01"})
        self.assertEqual(session, [{"content": "2024-01-01\nAlice: hi", "memorydata_doc": "md-doc-00000001"}])
        self.assertEqual(self.worker.format_session({"doc_id": "d", "text": "x", "date": None})[0]["content"], "x")
        self.assertEqual(self.worker.chunk_session(session), [{"content": "2024-01-01\nAlice: hi"}])
        with self.assertRaises(ValueError):
            self.worker.chunk_session("not a session")

    def test_source_resolution_follows_chunk_edges_and_skips_entities(self):
        edges = {"gists-a": ["verbatim-1", "gists-b", "entity-x"], "facts-a": ["gists-a", "entity-x"],
                 "gists-b": ["verbatim-2"]}
        docs = {"verbatim-1": "d1", "verbatim-2": "d2"}
        groups = self.worker.resolve_source_docs(
            ["verbatim-2", "gists-a", "facts-a", "entity-x", "unknown"],
            verbatim_doc=docs.get, neighbors=lambda name: edges.get(name, []))
        self.assertEqual(groups, [["d2"], ["d1"], ["d1"], [], []])

    def test_query_uses_upstream_agent_and_reports_visited_nodes(self):
        service = self.worker.RememService(source_dir=ROOT)

        class Store:
            def __init__(self, rows):
                self.rows = rows

            def get_rows(self, names):
                return {n: self.rows[n] for n in names if n in self.rows}

        class Vertex:
            index = 0

        class Graph:
            vs = types.SimpleNamespace(find=lambda name: Vertex(), __getitem__=None)

        received = {}

        def rag_each_query(rag, query, return_chunk=None, question_metadata=None):
            received.update(query=query, return_chunk=return_chunk, metadata=question_metadata)
            return [0], [1.0], ([0], [1.0], {"agent_answer": "Tea", "agent_observation": ["step"],
                                              "memorydata_visited_nodes": ["verbatim-1", "verbatim-1", "gists-9"]})

        stores = {"verbatim": Store({"verbatim-1": {"content": "Alice drinks tea",
                                                    "metadata": {"memorydata_doc": "md-doc-00000000"}}}),
                  "gists": Store({"gists-9": {"content": "Alice likes tea"}})}
        service.rag = types.SimpleNamespace(rag_strategy=types.SimpleNamespace(_rag_each_query=rag_each_query),
                                            episodic_embedding_stores=stores, graph=None)
        service._neighbors = lambda name: ["verbatim-1"] if name == "gists-9" else []
        reply = service.handle("query", {"query": "Drink?", "question_date": "2024-01-02T10:00:00"})
        self.assertEqual(received, {"query": "Drink?", "return_chunk": "verbatim",
                                    "metadata": {"date": "2024-01-02T10:00:00"}})
        self.assertEqual(reply["answer"], "Tea")
        self.assertEqual([p["text"] for p in reply["passages"]], ["Alice drinks tea", "Alice likes tea"])
        self.assertEqual(reply["source_docs"], [["md-doc-00000000"], ["md-doc-00000000"]])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            service.handle("delete_everything", {})
        with self.assertRaisesRegex(RuntimeError, "not open"):
            self.worker.RememService().handle("query", {})


class WorkerProcessTests(unittest.TestCase):
    def test_real_subprocess_protocol_survives_native_stdout_and_reports_errors(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"FAKE_WORKER_KIND": "remem", "PYTHONPATH": "should-be-dropped"}):
            env = worker_environment({"api_key_env": "K", "base_url_env": "U"},
                                     environment={"K": SECRET, "U": "http://offline.invalid/v1/", "PATH": os.environ["PATH"],
                                                  "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")})
            self.assertNotIn("PYTHONPATH", env)
            self.assertEqual(env["OPENAI_BASE_URL"], "http://offline.invalid/v1")
            env["FAKE_WORKER_KIND"] = "remem"
            worker = WorkerProcess(sys.executable, FAKE_WORKER, cwd=Path(directory) / "native",
                                   env=env, log_dir=directory, timeout=60)
            try:
                docs = [{"doc_id": "md-doc-00000000", "text": "Alice drinks tea", "date": None}]
                self.assertEqual(worker.call("open", docs=docs), {"opened": True})
                worker.call("index")
                self.assertEqual(worker.call("query", query="q")["source_docs"], [["md-doc-00000000"]])
                with self.assertRaisesRegex(WorkerError, "unknown op"):
                    worker.call("explode")
            finally:
                worker.close()
            self.assertIn("native noise", (Path(directory) / "worker.log").read_text(encoding="utf-8"))
            self.assertNotIn(SECRET, (Path(directory) / "worker.log").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, OPENAI_API_KEY=SECRET, OPENAI_BASE_URL="http://offline.invalid/v1",
                       FAKE_WORKER_KIND="remem", FAKE_HANG_THREAD="1")
            # A crashed parent: stdin closes, a native non-daemon thread lingers.
            orphan = WorkerProcess(sys.executable, FAKE_WORKER, cwd=Path(directory) / "a", env=env,
                                   log_dir=directory, timeout=60)
            orphan.call("open", docs=[])
            orphan.process.stdin.close()
            self.assertEqual(orphan.process.wait(timeout=30), 0)
            orphan.close()
            self.assertIn(f"worker pid {orphan.process.pid}", (Path(directory) / "worker.log").read_text())
            # A worker nobody references any more is still stopped by close_all_workers (atexit).
            pid_holder = WorkerProcess(sys.executable, FAKE_WORKER, cwd=Path(directory) / "b", env=env,
                                       log_dir=directory, timeout=60).process
            import gc
            gc.collect()
            close_all_workers()
            self.assertIsNotNone(pid_holder.wait(timeout=30))
            os.rmdir(Path(directory) / "b")  # its working directory is released
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, OPENAI_API_KEY=SECRET, OPENAI_BASE_URL="http://offline.invalid/v1",
                       FAKE_WORKER_KIND="remem", FAKE_MP_CHILD="1")
            # multiprocessing children (TSM's mp.Manager) inherit the cwd; neither a
            # clean shutdown nor a forced kill may leave them holding it.
            for name, kill in (("clean", False), ("killed", True)):
                cwd = Path(directory) / name
                worker = WorkerProcess(sys.executable, FAKE_WORKER, cwd=cwd, env=env, log_dir=directory, timeout=60)
                worker.call("open", docs=[])
                worker.close(kill=kill)
                shutil.rmtree(cwd)
        with self.assertRaisesRegex(FileNotFoundError, "conda environment"):
            WorkerProcess("", FAKE_WORKER, cwd=".", env={}, log_dir=".")
        with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
            worker_environment({}, environment={})


def serve_chat(testcase, calls):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            calls.append(body)
            payload = json.dumps({"id": "offline", "object": "chat.completion", "created": 0, "model": "offline-model",
                                  "choices": [{"index": 0, "message": {"role": "assistant", "content": "Tea"},
                                               "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 20, "completion_tokens": 1, "total_tokens": 21}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    testcase.addCleanup(server.server_close)
    testcase.addCleanup(server.shutdown)
    return f"http://127.0.0.1:{server.server_port}/v1"


def run_main_pipeline(testcase, preset, kind, extra_config=None, extra_env=None):
    """Two LoCoMo samples through main.py with a real fake-worker subprocess; returns paths and calls."""
    from utils.agent import AgentWrapper
    import main

    chat_calls = []
    directory = tempfile.TemporaryDirectory()
    testcase.addCleanup(directory.cleanup)
    testcase.addCleanup(close_all_workers)  # runs first: workers hold their folders open
    root = Path(directory.name)
    config = yaml.safe_load((ROOT / "config" / preset).read_text(encoding="utf-8"))
    config.update(base_url=serve_chat(testcase, chat_calls), model="offline-model", model_api_trust_env=False,
                  **{f"{kind}_python": sys.executable}, **(extra_config or {}))
    dataset = yaml.safe_load((ROOT / "benchmark/locomo/config/Locomo_qa_4cat_600_dist.yaml").read_text())
    dataset.update(test_files=str(root / "data.json"), max_test_samples=2, chunk_size=4096)
    samples = [{"sample_id": f"sample-{i}", "conversation": {
        "speaker_a": "Alice", "speaker_b": "Bob", "session_1_date_time": "1:56 pm on 8 May, 2023",
        "session_1": [{"speaker": "Alice", "dia_id": "D1:1", "text": "I drink Tea. " + "History detail. " * 160}]},
        "qa": [{"question": "What drink?", "answer": "Tea", "category": 4, "evidence": ["D1:1"]},
               {"question": "What secret?", "answer": "SECRET_GOLD_ONLY", "category": 4, "evidence": []}]}
        for i in range(2)]
    (root / "data.json").write_text(json.dumps(samples))
    for name, value in [("agent.yaml", config), ("dataset.yaml", dataset)]:
        (root / name).write_text(yaml.safe_dump(value))

    def run(*flags):
        args = ["main.py", "--agent_config", str(root / "agent.yaml"), "--dataset_config",
                str(root / "dataset.yaml"), "--artifact_root", str(root / "runs"), *flags]
        with patch.object(sys, "argv", args):
            main.main()

    env = {"OPENAI_API_KEY": SECRET, "FAKE_WORKER_KIND": kind, **(extra_env or {})}
    adapter_module = sys.modules[f"methods.{kind}.{kind}_adapter"]
    stack = [patch.dict(os.environ, env), patch.object(AgentWrapper, "_initialize_tokenizer",
                                                       return_value=CharacterTokenizer()),
             patch.object(adapter_module, "WORKER_SCRIPT", FAKE_WORKER)]
    for item in stack:
        item.start()
        testcase.addCleanup(item.stop)
    return root, run, chat_calls


def worker_calls(root):
    calls = []
    for path in sorted((root / "runs").rglob("calls.jsonl")):
        calls.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return calls


class MainPipelineTests(unittest.TestCase):
    def test_main_builds_reuses_and_forces_rebuild_without_leaking_gold_or_secrets(self):
        from utils.initialization import initialize_and_memorize_agent
        from benchmark.memoryagentbench.prompts.benchmark_templates import get_template

        root, run, chat_calls = run_main_pipeline(self, "topological_remem.yaml", "remem")
        run("--max_test_queries_ablation", "1")  # first context only
        self.assertEqual([c["op"] for c in worker_calls(root)], ["open", "index", "query"])
        first_docs = next((root / "runs" / "agents").rglob(RememAdapter.DOCS_FILE))
        snapshot = first_docs.read_bytes()
        run()
        ops = [c["op"] for c in worker_calls(root)]
        # Context 0 reopens its saved index (open+index) without re-memorizing; context 1 is built.
        self.assertEqual((ops.count("open"), ops.count("index"), ops.count("query")), (3, 3, 4))
        self.assertEqual(first_docs.read_bytes(), snapshot)
        calls_before = len(ops)
        chats_before = len(chat_calls)
        run()
        self.assertEqual((len(worker_calls(root)), len(chat_calls)), (calls_before, chats_before),
                         "Completed queries must not call the worker or the answer model again")
        results = list((root / "runs" / "outputs").rglob("*_results.json"))
        self.assertEqual(len(results), 1)
        payload = json.loads(results[0].read_text())
        self.assertEqual(len(payload["data"]), 4)
        self.assertTrue(all(row["output"] == "Tea" for row in payload["data"]))
        self.assertEqual(payload["data"][0]["retrieved_source_id_groups"], [["D1:1"]])
        state_dirs = [p.parent for p in (root / "runs" / "agents").rglob(RememAdapter.STATE_FILE)]
        self.assertEqual(len(state_dirs), 2)
        for state_dir in state_dirs:
            for path in state_dir.rglob("*"):
                if path.is_file():
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    self.assertNotIn(SECRET, text)
                    self.assertNotIn("SECRET_GOLD_ONLY", text)
            self.assertNotIn("LOCOMO_META", (state_dir / RememAdapter.DOCS_FILE).read_text(encoding="utf-8"))
        for body in chat_calls:
            self.assertNotIn("SECRET_GOLD_ONLY", json.dumps(body))
        config = yaml.safe_load((root / "agent.yaml").read_text())
        dataset = yaml.safe_load((root / "dataset.yaml").read_text())
        with self.assertRaisesRegex(ValueError, "differs"):
            initialize_and_memorize_agent(config, dataset, str(state_dirs[0]), ["changed history"], 0, 2)
        run("--force")
        # --force deletes the local state (and its old call log) and rebuilds both contexts.
        self.assertEqual([c["op"] for c in worker_calls(root)].count("index"), 2)
        self.assertNotEqual(first_docs.read_bytes(), b"")
        for template in ["system", "memorize", "query", "memory_answer"]:
            self.assertTrue(get_template("locomo_qa", template, config["agent_name"]))

    def test_native_answer_mode_uses_the_agent_answer_without_memorydata_answer_call(self):
        root, run, chat_calls = run_main_pipeline(self, "topological_remem.yaml", "remem",
                                                  extra_config={"remem_answer_mode": "native"})
        run("--max_test_queries_ablation", "1")
        self.assertEqual(chat_calls, [])
        payload = json.loads(next((root / "runs" / "outputs").rglob("*_results.json")).read_text())
        self.assertTrue(all(row["output"] == "Tea" for row in payload["data"]))


if __name__ == "__main__":
    unittest.main()
