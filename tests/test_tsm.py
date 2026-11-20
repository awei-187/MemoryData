"""Offline TSM integration checks: adapter, worker glue and the main pipeline.

No model, embedding service, Neo4j or TSM environment is used. The worker is
replaced by tests/fixtures/fake_memory_worker.py (same protocol, one file per
"Neo4j group"), and TSM's retriever by a small fake.
"""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from methods.tsm.tsm_adapter import TsmAdapter, native_config, split_evidence
try:  # unittest discover -s tests puts tests/ itself on sys.path
    from test_remem import ROOT, SECRET, run_main_pipeline, worker_calls
except ImportError:  # python -m unittest tests.test_tsm
    from tests.test_remem import ROOT, SECRET, run_main_pipeline, worker_calls


def load_worker_module():
    spec = importlib.util.spec_from_file_location("tsm_worker_under_test", ROOT / "methods/tsm/tsm_worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeWorker:
    def __init__(self, log):
        self.log = log
        self.closed = False

    def call(self, op, timeout=None, **args):
        self.log.append((op, args))
        if op == "open":
            self.docs = args["docs"]
            self.cache = Path(args["cache_folder"])
            return {"opened": True}
        if op in {"index", "reopen"}:
            self.cache.mkdir(parents=True, exist_ok=True)
            (self.cache / "tree.json").write_text("{}")
            return {"tree_nodes": len(self.docs), "snapshots": 1}
        if op == "query":
            reply = {"evidence": "--- Retrieved Chunks ---\nB\n\nA\n\n--- Snapshot Summaries ---\n[Snapshot s1] persona",
                     "leaf_positions": [1, 0, 7], "leaf_texts": {"1": "B", "0": "A", "7": ""}}
            if args.get("native_answer"):
                reply["answer"] = "native Tea"
            return reply
        if op == "reset":
            return {"reset": True}
        raise AssertionError(op)

    def close(self, kill=False):
        self.closed = True


class AdapterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.calls = []
        self.workers = []
        self.config = {"model": "gpt-4o-mini", "embedding_model": "text-embedding-3-small", "embedding_dim": 1536,
                       "retrieve_num": 1, "_memory_history_fingerprint": "history-a"}
        self.dataset = {"dataset": "LoCoMo", "sub_dataset": "locomo_qa"}

    def factory(self):
        worker = FakeWorker(self.calls)
        self.workers.append(worker)
        return worker

    def adapter(self, config=None, name="state"):
        return TsmAdapter(config or self.config, self.dataset, self.root / name, worker_factory=self.factory)

    def ready(self):
        adapter = self.adapter()
        adapter.add_chunk("Alice drinks tea", source_ids=["D1:1"], timestamp="2023-05-08T13:56:00")
        adapter.add_chunk("Bob drinks coffee", source_ids=["D1:2"])
        adapter.save()
        return adapter

    def test_build_uses_owned_group_release_settings_and_no_secrets(self):
        adapter = self.ready()
        self.assertRegex(adapter.state["group_id"], r"^md_tsm_[a-f0-9]{32}$")
        op, args = self.calls[0]
        self.assertEqual(op, "open")
        self.assertEqual(args["group_id"], adapter.state["group_id"])
        self.assertEqual(args["config"]["extraction"], {"method": "graphiti"})
        self.assertEqual(args["config"]["entity_clustering"], {"enable": True, "threshold": 0.1})
        self.assertEqual(args["config"]["retriever"]["kwargs"]["max_chunk_setting"], 20)
        self.assertEqual(args["config"]["embedding"]["api_key_env"], "MEMORYDATA_EMBEDDING_API_KEY")
        self.assertEqual([op for op, _ in self.calls], ["open", "index"])
        self.assertNotIn("Alice drinks tea", adapter.path.read_text(encoding="utf-8"))

    def test_leaf_ids_map_to_sources_summary_has_no_source_and_top_k_caps_chunks(self):
        adapter = self.ready()
        answer, entries = adapter.retrieve("Drink?", query_timestamp="2023/05/30 (Tue) 10:00")
        self.assertEqual(answer, "")
        self.assertEqual([(e["text"], e["source_ids"]) for e in entries],
                         [("B", ["D1:2"]), ("[Snapshot s1] persona", [])])
        query = self.calls[-1][1]
        self.assertEqual((query["question_date"], query["native_answer"]), ("2023-05-30T10:00:00", False))
        native, _ = adapter.retrieve("Drink?", native_answer=True)
        self.assertEqual(native, "native Tea")

    def test_reload_reopens_cache_and_refuses_new_history(self):
        self.ready()
        adapter = self.adapter()
        adapter.load()
        adapter.retrieve("Drink?")
        self.assertEqual([op for op, _ in self.calls][2:], ["open", "reopen", "query"])
        with self.assertRaisesRegex(RuntimeError, "completed TSM index"):
            adapter.add_chunk("more")

    def test_unfinished_build_resets_its_own_group_before_rebuilding(self):
        first = self.adapter()
        first.add_chunk("stale")
        group = first.state["group_id"]
        second = self.adapter()
        second.add_chunk("fresh")
        reset = [(op, args) for op, args in self.calls if op == "reset"]
        self.assertEqual(len(reset), 1)
        self.assertEqual(self.calls[0][1]["group_id"], group)
        self.assertEqual(self.calls[0][1]["docs"], [])
        self.assertTrue(self.workers[0].closed, "the reset worker must not be reused for the build")
        second.save()
        self.assertEqual([d["text"] for d in self.calls[-2][1]["docs"]], ["fresh"])

    def test_retriever_settings_reuse_build_settings_refuse(self):
        self.ready()
        self.adapter(dict(self.config, tsm_native_config={"retriever_kwargs": {"max_chunk_setting": 10}})).load()
        for change in ({"tsm_native_config": {"entity_clustering": {"enable": True, "threshold": 0.2}}},
                       {"_memory_history_fingerprint": "history-b"}, {"tsm_neo4j_uri": "bolt://other:7687"}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "differs"):
                self.adapter(dict(self.config, **change))

    def test_force_reset_deletes_only_the_recorded_group(self):
        adapter = self.ready()
        group = adapter.state["group_id"]
        with patch.object(TsmAdapter, "start_worker", lambda self: FakeWorker(self_calls)):
            self_calls = []
            TsmAdapter.reset_saved(self.config, self.dataset, adapter.state_dir)
        self.assertEqual([op for op, _ in self_calls], ["open", "reset"])
        self.assertEqual(self_calls[0][1]["group_id"], group)
        with self.assertRaisesRegex(ValueError, "endpoint changed"):
            TsmAdapter.reset_saved(dict(self.config, tsm_neo4j_uri="bolt://other:7687"), self.dataset,
                                   adapter.state_dir)
        state = json.loads(adapter.path.read_text())
        state["group_id"] = "someone_elses_group"
        adapter.path.write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "not owned"):
            TsmAdapter.reset_saved(self.config, self.dataset, adapter.state_dir)

    def test_missing_neo4j_password_and_invalid_configuration_fail_early(self):
        adapter = self.adapter()
        with patch.dict(os.environ, {"NEO4J_PASSWORD": ""}), self.assertRaisesRegex(ValueError, "NEO4J_PASSWORD"):
            adapter.start_worker()
        for change, message in (({"tsm_native_config": {"use_user_only": True}}, "Unsupported"),
                                ({"tsm_answer_mode": "other"}, "answer_mode"),
                                ({"embedding_model": ""}, "embedding model")):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, message):
                self.adapter(dict(self.config, **change))

    def test_evidence_split_handles_summary_only_results(self):
        self.assertEqual(split_evidence("only summaries", []), ([], "only summaries"))
        self.assertEqual(split_evidence("", []), ([], ""))
        self.assertEqual(native_config(self.config)["temporal"], {"enable": False})


class WorkerGlueTests(unittest.TestCase):
    def setUp(self):
        self.worker = load_worker_module()

    def test_turn_format_and_leaf_positions(self):
        self.assertEqual(self.worker.format_turn({"text": "Alice: hi", "date": "2023-05-08T13:56:00"}),
                         "Date: 2023-05-08T13:56:00\nAlice: hi")
        self.assertEqual(self.worker.format_turn({"text": "x", "date": None}), "x")
        self.assertEqual(self.worker.leaf_positions({"a": ["leaf_3", "topic_1", "leaf_0"], "b": ["leaf_3"]}), [3, 0])

    def test_query_passes_question_date_and_reports_observed_leaves(self):
        service = self.worker.TsmService(source_dir=ROOT)
        received = {}

        class Retriever:
            cache_tree = {"leaf_0": {"text": "Date: 2023-05-08\nAlice drinks tea"}}

            def query(self, question, **kwargs):
                received.update(question=question, **kwargs)
                self.memorydata_last_chunk_ids = {"chunks": ["leaf_0"]}
                return {"chunks": "Date: 2023-05-08\nAlice drinks tea"}

        service.bundle = type("Bundle", (), {"retriever_kwargs": {"max_chunk_setting": 20}})()
        service.retriever = Retriever()
        reply = service.handle("query", {"query": "Drink?", "question_date": "2023-05-30T00:00:00"})
        self.assertEqual(received, {"question": "Drink?", "max_chunk_setting": 20,
                                    "question_date": "2023-05-30T00:00:00"})
        self.assertEqual((reply["leaf_positions"], reply["leaf_texts"]),
                         ([0], {"0": "Date: 2023-05-08\nAlice drinks tea"}))
        service.retriever.query = lambda question, **kwargs: {"chunks": "", "error": "boom"}
        with self.assertRaisesRegex(RuntimeError, "boom"):
            service.handle("query", {"query": "Drink?"})
        with self.assertRaisesRegex(RuntimeError, "not open"):
            self.worker.TsmService().handle("query", {})


class MainPipelineTests(unittest.TestCase):
    def test_main_builds_reuses_and_force_resets_only_owned_groups(self):
        graphs = tempfile.TemporaryDirectory()
        self.addCleanup(graphs.cleanup)
        graph_dir = Path(graphs.name)
        (graph_dir / "foreign_group.json").write_text("[]")
        root, run, chat_calls = run_main_pipeline(
            self, "topological_tsm.yaml", "tsm",
            extra_env={"FAKE_GRAPH_DIR": str(graph_dir), "NEO4J_PASSWORD": "neo4j-secret-not-on-disk"})
        run()
        ops = [c["op"] for c in worker_calls(root)]
        self.assertEqual((ops.count("index"), ops.count("query")), (2, 4))
        groups_before = {p.name for p in graph_dir.glob("md_tsm_*.json")}
        self.assertEqual(len(groups_before), 2)
        payload = json.loads(next((root / "runs" / "outputs").rglob("*_results.json")).read_text())
        self.assertTrue(all(row["output"] == "Tea" for row in payload["data"]))
        self.assertEqual(payload["data"][0]["retrieved_source_id_groups"], [["D1:1"], []])
        for path in (root / "runs" / "agents").rglob("*"):
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="ignore")
                self.assertNotIn(SECRET, text)
                self.assertNotIn("neo4j-secret-not-on-disk", text)
                self.assertNotIn("SECRET_GOLD_ONLY", text)
        run("--force")
        groups_after = {p.name for p in graph_dir.glob("md_tsm_*.json")}
        self.assertEqual(len(groups_after), 2)
        self.assertFalse(groups_before & groups_after)
        self.assertTrue((graph_dir / "foreign_group.json").exists())
        config = yaml.safe_load((root / "agent.yaml").read_text())
        self.assertEqual(config["agent_name"], "Agentic_memory_tsm")


if __name__ == "__main__":
    unittest.main()
