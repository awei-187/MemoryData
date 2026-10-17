import argparse
import ast
import asyncio
import copy
import csv
import json
import random
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from benchmark.omnimemeval.common import PROTOCOL, atomic_json, require_identity
from benchmark.omnimemeval.loader import iter_eval_units, persona_options, flatten_beam_session
from benchmark.omnimemeval.templates import format_question
from evaluation.omnimemeval.evaluate import evaluate, build_calls
from evaluation.omnimemeval.scoring import binary_label, extract_choice_answer, kendall_tau_b, nugget_score, ordering_score

ROOT = Path(__file__).resolve().parents[1]


class DatasetFixtures:
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)

    def config(self, kind, name, data, lines=False, **extra):
        path = self.root / name
        path.write_text("\n".join(json.dumps(row) for row in data) if lines else json.dumps(data), encoding="utf-8")
        return dict(dataset="OmniMemEval", sub_dataset="omni_" + kind, test_files=str(path), chunk_size=4096, max_test_samples=0, **extra)

    def locomo_config(self):
        return self.config("locomo", "locomo.json", [{"sample_id": "conv-26", "conversation": {"session_1_date_time": "2022-05-04", "session_1": [{"speaker": "Alice", "dia_id": "D1:1", "text": "My dog is Spot."}]}, "qa": [{"question": "Dog?", "answer": "GOLD_ONLY", "category": 1, "evidence": ["D1:1"]}, {"question": "When?", "answer": "Yesterday", "category": 2}, {"question": "Adversarial?", "answer": "secret", "category": 5}]}])


class DatasetTests(DatasetFixtures, unittest.TestCase):
    def test_locomo_mapping_no_label_leak_and_stable_ids(self):
        config = self.locomo_config()
        unit = next(iter_eval_units(config))
        text = unit["context_chunks"][0]["text"]
        self.assertIn("Alice", text)
        self.assertIn("2022-05-04", text)
        self.assertNotIn("GOLD_ONLY", text)
        self.assertEqual(len(unit["questions"]), 2)
        filtered = next(iter_eval_units(dict(config, categories=[2])))
        self.assertEqual(filtered["questions"][0]["question_id"], unit["questions"][1]["question_id"])
        self.assertEqual(unit["context_chunks"][0]["source_ids"], ["D1:1"])

    def test_lme_preserves_date_and_removes_special_tokens(self):
        config = self.config("longmemeval", "lme.json", [{"question_id": "q0", "question": "Job?", "answer": "GOLD", "question_date": "2020-02-01", "question_type": "knowledge-update", "haystack_sessions": [[{"role": "assistant", "content": "teacher<|endoftext|>"}]], "haystack_dates": ["2020-01-01"], "haystack_session_ids": ["s1"], "answer_session_ids": ["s1"]}])
        unit = next(iter_eval_units(config))
        self.assertNotIn("<|endoftext|>", str(unit))
        self.assertIn("assistant", unit["context_chunks"][0]["text"])
        self.assertIn("2020-02-01", format_question(unit["questions"][0]))
        self.assertNotIn("GOLD", format_question(unit["questions"][0]))

    def test_persona_grouping_options_and_local_random(self):
        histories = self.root / "histories"
        histories.mkdir()
        (histories / "chat_persona7.json").write_text(json.dumps({"chat_history": [{"role": "user", "content": "I like tea."}]}))
        path = self.root / "benchmark.csv"
        rows = [{"persona_id": "7", "user_query": "{'content': 'Drink?'}", "correct_answer": "Tea", "incorrect_answers": '["A","B","C"]', "pref_type": "updated"} for _ in range(2)]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        config = dict(dataset="OmniMemEval", sub_dataset="omni_personamem_v2", test_files=str(path), history_dir=str(histories))
        random.seed(99)
        before = random.getstate()
        unit = next(iter_eval_units(config))
        self.assertEqual(before, random.getstate())
        self.assertEqual(len(unit["questions"]), 2)
        self.assertEqual(unit["questions"][0]["question"], "Drink?")
        self.assertNotIn("correct_answer", format_question(unit["questions"][0]))
        expected = ["Tea", "A", "B", "C"]
        random.Random(1).shuffle(expected)
        self.assertEqual(list(unit["questions"][1]["options"].values()), expected)

    def test_halumem_full_history_no_annotation_ingestion(self):
        config = self.config("halumem", "hm.jsonl", [{"uuid": "u1", "sessions": [{"start_time": "2020", "dialogue": [{"role": "user", "content": "Old job"}], "memory_points": [{"memory_content": "GOLD_ONLY"}], "questions": [{"question": "Job?", "answer": "GOLD_ONLY", "question_type": "Memory Conflict", "difficulty": "easy"}]}, {"start_time": "2021", "dialogue": [{"role": "user", "content": "New job"}], "questions": []}]}], lines=True)
        unit = next(iter_eval_units(config))
        self.assertEqual(unit["session_count"], 2)
        self.assertNotIn("GOLD_ONLY", str(unit["context_chunks"]))
        self.assertEqual(unit["questions"][0]["history_protocol"], "all_user_sessions_before_qa")

    def test_beam_nested_sessions_diverse_smoke_and_lazy_consumption(self):
        row = {"conversation_id": "b1", "chat": [{"plan-0": [{"turns": [[{"role": "user", "content": "First event", "time_anchor": "2020"}]]}]}], "probing_questions": repr({"information_extraction": [{"question": "Fact?", "answer": "x", "rubric": ["RUBRIC_ONLY"]}], "event_ordering": [{"question": "Order?", "rubric": ["First", "Second"]}], "summarization": [{"question": "Summarize", "ideal_summary": "Summary"}]})}
        config = self.config("beam", "beam.json", [row], lines=True, max_queries_per_unit=3, smoke_diverse_questions=True)
        # Malformed later unit proves max_test_samples does not parse ahead.
        with Path(config["test_files"]).open("a") as stream:
            stream.write('\n{"bad": true}\n')
        units = list(iter_eval_units(dict(config, max_test_samples=1)))
        self.assertEqual(len(units), 1)
        unit = units[0]
        self.assertEqual(len(unit["questions"]), 3)
        self.assertNotIn("RUBRIC_ONLY", str(unit["context_chunks"]))
        self.assertNotIn("RUBRIC_ONLY", format_question(unit["questions"][0]))
        self.assertIn("2020", unit["context_chunks"][0]["text"])

    def test_missing_data_and_empty_filter_fail(self):
        with self.assertRaises(FileNotFoundError):
            next(iter_eval_units(dict(dataset="OmniMemEval", sub_dataset="omni_locomo", test_files=str(self.root / "missing"))))
        with self.assertRaises(ValueError):
            list(iter_eval_units(dict(self.locomo_config(), sample_ids=["missing"])))

    def test_large_jsonl_stops_before_later_units(self):
        row = {"conversation_id": "b", "chat": [[{"role": "user", "content": "History"}]], "probing_questions": {"abstention": [{"question": "Unknown?", "answer": "unknown"}]}}
        config = self.config("beam", "large.jsonl", [row], lines=True)
        with Path(config["test_files"]).open("a") as stream:
            for _ in range(128):
                stream.write('\n' + json.dumps({"unprocessed": "x" * 65536}))
        self.assertGreater(Path(config["test_files"]).stat().st_size, 8 * 1024 * 1024)
        self.assertEqual(len(list(iter_eval_units(dict(config, max_test_samples=1)))), 1)


class ScoringTests(unittest.TestCase):
    def test_strict_labels(self):
        for text in ('{"label":"CORRECT"}', '```json\n{"label":"CORRECT"}\n```', "Explanation. CORRECT"):
            self.assertTrue(binary_label(text))
        self.assertFalse(binary_label('{"label":"WRONG"}'))
        for text in ("INCORRECT", "CORRECT or WRONG", '{"label":"INCORRECT"}', "", '{"label": WRONG}'):
            with self.assertRaises(ValueError):
                binary_label(text)

    def test_beam_scores_and_ordering(self):
        self.assertEqual([nugget_score('{"score":' + str(x) + '}') for x in (0, .5, 1)], [0, .5, 1])
        self.assertEqual(ordering_score('{"positions":[{"position":1},{"position":2}]}', 2)[0], 1)
        self.assertEqual(ordering_score('{"positions":[{"position":2},{"position":1}]}', 2)[0], 0)
        self.assertEqual(ordering_score('{"positions":[]}', 2)[0], 0)
        self.assertEqual(ordering_score('{"positions":[{"position":1}]}', 2)[0], .25)
        with self.assertRaises(ValueError):
            nugget_score("not JSON")

    def test_valid_scoring_matches_upstream_pure_functions(self):
        source = ROOT.parent / "OmniMemEval/scripts"
        namespace = {"re": __import__("re"), "json": json}
        for relative, names in (("beam/beam_eval.py", {"_kendall_tau_b", "_extract_item_score"}), ("personamem_v2/pm_responses.py", {"extract_choice_answer"})):
            tree = ast.parse((source / relative).read_text(encoding="utf-8"))
            module = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
            exec(compile(module, str(source / relative), "exec"), namespace)
        for permutation in ([1, 2, 3], [3, 2, 1], [1, 1, 2]):
            self.assertEqual(kendall_tau_b([0, 1, 2], permutation), namespace["_kendall_tau_b"]([0, 1, 2], permutation))
        for value in (0, .25, .5, .75, 1):
            raw = json.dumps({"score": value})
            self.assertEqual(nugget_score(raw), namespace["_extract_item_score"](raw)[0])
        for answer in ("(a)", "Final answer (b)", "(a) or (b)", "<final_answer>(b)</final_answer>"):
            self.assertEqual(extract_choice_answer(answer, "(b)"), namespace["extract_choice_answer"](answer, "(b)"))

    def test_scoring_resume_invalidation_and_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            row = {"qa_pair_id": "q1", "status": "success", "output": "dog", "answer": "dog", "eval_metadata": {"benchmark": "locomo", "question_text": "Pet?", "category": "1"}}
            source = {"protocol": PROTOCOL, "identity": {"id": "r1"}, "data": [row, dict(row, qa_pair_id="q2", status="failed")]}
            atomic_json(path, source)
            calls = []
            async def judge(messages):
                calls.append(messages)
                return '{"label":"CORRECT"}'
            config = {"model": "fake", "num_runs": 1}
            _, summary = asyncio.run(evaluate(path, config, complete=judge))
            self.assertEqual((summary["valid_scores"], summary["skipped"], summary["coverage"]), (1, 1, .5))
            asyncio.run(evaluate(path, config, complete=judge))
            self.assertEqual(len(calls), 1)
            source["data"][0]["output"] = "cat"
            atomic_json(path, source)
            asyncio.run(evaluate(path, config, complete=judge))
            self.assertEqual(len(calls), 2)
            with self.assertRaises(ValueError):
                asyncio.run(evaluate(path, dict(config, model="other"), complete=judge))
            self.assertEqual(json.loads(path.read_text())["data"][0]["output"], "cat")

    def test_failed_judge_preserves_attempts_and_can_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            row = {"qa_pair_id": "q", "status": "success", "output": "x", "answer": "x", "eval_metadata": {"benchmark": "halumem", "question_text": "Fact?"}}
            atomic_json(path, {"protocol": PROTOCOL, "identity": {}, "data": [row]})
            async def invalid(messages):
                return "INCORRECT"
            with patch("evaluation.omnimemeval.evaluate.asyncio.sleep", new=AsyncMock()):
                output, summary = asyncio.run(evaluate(path, {"model": "fake"}, complete=invalid))
            self.assertIsNone(summary["score"])
            self.assertEqual(summary["failed"], 1)
            saved = json.loads(output.read_text())
            self.assertEqual(len(saved["rows"][0]["attempts"]), 3)
            async def valid(messages):
                return '{"label":"WRONG"}'
            _, summary = asyncio.run(evaluate(path, {"model": "fake"}, complete=valid))
            self.assertEqual((summary["score"], summary["valid_scores"]), (0, 1))


class RuntimeTests(DatasetFixtures, unittest.TestCase):
    def test_resume_retry_and_content_addressed_state(self):
        from benchmark.omnimemeval.runtime import run
        config = self.locomo_config()
        args = argparse.Namespace(artifact_root=str(self.root / "artifacts"), force=False, retry_failed_queries=False, max_test_queries_ablation=0)
        ingested, queried = [], []
        class FakeAgent:
            def send_message(self, query, **kwargs):
                self_outer.assertNotIn("rubric", kwargs["eval_metadata"])
                queried.append(kwargs["query_id"])
                if len(queried) == 1:
                    raise RuntimeError("injected query failure")
                return {"output": "Spot", "input_len": 10, "output_len": 1}
        self_outer = self
        def initialize(ac, dc, path, chunks, *rest, **kwargs):
            ingested.append((ac["_omni_sample_id"], kwargs["force_rebuild"]))
            Path(path).mkdir(parents=True, exist_ok=True)
            return FakeAgent(), None
        def metrics(response, query, answer, dc, running, rows, qid, cid, pair, metadata):
            rows.append(dict(response, qa_pair_id=pair, query_id=qid, context_id=cid, answer=answer, eval_metadata=metadata))
            return running, rows
        with patch.dict("sys.modules", {"utils.initialization": types.SimpleNamespace(initialize_and_memorize_agent=initialize), "utils.eval_other_utils": types.SimpleNamespace(metrics_summarization=metrics)}):
            path = run(args, {"agent_name": "fake", "model": "fake"}, config)
            run(args, {"agent_name": "fake", "model": "fake"}, config)
            self.assertEqual(len(queried), 2)
            args.retry_failed_queries = True
            run(args, {"agent_name": "fake", "model": "fake"}, config)
            self.assertEqual(len(queried), 3)
            self.assertEqual(ingested[-1][1], False)
            self.assertEqual(len(json.loads(path.read_text())["data"]), 2)
            changed = run(args, {"agent_name": "fake", "model": "different"}, config)
            self.assertNotEqual(path, changed)
            self.assertTrue(ingested[-1][1])


class CompatibilityTests(unittest.TestCase):
    def test_all_method_presets_have_all_new_task_templates(self):
        import yaml
        from benchmark.memoryagentbench.prompts.benchmark_templates import get_template
        configs = list((ROOT / "config").glob("*.yaml"))
        self.assertEqual(len(configs), 22)
        for path in configs:
            agent = yaml.safe_load(path.read_text(encoding="utf-8"))
            for benchmark in ("locomo", "longmemeval", "personamem_v2", "halumem", "beam"):
                for name in ("system", "memorize", "query", "fact_extraction", "memory_answer"):
                    self.assertTrue(get_template("omni_" + benchmark, name, agent["agent_name"]))

    def test_original_dataset_dispatch_and_scoring(self):
        from benchmark.memoryagentbench import loader
        from utils.eval_other_utils import post_process
        cases = [("LoCoMo", "locomo_qa", "load_locomo_eval_data"), ("LongBench", "longbench_rep150_proportional", "load_longbench_eval_data"), ("MemBench", "membench_simple", "load_membench_eval_data"), ("Accurate_Retrieval", "longmemeval_s*", "load_data_huggingface")]
        for dataset, sub, symbol in cases:
            config = {"dataset": dataset, "sub_dataset": sub, "max_test_samples": 1, "seed": 42}
            with patch.object(loader, symbol, return_value={"data": []}) as called:
                self.assertEqual(loader.load_eval_data(config), {"data": []})
                called.assert_called_once()
            text = "B" if dataset in {"LongBench", "MemBench"} else "Spot"
            metrics, _ = post_process({"output": text}, text, config)
            self.assertEqual(metrics["exact_match"], 1)
            self.assertNotIn("memorydata_aux_exact_match", metrics)

    def test_original_resume_recomputes_legacy_metrics(self):
        from utils.initialization import load_existing_results
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            atomic_json(path, {"data": [{"output": "Spot", "answer": "Spot", "query": "Pet?", "query_id": 0, "context_id": 0, "input_len": 5, "output_len": 1, "query_time_len": .1}]})
            metrics, rows, completed, skipped = load_existing_results(str(path), {"sub_dataset": "longmemeval_s*", "debug": False}, [[("Pet?", "Spot", "p1")]])
            self.assertEqual(metrics["exact_match"], [1])
            self.assertEqual((len(rows), completed, skipped), (1, {0}, {0}))

    def test_new_time_and_rag_history_budget_without_service(self):
        from utils.agent import AgentWrapper
        agent = AgentWrapper.__new__(AgentWrapper)
        agent.sub_dataset = "omni_beam"
        agent.agent_name = "Embedding_rag_qwen3_embedding_4b"
        agent._current_eval_metadata = {}
        self.assertEqual(agent._benchmark_time_suffix(), "")
        agent._current_eval_metadata = {"question_time": "2020-01-01"}
        self.assertIn("2020-01-01", agent._benchmark_time_suffix())
        agent.context = ""
        agent.chunks = []
        agent.context_len = 0
        agent.chunk_size = 10
        agent.input_length_limit = 5
        for text in ("one", "two", "three"):
            agent._handle_rag_agent(text, True, None, 0)
        self.assertEqual(agent.chunks, ["one", "two", "three"])
        agent.sub_dataset = "longmemeval_s*"
        self.assertTrue(agent._benchmark_time_suffix().startswith("\n\nCurrent Time:"))

    def test_prompts_identical_to_upstream_and_identity_refuses_mismatch(self):
        from benchmark.omnimemeval.common import file_digest
        self.assertEqual(file_digest(ROOT / "evaluation/omnimemeval/upstream_prompts.py"), file_digest(ROOT.parent / "OmniMemEval/scripts/utils/prompts.py"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.json"
            require_identity(path, {"data": "a"})
            with self.assertRaises(ValueError):
                require_identity(path, {"data": "b"})


if __name__ == "__main__":
    unittest.main()
