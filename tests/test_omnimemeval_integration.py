"""Real pipeline/SDK/vector-store tests with HTTP responses simulated in memory.

No server is started and no model weights are downloaded. Scores here verify
plumbing only; they are not model-quality measurements.
"""
import asyncio
import hashlib
import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import yaml
try:
    import httpx2
except ImportError:
    httpx2 = None

from benchmark.omnimemeval.common import atomic_json, file_digest
from evaluation.omnimemeval.evaluate import evaluate
from test_omnimemeval import DatasetFixtures, ROOT


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [ord(character) for character in text]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


class SimulatedModelPipelineTests(DatasetFixtures, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.requests = []
        self.environment = patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test", "MEM0_TELEMETRY": "False", "MEM0_DIR": str(self.root / "mem0_home")})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        from utils.agent import AgentWrapper
        tokenizer = patch.object(AgentWrapper, "_initialize_tokenizer", return_value=CharacterTokenizer())
        tokenizer.start()
        self.addCleanup(tokenizer.stop)
        owner = self
        def sync_send(client, request, *args, **kwargs):
            return owner.respond(request)
        async def async_send(client, request, *args, **kwargs):
            return owner.respond(request)
        targets = [("httpx.Client.send", sync_send), ("httpx.AsyncClient.send", async_send)]
        if httpx2 is not None:
            targets.extend([("httpx2.Client.send", sync_send), ("httpx2.AsyncClient.send", async_send)])
        for target, replacement in targets:
            mocked = patch(target, new=replacement)
            mocked.start()
            self.addCleanup(mocked.stop)

    def respond(self, request):
        self.assertEqual(request.url.host, "offline.invalid", "Unexpected external HTTP request")
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        if request.url.path.endswith("/embeddings"):
            inputs = body["input"]
            if isinstance(inputs, str) or (inputs and isinstance(inputs[0], int)):
                inputs = [inputs]
            data = []
            for index, value in enumerate(inputs):
                digest = hashlib.sha256(str(value).encode()).digest()
                vector = [value / 255 - .5 for value in digest] + [0.] * (2560 - len(digest))
                norm = math.sqrt(sum(value * value for value in vector))
                data.append({"object": "embedding", "index": index, "embedding": [value / norm for value in vector]})
            content = {"object": "list", "model": body["model"], "data": data, "usage": {"prompt_tokens": 12, "total_tokens": 12}}
        elif request.url.path.endswith("/chat/completions"):
            text = "\n".join(message.get("content", "") for message in body["messages"])
            if body.get("response_format"):
                answer = json.dumps({"facts": ["Alice likes tea."], "memory": [{"id": "0", "text": "Alice likes tea.", "event": "ADD"}]})
            elif '"positions"' in text:
                answer = '{"positions":[{"position":1},{"position":2}]}'
            elif "CORRECT" in text and "WRONG" in text:
                answer = '{"label":"CORRECT"}'
            elif "score" in text.lower() and "rubric" in text.lower():
                answer = '{"score":0.5}'
            elif "exactly one option" in text:
                answer = "(a)"
            else:
                answer = "Tea"
            content = {"id": "simulated", "object": "chat.completion", "created": 0, "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": answer}}], "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}}
        else:
            self.fail("Unexpected HTTP route: " + request.url.path)
        response_type = httpx2.Response if httpx2 is not None and isinstance(request, httpx2.Request) else httpx.Response
        return response_type(200, request=request, json=content)

    def datasets(self):
        common = {"context_max_length": 100000, "generation_max_length": 32, "max_test_samples": 1, "seed": 42, "shots": 0, "debug": False, "tag": "simulated"}
        locomo = self.locomo_config()
        locomo.update(common)
        lme = self.config("longmemeval", "lme.json", [{"question_id": "lme0", "question": "Drink?", "answer": "Tea", "haystack_sessions": [[{"role": "user", "content": "Alice likes tea."}]], "haystack_dates": ["2020-01-01"], "question_date": "2020-02-01", "question_type": "single-session-user"}])
        lme.update(common)
        hm = self.config("halumem", "hm.jsonl", [{"uuid": "user0", "sessions": [{"start_time": "2020", "dialogue": [{"role": "user", "content": "Alice likes tea."}], "questions": [{"question": "Drink?", "answer": "Tea", "question_type": "Basic Fact Recall"}]}]}], lines=True)
        hm.update(common)
        beam = self.config("beam", "beam.json", [{"conversation_id": "beam0", "chat": [[{"role": "user", "content": "Alice first liked coffee, then tea."}]], "probing_questions": {"information_extraction": [{"question": "Drink?", "answer": "Tea", "rubric": ["Mentions tea"]}], "event_ordering": [{"question": "Order?", "rubric": ["Coffee", "Tea"]}]}}], lines=True)
        beam.update(common)
        history = self.root / "histories"
        history.mkdir()
        (history / "chat_persona7.json").write_text(json.dumps({"chat_history": [{"role": "user", "content": "Alice likes tea."}]}))
        persona = self.root / "persona.csv"
        import csv
        with persona.open("w", newline="", encoding="utf-8") as stream:
            row = {"persona_id": "7", "user_query": "Drink?", "correct_answer": "Tea", "incorrect_answers": '["Coffee","Water","Juice"]', "pref_type": "preference"}
            writer = csv.DictWriter(stream, fieldnames=row)
            writer.writeheader()
            writer.writerow(row)
        pm = dict(common, dataset="OmniMemEval", sub_dataset="omni_personamem_v2", test_files=str(persona), history_dir=str(history), chunk_size=4096)
        return {"locomo": locomo, "longmemeval": lme, "personamem_v2": pm, "halumem": hm, "beam": beam}

    def invoke_main(self, agent_file, dataset_file, artifact_root):
        import main
        with patch.object(sys, "argv", ["main.py", "--agent_config", str(agent_file), "--dataset_config", str(dataset_file), "--artifact_root", str(artifact_root)]):
            main.main()

    def test_fifteen_real_adapter_pipelines_and_resume_with_simulated_http(self):
        report = {"mode": "simulated_http_no_model", "cases": []}
        datasets = self.datasets()
        methods = {"long_context": "reference_long_context_agent.yaml", "embedding_rag": "reference_embedding_rag.yaml", "mem0": "sequential_mem0.yaml"}
        for method, filename in methods.items():
            for kind, dataset in datasets.items():
                with self.subTest(method=method, benchmark=kind):
                    agent = yaml.safe_load((ROOT / "config" / filename).read_text(encoding="utf-8"))
                    agent.update(base_url="http://offline.invalid/v1", output_dir=f"./results/outputs/{method}")
                    if method != "long_context":
                        agent["embedding_base_url"] = "http://offline.invalid/v1"
                    agent_file = self.root / f"{method}.yaml"
                    agent_file.write_text(yaml.safe_dump(agent), encoding="utf-8")
                    dataset_file = self.root / f"{kind}.yaml"
                    dataset_file.write_text(yaml.safe_dump(dataset), encoding="utf-8")
                    artifacts = self.root / "runs" / method / kind
                    before = len(self.requests)
                    self.invoke_main(agent_file, dataset_file, artifacts)
                    paths = list(artifacts.rglob("benchmark_results.json"))
                    self.assertEqual(len(paths), 1)
                    path = paths[0]
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    self.assertTrue({"agent_config", "dataset_config", "data", "metrics", "averaged_metrics", "time_cost"} <= payload.keys())
                    self.assertTrue(all(row["status"] == "success" for row in payload["data"]), payload["data"])
                    self.assertEqual(path.relative_to(artifacts).parts[:3], ("outputs", method, "OmniMemEval"))
                    self.assertTrue((artifacts / "agents").is_dir())
                    self.assertNotIn("answer_context_truncated", payload["metrics"])
                    for key, values in payload["metrics"].items():
                        factor = 1 if "_len" in key or "_time" in key else 100
                        self.assertEqual(payload["averaged_metrics"][key], sum(values) / len(values) * factor)
                    if method != "long_context":
                        self.assertTrue(any(route.endswith("/embeddings") for route, _ in self.requests[before:]))
                    digest, count = file_digest(path), len(self.requests)
                    self.invoke_main(agent_file, dataset_file, artifacts)
                    self.assertEqual(file_digest(path), digest)
                    self.assertEqual(len(self.requests), count)
                    scoring_config = {"model": "mock-judge", "base_url": "http://offline.invalid/v1"}
                    score_path, summary = asyncio.run(evaluate(path, scoring_config))
                    self.assertEqual(summary["coverage"], 1, score_path.read_text(encoding="utf-8"))
                    digest, count = file_digest(score_path), len(self.requests)
                    asyncio.run(evaluate(path, scoring_config))
                    self.assertEqual(file_digest(score_path), digest)
                    self.assertEqual(len(self.requests), count)
                    report["cases"].append({"method": method, "benchmark": kind, "status": "passed", "questions": len(payload["data"])})
        self.assertEqual(len(report["cases"]), 15)
        atomic_json(ROOT / "results/logs/omnimemeval/simulated_validation.json", report)

    def test_original_locomo_main_pipeline_is_unchanged(self):
        dataset = self.locomo_config()
        raw = json.loads(Path(dataset["test_files"]).read_text())
        raw[0]["conversation"]["session_1"][0]["text"] = "History note. " * 180 + "Alice likes tea."
        Path(dataset["test_files"]).write_text(json.dumps(raw))
        dataset.update(dataset="LoCoMo", sub_dataset="locomo_qa", locomo_categories=[1, 2, 3, 4], max_test_samples=1, context_max_length=100000, generation_max_length=32, seed=42, shots=0, debug=False, tag="legacy_mock")
        config = yaml.safe_load((ROOT / "config/reference_long_context_agent.yaml").read_text(encoding="utf-8"))
        config.update(base_url="http://offline.invalid/v1", output_dir="./results/outputs/legacy")
        agent_file, dataset_file = self.root / "agent.yaml", self.root / "data.yaml"
        agent_file.write_text(yaml.safe_dump(config))
        dataset_file.write_text(yaml.safe_dump(dataset))
        artifacts = self.root / "legacy"
        self.invoke_main(agent_file, dataset_file, artifacts)
        paths = list((artifacts / "outputs/legacy/LoCoMo").glob("*_results.json"))
        self.assertEqual(len(paths), 1)
        payload = json.loads(paths[0].read_text())
        self.assertNotIn("protocol", payload)
        self.assertIn("time_cost", payload)
        self.assertEqual(len(payload["data"]), 2)
        count = len(self.requests)
        self.invoke_main(agent_file, dataset_file, artifacts)
        self.assertEqual(len(self.requests), count)


class OriginalInterfaceTests(DatasetFixtures, unittest.TestCase):
    def test_public_loader_and_conversation_creator_contract(self):
        from benchmark.memoryagentbench.loader import load_eval_data
        from utils.conversation_creator import ConversationCreator
        config = self.locomo_config()
        config.update(max_test_samples=0, context_max_length=100000)
        loaded = load_eval_data(config)
        self.assertIsInstance(loaded, dict)
        sample = next(iter(loaded["data"]))
        self.assertTrue({"context_chunks", "questions", "answers", "qa_pair_ids", "eval_metadata"} <= sample.keys())
        self.assertIsInstance(sample["questions"][0], str)
        creator = ConversationCreator({"agent_name": "Long_context_agent"}, config)
        pairs = creator.get_query_and_answers()[0]
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0][1], "GOLD_ONLY")
        self.assertEqual(pairs[0][2], "conv-26:q0")
        self.assertEqual(len(pairs[0]), 4)
        self.assertIn("text", creator.get_chunks()[0][0])

    def test_original_score_outputs_match_head(self):
        from utils import eval_other_utils as current
        original = subprocess.check_output(["git", "show", "HEAD:utils/eval_other_utils.py"], cwd=ROOT).decode("utf-8")
        namespace = {"__name__": "legacy_metrics_reference"}
        exec(compile(original, "HEAD:utils/eval_other_utils.py", "exec"), namespace)
        for dataset, sub, output, answer in (("LoCoMo", "locomo_qa", "Answer: Tea", "Tea"), ("LongBench", "longbench_test", "B", "B"), ("MemBench", "membench_simple", "B", "B"), ("Accurate_Retrieval", "longmemeval_s*", "Tea", "Tea"), ("Accurate_Retrieval", "eventqa_full", "Tea", ["Tea"])):
            config = {"dataset": dataset, "sub_dataset": sub}
            with self.subTest(dataset=dataset, sub=sub):
                self.assertEqual(current.post_process({"output": output}, answer, config), namespace["post_process"]({"output": output}, answer, config))

    def test_original_tracked_paths_and_agent_configs_are_preserved(self):
        changed = subprocess.check_output(["git", "diff", "--name-status", "HEAD"], cwd=ROOT).decode()
        self.assertFalse(any(line.startswith(("D\t", "R")) for line in changed.splitlines()))
        tracked_configs = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "HEAD", "config"], cwd=ROOT).decode().splitlines()
        self.assertEqual({str(path.relative_to(ROOT)).replace("\\", "/") for path in (ROOT / "config").glob("*.yaml")}, set(tracked_configs))
        for name in tracked_configs:
            original = subprocess.check_output(["git", "show", "HEAD:" + name], cwd=ROOT)
            # Git's Windows checkout may normalize newlines.
            self.assertEqual((ROOT / name).read_bytes().replace(b"\r\n", b"\n"), original.replace(b"\r\n", b"\n"))


if __name__ == "__main__":
    unittest.main()
