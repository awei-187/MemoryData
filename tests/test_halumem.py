"""HaluMem loader, session-interleaved execution, Hindsight settle and QA judge; no model inference."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import yaml

from benchmark.halumem.loader import load_halumem_eval_data, parse_halumem_time

ROOT = Path(__file__).resolve().parents[1]
SECRET = "GOLD_ONLY_SECRET"


def turn(role, text, ts):
    return {"role": role, "content": text, "timestamp": ts, "dialogue_turn": 0}


def session(index, turns, questions=(), generated=False):
    value = {
        "start_time": f"Jan 0{index + 1}, 2026, 10:00:00", "end_time": f"Jan 0{index + 1}, 2026, 11:00:00",
        "dialogue": turns, "memory_points": [{"memory_content": SECRET + " memory", "is_update": "False"}],
        "questions": list(questions), "dialogue_turn_num": len(turns),
    }
    if generated:
        value["is_generated_qa_session"] = True
    return value


def question(text, answer, qtype="Basic Fact Recall"):
    return {"question": text, "answer": answer, "difficulty": "easy", "question_type": qtype,
            "evidence": [{"memory_content": f"{SECRET} evidence for {text}", "memory_type": "Persona Memory"}]}


def synthetic_users():
    long_text = "I train for marathons every weekend and keep detailed logs. " * 6
    alice = {
        "uuid": "user-a", "persona_info": f"[Recorded] Name: Alice Smith; Gender: Female; Secret: {SECRET}",
        "sessions": [
            session(0, [turn("user", "I drink green tea every morning.", "Jan 01, 2026, 10:00:00"),
                        turn("assistant", "Noted, green tea.", "Jan 01, 2026, 10:00:05")] * 20,
                    [question("What does Alice drink?", "Green tea")]),
            session(1, [turn("user", long_text, "Jan 02, 2026, 10:00:00"),
                        turn("assistant", long_text, "Jan 02, 2026, 10:01:00"),
                        turn("user", "I moved to Lisbon.", "Jan 02, 2026, 10:02:00")]),
            session(2, [turn("user", "Now I prefer black coffee.", "Jan 03, 2026, 10:00:00")] * 30,
                    [question("What does Alice drink now?", "Black coffee", "Dynamic Update"),
                     question("Where does Alice live?", "Lisbon", "Multi-hop Inference")]),
            session(3, [turn("user", "Generated filler.", "Jan 04, 2026, 10:00:00")] * 40,
                    [question("Generated question?", "x")], generated=True),
        ],
    }
    bob = {"uuid": "user-b", "persona_info": "Name: Bob; Gender: Male;",
           "sessions": [session(0, [turn("user", "I play chess daily. " * 20, "Jan 01, 2026, 09:00:00")] * 10,
                                [question("What game?", "Chess")])]}
    return [alice, bob]


def write_dataset(root, users=None, **overrides):
    data = root / "halumem.jsonl"
    data.write_text("\n".join(json.dumps(u) for u in (users or synthetic_users())) + "\n", encoding="utf-8")
    config = yaml.safe_load((ROOT / "benchmark/halumem/config/HaluMem_medium.yaml").read_text(encoding="utf-8"))
    config.update(test_files=str(data), chunk_size=600, **overrides)
    path = root / "dataset.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config, path


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config, _ = write_dataset(self.root)

    def test_chunks_follow_sessions_and_questions_follow_their_session(self):
        alice = load_halumem_eval_data(self.config)["data"][0]
        chunks = alice["context_chunks"]
        self.assertTrue(all(len(c["text"]) <= 600 or c["text"].count("\n") == 1 for c in chunks))
        sessions = [c["session_index"] for c in chunks]
        self.assertEqual(sessions, sorted(sessions))
        self.assertGreater(sessions.count(1), 1, "long session must be split")
        for c in chunks:  # every chunk carries its own session header
            self.assertTrue(c["text"].startswith(f"Session {c['session_index'] + 1} ("))
        ask = [m["ask_after_chunk"] for m in alice["eval_metadata"]]
        self.assertEqual(ask[0], sessions.count(0))
        self.assertEqual(ask[1], ask[2])
        self.assertEqual(ask[1], len([s for s in sessions if s <= 2]))
        self.assertNotIn("Generated question?", alice["questions"])  # generated QA sessions are not evaluated
        self.assertEqual(sessions[-1], 3)  # ...but their dialogue is still memorized
        self.assertEqual(alice["eval_metadata"][0]["question_date"], "2026-01-01T11:00:00")
        self.assertEqual(chunks[0]["timestamp"], "2026-01-01T10:00:00")

    def test_only_dialogue_enters_memory(self):
        alice = load_halumem_eval_data(self.config)["data"][0]
        text = "\n".join(c["text"] for c in alice["context_chunks"])
        self.assertIn("Alice Smith: I drink green tea", text)
        self.assertIn("Assistant: Noted", text)
        self.assertNotIn(SECRET, text)
        self.assertNotIn("Female", text)
        self.assertEqual(alice["eval_metadata"][0]["evidence"][0]["memory_type"], "Persona Memory")

    def test_user_selection_and_strict_time_format(self):
        only_bob = load_halumem_eval_data({**self.config, "halumem_user_ids": "user-b"})["data"]
        self.assertEqual([u["sample_id"] for u in only_bob], ["user-b"])
        self.assertEqual(len(load_halumem_eval_data({**self.config, "max_test_samples": 1})["data"]), 1)
        with self.assertRaisesRegex(ValueError, "not found"):
            load_halumem_eval_data({**self.config, "halumem_user_ids": ["missing"]})
        with self.assertRaisesRegex(ValueError, "Unsupported HaluMem timestamp"):
            parse_halumem_time("2026-01-01 10:00")

    def test_templates_and_conversation_creator(self):
        from benchmark.memoryagentbench.prompts.benchmark_templates import get_template
        from utils.conversation_creator import ConversationCreator
        for agent in ["Long_context_agent_x", "rag_bm25", "Agentic_memory_hindsight", "MemOS_x", "memagent_x"]:
            self.assertIn("{question}", get_template("halumem_medium", "query", agent))
        answer = get_template("halumem_medium", "memory_answer", "Agentic_memory_hindsight")
        self.assertIn("prioritize the most recent memory", answer)
        self.assertIn("M1", answer.format(memories="M1", question="Q"))
        creator = ConversationCreator({"agent_name": "Agentic_memory_hindsight", "agent_chunk_size": 600}, self.config)
        pairs = creator.get_query_and_answers()
        self.assertEqual([len(p) for p in pairs], [3, 1])
        self.assertIn("What does Alice drink?", pairs[0][0][0])
        self.assertEqual(pairs[0][0][2], "user-a:s0:q0")


class FakeAgent:
    """Records the order of memorize / prepare / ask calls made by the interleaved runner."""

    def __init__(self, log, fail_on=()):
        self.log = log
        self.fail_on = set(fail_on)

    def send_message(self, message, memorizing=False, query_id=None, context_id=None, eval_metadata=None):
        if memorizing:
            self.log.append(("memorize", message["chunk_id"]))
            return "Memorized"
        question = eval_metadata["question"]
        self.log.append(("ask", question))
        if question in self.fail_on:
            raise RuntimeError("transient answer failure")
        return {"output": "answer", "input_len": 1, "output_len": 1, "memory_construction_time": 0, "query_time_len": 0}

    def prepare_for_queries(self):
        self.log.append(("prepare",))

    def finish_query_batch(self):
        self.log.append(("finish",))

    def save_agent(self):
        self.log.append(("save",))


class InterleavedRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config, self.dataset_path = write_dataset(self.root)
        agent = {"agent_name": "Long_context_agent_fake", "model": "offline", "output_dir": str(self.root / "out"),
                 "input_length_limit": 100000, "buffer_length": 10, "temperature": 0}
        self.agent_path = self.root / "agent.yaml"
        self.agent_path.write_text(yaml.safe_dump(agent), encoding="utf-8")
        self.log, self.creations, self.fail_on = [], [], set()

    def run_main(self, *flags):
        import main

        def create(agent_config, dataset_config, folder, chunks, context_index):
            self.creations.append(context_index)
            self.log.append(("create", context_index))
            return FakeAgent(self.log, self.fail_on)

        args = ["main.py", "--agent_config", str(self.agent_path), "--dataset_config", str(self.dataset_path),
                "--artifact_root", str(self.root / "runs"), *flags]
        with patch.object(sys, "argv", args), patch.object(main, "create_interleaved_agent", create):
            main.main()
        results = list((self.root / "out").rglob("*_results.json"))
        self.assertEqual(len(results), 1)
        return json.loads(results[0].read_text(encoding="utf-8")), results[0]

    def test_questions_are_asked_right_after_their_session(self):
        payload, _ = self.run_main()
        users = load_halumem_eval_data(self.config)["data"]
        alice_chunks = [c["chunk_id"] for c in users[0]["context_chunks"]]
        first, second = users[0]["eval_metadata"][0]["ask_after_chunk"], users[0]["eval_metadata"][1]["ask_after_chunk"]
        expected = ([("create", 0)] + [("memorize", c) for c in alice_chunks[:first]]
                    + [("prepare",), ("ask", "What does Alice drink?"), ("finish",)]
                    + [("memorize", c) for c in alice_chunks[first:second]]
                    + [("prepare",), ("ask", "What does Alice drink now?"), ("ask", "Where does Alice live?"), ("finish",)]
                    + [("memorize", c) for c in alice_chunks[second:]] + [("save",)])
        self.assertEqual(self.log[:len(expected)], expected)
        self.assertEqual(self.log[len(expected)], ("create", 1))
        self.assertEqual(len(payload["data"]), 4)
        self.assertIn("interleaved", Path(_).name)

    def test_unfinished_context_is_rebuilt_and_only_missing_questions_are_asked(self):
        self.fail_on = {"What does Alice drink now?"}
        payload, _ = self.run_main()
        self.assertEqual(sum(row.get("status") == "failed" for row in payload["data"]), 1)
        self.log.clear()
        self.fail_on = set()
        payload, _ = self.run_main("--retry_failed_queries")
        asked = [entry[1] for entry in self.log if entry[0] == "ask"]
        self.assertEqual(asked, ["What does Alice drink now?"])
        self.assertEqual(self.log[0], ("create", 0))  # rebuilt from the first chunk
        memorized = [entry for entry in self.log if entry[0] == "memorize"]
        self.assertEqual(len(memorized), len(load_halumem_eval_data(self.config)["data"][0]["context_chunks"]))
        self.assertNotIn(("create", 1), self.log)  # completed context is skipped
        self.assertFalse(any(row.get("status") == "failed" for row in payload["data"]))

    def test_query_limit_stops_mid_context(self):
        self.run_main("--max_test_queries_ablation", "2")
        asked = [entry[1] for entry in self.log if entry[0] == "ask"]
        self.assertEqual(asked, ["What does Alice drink?", "What does Alice drink now?"])
        self.assertNotIn(("save",), self.log)

    def test_unsupported_agents_fail_before_any_work(self):
        from utils.initialization import create_interleaved_agent
        with self.assertRaisesRegex(ValueError, "MemoRAG"):
            create_interleaved_agent({"agent_name": "Agentic_memory_memo_rag"}, self.config, str(self.root / "x"), [], 0)
        for name in ["Agentic_memory_remem", "Agentic_memory_tsm"]:
            (self.root / "x").mkdir(exist_ok=True)
            with self.assertRaisesRegex(ValueError, "REMem and TSM"):
                create_interleaved_agent({"agent_name": name}, self.config, str(self.root / "x"), [], 0)
            self.assertTrue((self.root / "x").exists())  # refused before any reset
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            create_interleaved_agent({"agent_name": "Structure_rag_mem0", "mem0_checkpoint_enabled": True},
                                     self.config, str(self.root / "x"), [], 0)


class OfflineAnswerClient:
    def __init__(self):
        self.messages = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.messages.append(kwargs["messages"])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
                               usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))


class AgentInterleavingTests(unittest.TestCase):
    def test_bm25_rag_sees_history_memorized_after_a_question(self):
        from test_hindsight import CharacterTokenizer
        from utils.agent import AgentWrapper
        dataset = yaml.safe_load((ROOT / "benchmark/halumem/config/HaluMem_medium.yaml").read_text(encoding="utf-8"))
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        config = {"agent_name": "rag_bm25", "model": "offline", "output_dir": temporary.name, "retrieve_num": 1,
                  "artifact_root": temporary.name, "input_length_limit": 100000, "buffer_length": 10, "temperature": 0}
        client = OfflineAnswerClient()
        with patch.object(AgentWrapper, "_initialize_tokenizer", return_value=CharacterTokenizer()), \
             patch.object(AgentWrapper, "_create_oai_client", return_value=client):
            agent = AgentWrapper(config, dataset, str(Path(temporary.name) / "agent"))
            agent.send_message({"text": "Alice drinks green tea."}, memorizing=True, context_id=0)
            agent.prepare_for_queries()
            agent.send_message("What does Alice drink? tea", query_id=0, context_id=0)
            agent.send_message({"text": "Alice now lives in Lisbon harbour district."}, memorizing=True, context_id=0)
            agent.send_message("Where does Alice live? Lisbon harbour", query_id=1, context_id=0)
        first, second = (json.dumps(messages) for messages in client.messages)
        self.assertIn("green tea", first)
        self.assertIn("Lisbon harbour district", second)  # rebuilt index includes the later chunk

    def test_buffered_memories_are_flushed_at_question_points(self):
        from utils.agent import AgentWrapper
        for name, attribute, method in [("Agentic_memory_simplemem", "simplemem", "flush_for_queries"),
                                        ("Agentic_memory_memochat", "memochat", "save"),
                                        ("Agentic_memory_hindsight", "hindsight", "settle")]:
            agent = AgentWrapper.__new__(AgentWrapper)
            agent.agent_name = name
            stub = SimpleNamespace(**{method: unittest.mock.Mock()})
            setattr(agent, attribute, stub)
            agent.prepare_for_queries()
            getattr(stub, method).assert_called_once()


class HindsightSettleTests(unittest.TestCase):
    def setUp(self):
        from test_hindsight import Backend
        from methods.hindsight.hindsight_adapter import HindsightAdapter
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.backend = Backend()
        environment = patch.dict(os.environ, {"HINDSIGHT_API_URL": "", "HINDSIGHT_API_KEY": ""})
        environment.start()
        self.addCleanup(environment.stop)
        self.adapter = HindsightAdapter({"hindsight_base_url": "http://offline.invalid", "retrieve_num": 5},
                                        {"dataset": "HaluMem", "sub_dataset": "halumem_medium"},
                                        Path(self.temporary.name) / "state",
                                        transport=httpx.MockTransport(self.backend.transport))

    def test_settle_allows_questions_between_retains_until_final_save(self):
        a = self.adapter
        a.add_chunk("Session 1: Alice drinks tea", timestamp="2026-01-01T10:00:00")
        with self.assertRaisesRegex(RuntimeError, "settle"):
            a.retrieve("drink?")
        a.settle()
        self.assertEqual([r["text"] for r in a.retrieve("drink?", query_timestamp="2026-01-01T11:00:00")],
                         ["Session 1: Alice drinks tea"])
        self.assertEqual(self.backend.calls[-1][2]["query_timestamp"], "2026-01-01T11:00:00")
        a.add_chunk("Session 2: Alice moved to Lisbon")
        with self.assertRaisesRegex(RuntimeError, "settle"):
            a.retrieve("where?")
        self.backend.stats["pending_consolidation"] = 1
        with patch("methods.hindsight.hindsight_adapter.time.monotonic", side_effect=[0, 301]), \
             self.assertRaises(TimeoutError):
            a.settle()
        self.assertFalse(a.state["ready"])
        self.backend.stats["pending_consolidation"] = 0
        a.settle()
        self.assertEqual(len(a.retrieve("where?")), 2)
        self.assertFalse(a.state["ready"])
        self.assertNotIn("DELETE", [m for m, _, _ in self.backend.calls])
        a.save()
        self.assertTrue(a.state["ready"])
        with self.assertRaisesRegex(RuntimeError, "append"):
            a.add_chunk("Session 3")


class HindsightInterleavedPipelineTests(unittest.TestCase):
    def test_main_runs_hindsight_session_by_session_over_http(self):
        import main
        from test_hindsight import Backend, CharacterTokenizer, serve_over_loopback
        from utils.agent import AgentWrapper
        from methods.hindsight.hindsight_adapter import STATE_FILE

        backend = Backend()
        endpoint = serve_over_loopback(self, backend)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            users = synthetic_users()[:1]
            dataset, dataset_path = write_dataset(root, users)
            config = yaml.safe_load((ROOT / "config/hybrid_hindsight.yaml").read_text(encoding="utf-8"))
            config.update(base_url=endpoint + "/v1", hindsight_base_url=endpoint, model="offline-model",
                          model_api_trust_env=False, output_dir=str(root / "out"), agent_chunk_size=600)
            (root / "agent.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
            args = ["main.py", "--agent_config", str(root / "agent.yaml"), "--dataset_config", str(dataset_path),
                    "--artifact_root", str(root / "runs")]
            with patch.dict(os.environ, {"OPENAI_API_KEY": "offline", "HINDSIGHT_API_URL": ""}), \
                 patch.object(AgentWrapper, "_initialize_tokenizer", return_value=CharacterTokenizer()), \
                 patch.object(sys, "argv", args):
                main.main()

            steps = []
            for method, path, body in backend.calls:
                step = ("retain" if path.endswith("/memories") else "recall" if path.endswith("/recall")
                        else "answer" if path.endswith("/chat/completions") else "stats" if "/stats" in path
                        else method.lower())
                if not steps or steps[-1][0] != step:
                    steps.append([step, 0])
                steps[-1][1] += 1
            # retain s0 -> settle -> recall+answer -> retain s1..s2 -> settle -> 2 x (recall+answer) -> retain s3 -> final save
            self.assertEqual([s for s, _ in steps], ["put", "retain", "stats", "recall", "answer", "retain", "stats",
                                                     "recall", "answer", "recall", "answer", "retain", "stats"])
            loaded = load_halumem_eval_data(dataset)["data"][0]
            first = loaded["eval_metadata"][0]["ask_after_chunk"]
            self.assertEqual(steps[1][1], first)
            recalls = [body for _, path, body in backend.calls if path.endswith("/recall")]
            self.assertEqual(recalls[0]["query_timestamp"], "2026-01-01T11:00:00")
            self.assertEqual(recalls[1]["query_timestamp"], "2026-01-03T11:00:00")
            retains = [body["items"][0] for _, path, body in backend.calls if path.endswith("/memories")]
            self.assertEqual(retains[0]["timestamp"], "2026-01-01T10:00:00")
            self.assertFalse(any(SECRET in item["content"] for item in retains))
            state = json.loads(next((root / "runs").rglob(STATE_FILE)).read_text(encoding="utf-8"))
            self.assertTrue(state["ready"])
            self.assertEqual(len(state["documents"]), len(loaded["context_chunks"]))
            results = json.loads(next((root / "out").rglob("*_results.json")).read_text(encoding="utf-8"))
            self.assertEqual([row["output"] for row in results["data"]], ["Tea"] * 3)

    def test_settle_timeout_at_a_question_point_resumes_the_same_bank(self):
        import main
        from test_hindsight import Backend, CharacterTokenizer, serve_over_loopback
        from utils.agent import AgentWrapper
        from utils.initialization import resume_interleaved_agent
        from methods.hindsight.hindsight_adapter import STATE_FILE

        class StallingBackend(Backend):
            """Reports native document counts; consolidation stalls once the bank holds `stall_at` documents."""
            stall_at = None

            def respond(self, method, path, body):
                status, value = super().respond(method, path, body)
                if method == "GET" and "/stats" in path:
                    documents = len(next(iter(self.banks.values()), {}))
                    value.update(total_documents=documents,
                                 pending_consolidation=int(self.stall_at is not None and documents >= self.stall_at))
                return status, value

        backend = StallingBackend()
        endpoint = serve_over_loopback(self, backend)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, dataset_path = write_dataset(root, synthetic_users()[:1])
            loaded = load_halumem_eval_data(dataset)["data"][0]
            chunks = loaded["context_chunks"]
            first, second = (m["ask_after_chunk"] for m in loaded["eval_metadata"][:2])
            config = yaml.safe_load((ROOT / "config/hybrid_hindsight.yaml").read_text(encoding="utf-8"))
            config.update(base_url=endpoint + "/v1", hindsight_base_url=endpoint, model="offline-model",
                          model_api_trust_env=False, output_dir=str(root / "out"), agent_chunk_size=600,
                          hindsight_settle_timeout_seconds=0.01)
            agent_path = root / "agent.yaml"
            agent_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            args = ["main.py", "--agent_config", str(agent_path), "--dataset_config", str(dataset_path),
                    "--artifact_root", str(root / "runs")]
            environment = {"OPENAI_API_KEY": "offline", "HINDSIGHT_API_URL": ""}
            with patch.dict(os.environ, environment), \
                 patch.object(AgentWrapper, "_initialize_tokenizer", return_value=CharacterTokenizer()), \
                 patch.object(sys, "argv", args):
                backend.stall_at = second  # the second question point never settles in time
                with self.assertRaises(TimeoutError):
                    main.main()
                state_path = next((root / "runs").rglob(STATE_FILE))
                stalled = json.loads(state_path.read_text(encoding="utf-8"))
                self.assertEqual(len(stalled["documents"]), second)

                # A question before the retained prefix still unanswered: the bank cannot be reused.
                self.assertIsNone(resume_interleaved_agent(config, dataset, str(state_path.parent), chunks, [first]))

                backend.stall_at = None
                calls = len(backend.calls)
                main.main()
            rerun = backend.calls[calls:]
            self.assertNotIn("DELETE", [m for m, _, _ in rerun])
            self.assertNotIn("PUT", [m for m, _, _ in rerun])
            retained = [body["items"][0]["document_id"] for _, path, body in rerun if path.endswith("/memories")]
            self.assertEqual(len(retained), len(chunks) - second)  # only the history after the stalled point
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["bank_id"], stalled["bank_id"])
            self.assertTrue(state["ready"])
            self.assertEqual(len(backend.banks[state["bank_id"]]), len(chunks))
            self.assertEqual(list(state["documents"])[:second], list(stalled["documents"]))
            self.assertEqual(sum(path.endswith("/recall") for _, path, _ in rerun), 2)  # only the unanswered questions
            results = json.loads(next((root / "out").rglob("*_results.json")).read_text(encoding="utf-8"))
            self.assertEqual([row["output"] for row in results["data"]], ["Tea"] * 3)


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_results(self, rows):
        path = self.root / "halumem_results.json"
        path.write_text(json.dumps({"data": rows}), encoding="utf-8")
        return path

    @staticmethod
    def row(qid, output="answer", qtype="Basic Fact Recall", difficulty="easy", status=None):
        value = {"qa_pair_id": qid, "answer": "ref", "output": output, "query": "formatted",
                 "eval_metadata": {"dataset": "halumem_qa", "qa_pair_id": qid, "question": f"question {qid}",
                                   "question_type": qtype, "difficulty": difficulty,
                                   "evidence": [{"memory_content": "evidence " + qid}]}}
        if status:
            value.update(status=status, output="")
        return value

    def client(self, replies):
        calls = []

        def create(**kwargs):
            prompt = kwargs["messages"][0]["content"]
            calls.append((kwargs["model"], kwargs["temperature"], prompt))
            qid = next(q for q in replies if f"question {q}\n" in prompt)
            reply = replies[qid].pop(0) if isinstance(replies[qid], list) else replies[qid]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls

    @staticmethod
    def reply(label):
        return f'Reasoning first.\n```json\n{{"reasoning": "r", "evaluation_result": "{label}"}}\n```'

    def test_official_prompt_labels_denominators_and_breakdowns(self):
        from evaluation.halumem.halumem_judge import judge_result_file, parse_judge_response
        path = self.write_results([
            self.row("a"), self.row("b", qtype="Dynamic Update", difficulty="hard"), self.row("c"),
            self.row("d", status="failed"), {"qa_pair_id": "x", "eval_metadata": {"dataset": "locomo_qa"}},
        ])
        client, calls = self.client({
            "a": self.reply("Correct"),
            "b": ["not json", self.reply("Hallucination")],  # format errors are retried
            "c": [self.reply("Correct | Hallucination | Omission")] * 2,  # copied template label stays invalid
        })
        sidecar = judge_result_file(path, client=client, judge_model="judge-x", max_attempts=2, sleep=lambda s: None)
        self.assertTrue(all(model == "judge-x" and temperature == 0 for model, temperature, _ in calls))
        self.assertIn("Key Memory Points", calls[0][2])
        self.assertIn("evidence a", next(p for _, _, p in calls if "question a\n" in p))
        overall = sidecar["summary"]["overall"]
        self.assertEqual((overall["total"], overall["valid"], overall["answer_failed"], overall["judge_failed"]), (4, 2, 1, 1))
        self.assertEqual(overall["correct_ratio(all)"], 0.25)
        self.assertEqual(overall["correct_ratio(valid)"], 0.5)
        self.assertEqual(sidecar["summary"]["by_question_type"]["Dynamic Update"]["hallucination_count"], 1)
        self.assertEqual(sidecar["summary"]["by_difficulty"]["hard"]["total"], 1)
        self.assertTrue(Path(str(path).replace(".json", "_halumem_judge.json")).exists())
        with self.assertRaises(ValueError):
            parse_judge_response('```json\n{"evaluation_result": "Other"}\n```')

        # Rerun reuses valid labels and re-judges only failed or changed rows.
        client, calls = self.client({"c": self.reply("Omission"), "a": self.reply("Hallucination")})
        sidecar = judge_result_file(path, client=client, judge_model="judge-x", sleep=lambda s: None)
        self.assertEqual(len(calls), 1)
        self.assertEqual(sidecar["summary"]["overall"]["omission_count"], 1)
        rows = json.loads(path.read_text())["data"]
        rows[0]["output"] = "changed answer"
        path.write_text(json.dumps({"data": rows}))
        client, calls = self.client({"a": self.reply("Hallucination")})
        judge_result_file(path, client=client, judge_model="judge-x", sleep=lambda s: None)
        self.assertEqual(len(calls), 1)
        client, calls = self.client({q: self.reply("Correct") for q in "abc"})
        judge_result_file(path, client=client, judge_model="other-judge", sleep=lambda s: None)
        self.assertEqual(len(calls), 3)  # a different judge model never reuses labels

    def test_default_judge_model_is_gpt_4o_unless_configured(self):
        from evaluation.halumem.halumem_judge import resolve_judge_model
        with patch.dict(os.environ, {"HALUMEM_JUDGE_MODEL": ""}):
            self.assertEqual(resolve_judge_model(), "gpt-4o")
        with patch.dict(os.environ, {"HALUMEM_JUDGE_MODEL": "judge-env"}):
            self.assertEqual(resolve_judge_model(), "judge-env")
            self.assertEqual(resolve_judge_model("explicit"), "explicit")


if __name__ == "__main__":
    unittest.main()
