"""MemoryData side of the REMem integration (runs in the MemoryData environment).

REMem (Shu et al., ICLR 2026, https://github.com/intuit-ai-research/REMem) pins
openai<2, numpy<2 and torch, so the unmodified native code runs in its own conda
environment behind ``remem_worker.py``. REMem indexes a whole corpus at once
(``ReMem.index(docs)``): chunks are buffered while MemoryData memorizes and
indexed in ``save()``. Native files live in ``<state dir>/native``.
"""

from __future__ import annotations

from pathlib import Path

from methods.hindsight.hindsight_adapter import normalize_query_timestamp
from methods.worker_bridge import (BufferedWorkerAdapter, WorkerProcess, resolve_worker_python,
                                   worker_environment)
from utils.repo_paths import portable

WORKER_SCRIPT = Path(__file__).resolve().parent / "remem_worker.py"
ANSWER_MODES = {"memorydata", "native"}

# Upstream BaseConfig fields MemoryData sets. Values follow the authors'
# examples/locomo.py unless remem_native_config overrides them.
NATIVE_DEFAULTS = {
    "extract_method": "episodic_gist",
    "agent_fixed_tools": False,
    "agent_max_steps": 3,
    "qa_top_k": 10,
    "linking_top_k": 5,
    "retrieval_top_k": 200,
    "synonymy_edge_sim_threshold": 0.8,
    "graph_type": "facts_and_sim_passage_node_unidirectional",
    "embedding_batch_size": 16,
    "max_new_tokens": None,
    "temperature": 0.0,
    "qa_passage_prefix": "",
    "qa_prompt_template": "rag_qa_unified",
    "rerank_dspy_file": "src/remem/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
}
# Settings that only act at query time; changing them reuses a built index.
QUERY_TIME_KEYS = {"agent_fixed_tools", "agent_max_steps", "qa_top_k", "linking_top_k", "retrieval_top_k",
                   "qa_prompt_template", "rerank_dspy_file", "qa_passage_prefix"}


def native_config(agent_config):
    overrides = dict(agent_config.get("remem_native_config") or {})
    unknown = set(overrides) - set(NATIVE_DEFAULTS)
    if unknown:
        raise ValueError(f"Unsupported remem_native_config keys: {sorted(unknown)}")
    config = {**NATIVE_DEFAULTS, **overrides}
    if config["extract_method"] != "episodic_gist":
        raise ValueError("Only REMem's episodic_gist memory (the paper's method) is integrated")
    config["llm_name"] = str(agent_config.get("remem_llm_model") or agent_config.get("model") or "").strip()
    config["embedding_model_name"] = str(agent_config.get("remem_embedding_model")
                                         or agent_config.get("embedding_model") or "").strip()
    if not config["llm_name"] or not config["embedding_model_name"]:
        raise ValueError("REMem needs an LLM model and an embedding model")
    if "text-embedding" not in config["embedding_model_name"]:
        raise ValueError("This integration wires REMem to an OpenAI-compatible text-embedding-* model")
    return config


class RememAdapter(BufferedWorkerAdapter):
    PROTOCOL = "memorydata_remem_worker_v1"
    STATE_FILE = "remem_state.json"
    LABEL = "REMem"

    def __init__(self, agent_config, dataset_config, state_dir, *, worker_factory=None):
        self.config = native_config(agent_config)
        self.top_k = int(agent_config.get("retrieve_num", 10))
        self.answer_mode = str(agent_config.get("remem_answer_mode", "memorydata"))
        self.timeout = float(agent_config.get("remem_worker_timeout_seconds", 7200))
        if self.answer_mode not in ANSWER_MODES or self.top_k <= 0 or self.timeout <= 0:
            raise ValueError("remem_answer_mode must be memorydata or native; retrieve_num and timeout must be positive")
        dataset = {k: v for k, v in dataset_config.items() if k != "max_test_queries"}
        # The identity covers what determines the stored memory: native index
        # settings and models, data, history and the repository-relative state dir.
        super().__init__(agent_config, state_dir, {
            "native": {k: v for k, v in self.config.items() if k not in QUERY_TIME_KEYS},
            "dataset": portable(dataset),
            "history": agent_config.get("_memory_history_fingerprint"),
            "state_dir": portable(str(Path(state_dir).resolve())),
        }, worker_factory=worker_factory)
        self.indexed = False

    def start_worker(self):
        python = resolve_worker_python(self.agent_config, "remem_python", "remem_python_env")
        return WorkerProcess(python, WORKER_SCRIPT, cwd=self.native_dir, log_dir=self.state_dir,
                             env=worker_environment(self.agent_config), timeout=self.timeout)

    def _worker(self):
        return self.open_worker(working_dir=str((self.native_dir / "remem_index").resolve()),
                                config=self.config, docs=self.documents())

    def save(self):
        if self.state.get("ready"):
            return
        if not self.state["documents"]:
            raise RuntimeError("REMem received no history to index")
        stats = self._worker().call("index")
        self.indexed = True
        self.mark_ready(stats)

    def worker_lost(self):
        self.indexed = False

    def load(self):
        self.require_ready()
        # Without its graph, upstream index() would silently re-extract everything.
        if not (self.native_dir / "remem_index" / "graph.pkl").exists():
            raise RuntimeError("Saved REMem native index is missing; rerun with --force")

    def retrieve(self, query, *, query_timestamp=None, native_answer=False):
        """Run REMem's agentic retrieval; returns (native agent answer, visited passages with source IDs).

        The agent always produces its answer while retrieving (output_answer),
        so ``native_answer`` changes nothing here; the caller decides which answer to use.
        """
        self.require_ready()
        worker = self._worker()
        if not self.indexed:
            worker.call("index")  # reopens the saved index; upstream skips re-extraction
            self.indexed = True
        result = worker.call("query", query=query, question_date=normalize_query_timestamp(query_timestamp))
        passages = []
        for passage, docs in zip(result.get("passages") or [], result.get("source_docs") or []):
            text = str(passage.get("text") or "").strip()
            if text:
                passages.append({"text": text, "type": passage.get("type"), "source_ids": self.source_ids_for(docs)})
        # REMem bounds its own context (qa_top_k per tool call, agent_max_steps);
        # every visited passage is kept, in visit order, as the native answer saw it.
        return str(result.get("answer") or ""), passages
