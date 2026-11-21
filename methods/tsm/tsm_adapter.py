"""MemoryData side of the TSM integration (runs in the MemoryData environment).

TSM (Su et al., Findings of ACL 2026) is integrated from the release at
https://github.com/cosmicexotic/TSM, which is published by the first author's
account and cites the paper but is not linked from the paper itself. It pins
graphiti-core and needs Neo4j, so the unmodified code runs in its own conda
environment behind ``tsm_worker.py``. Like the authors' pipeline, a whole
history is built at once: chunks are buffered while MemoryData memorizes and
built in ``save()``. Each context owns one Neo4j group ``md_tsm_<uuid>``.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

from methods.hindsight.hindsight_adapter import query_timestamp_or_none
from methods.worker_bridge import (BufferedWorkerAdapter, WorkerProcess, resolve_worker_python,
                                   worker_environment)
from utils.repo_paths import portable

WORKER_SCRIPT = Path(__file__).resolve().parent / "tsm_worker.py"
ANSWER_MODES = {"memorydata", "native"}
SUMMARY_MARKER = "--- Snapshot Summaries ---"
CHUNK_MARKER = "--- Retrieved Chunks ---"

# Settings of the authors' configs/locomo/tsm.yaml and configs/longmemeval/tsm.yaml
# (identical apart from dataset names and paths).
RETRIEVER_DEFAULTS = {
    "enable_dense_retrieval": True, "prebuilt_index": True, "embedder": "PublicEmbedder",
    "embedding_batch_size": 128, "shortest_path_k": 4, "max_chunk_setting": 20,
    "enable_temporal_filter": True, "chunks_only": False, "use_gpu_faiss": False,
}
BUILD_DEFAULTS = {
    "cluster": {"length": 1000, "overlap": 200, "merge_num": 3},
    "entity_clustering": {"enable": True, "threshold": 0.1},
    "temporal": {"enable": False},
    "llm_max_tokens": 8192,
}


def graphiti_concurrency(agent_config):
    """Graphiti's SEMAPHORE_LIMIT (default 20; nested gathers can exceed it).
    Throughput only, so it is not part of the identity fingerprint."""
    value = agent_config.get("tsm_graphiti_concurrency")
    if value in (None, ""):
        return None
    if int(value) <= 0:
        raise ValueError("tsm_graphiti_concurrency must be a positive integer")
    return int(value)


def neo4j_settings(agent_config):
    return {
        "uri": str(agent_config.get("tsm_neo4j_uri") or "bolt://localhost:7687"),
        "user": str(agent_config.get("tsm_neo4j_user") or "neo4j"),
        "database": str(agent_config.get("tsm_neo4j_database") or "neo4j"),
        "password_env": str(agent_config.get("tsm_neo4j_password_env") or "NEO4J_PASSWORD"),
    }


def native_config(agent_config):
    """The release's YAML config structure, filled from MemoryData's unified models."""
    overrides = dict(agent_config.get("tsm_native_config") or {})
    unknown = set(overrides) - {"cluster", "entity_clustering", "retriever_kwargs", "llm_max_tokens"}
    if unknown:
        raise ValueError(f"Unsupported tsm_native_config keys: {sorted(unknown)}")
    build = {**BUILD_DEFAULTS, **{k: v for k, v in overrides.items() if k != "retriever_kwargs"}}
    model = str(agent_config.get("tsm_llm_model") or agent_config.get("model") or "").strip()
    embedding = str(agent_config.get("tsm_embedding_model") or agent_config.get("embedding_model") or "").strip()
    dimension = int(agent_config.get("tsm_embedding_dimensions") or agent_config.get("embedding_dim") or 1536)
    if not model or not embedding or dimension <= 0:
        raise ValueError("TSM needs an LLM model, an embedding model and positive embedding dimensions")
    neo4j = neo4j_settings(agent_config)
    return {
        "experiment": {"name": "memorydata_tsm", "method": "tsm", "entrypoint": "main_time.py"},
        "dataset": {"dataset_name": "MemoryData", "dataset_path": "memorydata.json"},
        "cluster": dict(build["cluster"]),
        "language": "en",
        "llm": {"api_model": model, "max_tokens": int(build["llm_max_tokens"]), "temperature": 0,
                "api_key_env": "OPENAI_API_KEY"},
        "embedding": {"backend": "openai_compatible", "model": embedding, "dimension": dimension,
                      "batch_size": 128, "api_key_env": "MEMORYDATA_EMBEDDING_API_KEY"},
        "paths": {"log_path": "outputs/logs", "answer_path": "outputs/answers", "cache_path": "outputs/cache"},
        "resume": {"resumeIndex": 0, "start_index": 0},
        "retriever": {"kwargs": {**RETRIEVER_DEFAULTS, **dict(overrides.get("retriever_kwargs") or {})}},
        "split_method": "round_based",
        "temporal": dict(build["temporal"]),
        "extraction": {"method": "graphiti"},
        "graphiti": {
            "neo4j": {"uri_env": "NEO4J_URI", "user_env": "NEO4J_USER", "password_env": "NEO4J_PASSWORD",
                      "database_env": "NEO4J_DATABASE", "uri": neo4j["uri"], "user": neo4j["user"],
                      "database": neo4j["database"]},
            "group_id_prefix": "",
            "reset_group": True,
        },
        "entity_clustering": dict(build["entity_clustering"]),
        # The release builds from user messages only; MemoryData chunks are mixed
        # multi-speaker text, so the whole chunk is used (see docs/tsm.md).
        "use_user_only": False,
        "parallel_api_calls": 8,
        "batch_size": 8,
        "use_batch_api": False,
        "llm_parallel": {"enable_parallel": True, "max_parallel_llm": 8, "llm_retry_count": 3, "llm_timeout": 120},
        "use_improved_paths": False,
    }


def split_evidence(evidence, leaf_texts):
    """TSM's evidence string -> (contexts, which are leaf chunks first, then the snapshot summaries)."""
    evidence = str(evidence or "").strip()
    summary = ""
    if SUMMARY_MARKER in evidence:
        summary = evidence.split(SUMMARY_MARKER, 1)[1].strip()
    elif evidence and not leaf_texts:
        summary = evidence  # only summaries were retrieved; upstream then omits the markers
    return [t for t in leaf_texts if t.strip()], summary


# Runtime patches applied by tsm_worker.py to release 0.1.1 (docs/tsm.md). Part of
# the identity, so an index built without them is never reused as a patched one.
RELEASE_PATCHES = ("dense-index-event-loop", "tree-leaves-all-chunks")


class TsmAdapter(BufferedWorkerAdapter):
    PROTOCOL = "memorydata_tsm_worker_v1"
    STATE_FILE = "tsm_state.json"
    LABEL = "TSM"

    def __init__(self, agent_config, dataset_config, state_dir, *, worker_factory=None):
        self.config = native_config(agent_config)
        self.neo4j = neo4j_settings(agent_config)
        self.top_k = int(agent_config.get("retrieve_num", 20))
        self.answer_mode = str(agent_config.get("tsm_answer_mode", "memorydata"))
        self.timeout = float(agent_config.get("tsm_worker_timeout_seconds", 7200))
        # Added per history chunk to the index/reopen timeout (build_timeout()); the
        # smoke build took about 120 s per 4096-token chunk at concurrency 4.
        self.index_seconds_per_chunk = float(agent_config.get("tsm_index_seconds_per_chunk", 600))
        if self.index_seconds_per_chunk < 0:
            raise ValueError("tsm_index_seconds_per_chunk must not be negative")
        self.concurrency = graphiti_concurrency(agent_config)
        if self.answer_mode not in ANSWER_MODES or self.top_k <= 0 or self.timeout <= 0:
            raise ValueError("tsm_answer_mode must be memorydata or native; retrieve_num and timeout must be positive")
        dataset = {k: v for k, v in dataset_config.items() if k != "max_test_queries"}
        build_config = {k: v for k, v in self.config.items() if k != "retriever"}
        super().__init__(agent_config, state_dir, {
            "native": build_config, "release_patches": list(RELEASE_PATCHES), "dataset": portable(dataset),
            "history": agent_config.get("_memory_history_fingerprint"),
            "state_dir": portable(str(Path(state_dir).resolve())),
        }, worker_factory=worker_factory, extra_state={"group_id": "md_tsm_" + uuid.uuid4().hex,
                                                        "neo4j_uri": self.neo4j["uri"],
                                                        "neo4j_database": self.neo4j["database"]})
        self.built = False

    @classmethod
    def check_owned_state(cls, state):
        super().check_owned_state(state)
        if not re.fullmatch(r"md_tsm_[a-f0-9]{32}", str(state.get("group_id", ""))):
            raise ValueError("Refusing to operate on a Neo4j group not owned by the MemoryData TSM adapter")

    def start_worker(self):
        password = os.environ.get(self.neo4j["password_env"])
        if not password:
            raise ValueError(f"Set {self.neo4j['password_env']} (the Neo4j password) in the project .env for TSM")
        python = resolve_worker_python(self.agent_config, "tsm_python", "tsm_python_env")
        env = worker_environment(self.agent_config)
        env.update({"NEO4J_URI": self.neo4j["uri"], "NEO4J_USER": self.neo4j["user"],
                    "NEO4J_DATABASE": self.neo4j["database"], "NEO4J_PASSWORD": password,
                    "TSM_LLM_BASE_URL": env["OPENAI_BASE_URL"],
                    "TSM_EMBEDDING_BASE_URL": env["MEMORYDATA_EMBEDDING_BASE_URL"]})
        if self.concurrency is not None:
            env["SEMAPHORE_LIMIT"] = str(self.concurrency)
        return WorkerProcess(python, WORKER_SCRIPT, cwd=self.native_dir, log_dir=self.state_dir,
                             env=env, timeout=self.timeout)

    def _worker(self, docs=None):
        return self.open_worker(cache_folder=str((self.native_dir / "cache").resolve()), config=self.config,
                                group_id=self.state["group_id"], docs=self.documents() if docs is None else docs)

    def discard_partial(self):
        # A previous unfinished build may have written to this context's group.
        if self.path.exists() and self.state.get("documents"):
            try:
                self._worker(docs=[]).call("reset")
            finally:
                self.close()  # the build reopens the worker with the real documents

    def save(self):
        if self.state.get("ready"):
            return
        if not self.state["documents"]:
            raise RuntimeError("TSM received no history to build")
        stats = self._worker().call("index", timeout=self.build_timeout())
        self.built = True
        self.mark_ready(stats)

    def worker_lost(self):
        self.built = False

    def load(self):
        self.require_ready()
        if not (self.native_dir / "cache" / "tree.json").exists():
            raise RuntimeError("Saved TSM cache is missing; rerun with --force")

    def retrieve(self, query, *, query_timestamp=None, native_answer=False):
        """TSM retrieval; returns (native answer or "", contexts with source IDs)."""
        self.require_ready()
        worker = self._worker()
        if not self.built:
            worker.call("reopen", timeout=self.build_timeout())
            self.built = True
        result = worker.call("query", query=query, question_date=query_timestamp_or_none(query_timestamp),
                             native_answer=native_answer)
        positions = [int(p) for p in result.get("leaf_positions") or []]
        texts = [str((result.get("leaf_texts") or {}).get(str(p), "")) for p in positions]
        leaves, summary = split_evidence(result.get("evidence"), texts)
        documents = self.state["documents"]
        entries = [{"text": text, "type": "chunk",
                    "source_ids": self.source_ids_for([documents[p]["doc_id"]] if 0 <= p < len(documents) else [])}
                   for p, text in zip(positions, texts) if text.strip()]
        entries = entries[:self.top_k]
        if summary:
            entries.append({"text": summary, "type": "snapshot_summaries", "source_ids": []})
        return str(result.get("answer") or ""), entries

    @classmethod
    def reset_saved(cls, agent_config, dataset_config, state_dir):
        """--force: delete this context's Neo4j group recorded in the local manifest, nothing else."""
        state = cls.read_state(state_dir)
        if not state:
            return
        cls.check_owned_state(state)
        adapter = cls.__new__(cls)
        adapter.agent_config = agent_config
        adapter.state_dir = Path(state_dir)
        adapter.native_dir = adapter.state_dir / cls.NATIVE_DIR
        adapter.config = native_config(agent_config)
        adapter.neo4j = neo4j_settings(agent_config)
        if (adapter.neo4j["uri"], adapter.neo4j["database"]) != (state.get("neo4j_uri"), state.get("neo4j_database")):
            raise ValueError("TSM Neo4j endpoint changed; reset with the old endpoint or use a new --artifact_root")
        adapter.timeout = float(agent_config.get("tsm_worker_timeout_seconds", 7200))
        adapter.concurrency = graphiti_concurrency(agent_config)
        adapter.state = state
        adapter.worker = None
        adapter.worker_factory = adapter.start_worker
        try:
            adapter._worker(docs=[]).call("reset")
        finally:
            adapter.close()
