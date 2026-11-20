"""Parent side of a JSON-lines bridge to a memory method running in its own Python environment.

Some native memory systems pin dependencies that conflict with the MemoryData
environment (for example REMem needs openai<2 and numpy<2, TSM pins
graphiti-core). Like the Hindsight REST adapter, MemoryData then talks to the
native code through a narrow protocol instead of importing it:

* the worker is started as ``<python> -I -u <worker script>`` with the method's
  own interpreter, so neither PYTHONPATH nor the script directory leaks in;
* credentials reach the worker only through its environment, never through
  arguments, requests, manifests or logs written by MemoryData;
* each request is one JSON line ``{"id", "op", "args"}`` and each reply one JSON
  line ``{"id", "ok", "result" | "error"}`` on the worker's original stdout;
  everything native code prints goes to ``worker.log`` in the state directory.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import queue
import shutil
import subprocess
import threading
from pathlib import Path

LOG_FILE = "worker.log"
# Strong references: after a crash the adapter may already be garbage when
# atexit runs, and a weakly held worker would never be stopped.
_LIVE_WORKERS = set()


def close_all_workers():
    """Stop every live worker. A worker holds its context directory as working
    directory (locked on Windows) and the native index in memory, so callers
    close them before a new context starts and before --force deletes state."""
    for worker in list(_LIVE_WORKERS):
        worker.close()


atexit.register(close_all_workers)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def atomic_save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class WorkerError(RuntimeError):
    """The worker reported a failure or stopped answering."""


class WorkerProcess:
    """One long-lived worker per context; requests are strictly sequential."""

    def __init__(self, python, script, *, cwd, env, log_dir, timeout=3600.0):
        python = str(python or "").strip()
        if not python or not Path(python).is_file():
            raise FileNotFoundError(
                f"Worker interpreter not found: {python!r}. Create the method's conda environment "
                "and set its python path in the agent config."
            )
        if not Path(script).is_file():
            raise FileNotFoundError(f"Worker script not found: {script}")
        Path(cwd).mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_dir) / LOG_FILE
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout = float(timeout)
        self.closed = False
        self._log = self.log_path.open("ab")
        self._next_id = 0
        self._replies = queue.Queue()
        self.process = subprocess.Popen(
            [python, "-I", "-u", str(script)],
            cwd=str(cwd), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._log, bufsize=0,
        )
        self._reader = threading.Thread(target=self._read_replies, daemon=True)
        self._reader.start()
        _LIVE_WORKERS.add(self)
        self._log.write(f"[memorydata] worker pid {self.process.pid} started: {script}\n".encode("utf-8"))
        self._log.flush()

    def _read_replies(self):
        for raw in iter(self.process.stdout.readline, b""):
            self._replies.put(raw)
        self._replies.put(None)

    def _log_tail(self, limit=4000):
        try:
            self._log.flush()
            data = self.log_path.read_bytes()[-limit:]
            return data.decode("utf-8", errors="replace")
        except OSError:
            return ""

    def call(self, op, timeout=None, **args):
        if self.process.poll() is not None:
            raise WorkerError(f"Worker exited with code {self.process.returncode}\n{self._log_tail()}")
        self._next_id += 1
        request_id = self._next_id
        line = json.dumps({"id": request_id, "op": op, "args": args}, ensure_ascii=False) + "\n"
        try:
            self.process.stdin.write(line.encode("utf-8"))
            self.process.stdin.flush()
        except OSError as exc:
            raise WorkerError(f"Worker stdin closed: {exc}\n{self._log_tail()}") from exc
        try:
            raw = self._replies.get(timeout=self.timeout if timeout is None else timeout)
        except queue.Empty as exc:
            self.close(kill=True)
            raise WorkerError(f"Worker did not answer {op!r} in time\n{self._log_tail()}") from exc
        if raw is None:
            raise WorkerError(f"Worker exited during {op!r}\n{self._log_tail()}")
        reply = json.loads(raw.decode("utf-8"))
        if reply.get("id") != request_id:
            raise WorkerError(f"Worker reply id {reply.get('id')} does not match request {request_id}")
        if not reply.get("ok"):
            raise WorkerError(f"Worker {op!r} failed: {reply.get('error')}\n{self._log_tail()}")
        return reply.get("result")

    def close(self, kill=False):
        if self.closed:
            return
        self.closed = True
        _LIVE_WORKERS.discard(self)
        if self.process.poll() is None:
            if not kill:
                try:
                    self.call("shutdown", timeout=30)
                except Exception:
                    kill = True
            if kill or self.process.poll() is None:
                self.process.kill()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
        for stream in (self.process.stdin, self.process.stdout):
            try:
                stream.close()
            except OSError:
                pass
        if not self._log.closed:
            self._log.close()


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class BufferedWorkerAdapter:
    """Per-context state shared by batch-indexing methods behind a worker.

    * ``<STATE_FILE>``: protocol, identity fingerprint, ready flag and, per
      document, its ID, source IDs and timestamp. No credentials, no texts.
    * ``documents.jsonl``: chunk texts, kept locally so a reopened index is
      built from exactly the documents recorded in the manifest.
    * ``native/``: files written by the native method through the worker.

    An ingestion that never reached ``ready`` is discarded and rebuilt in full,
    like the Hindsight adapter does; nothing is resumed half-way.
    """

    PROTOCOL = None
    STATE_FILE = None
    LABEL = None
    DOCS_FILE = "documents.jsonl"
    NATIVE_DIR = "native"

    def __init__(self, agent_config, state_dir, identity_payload, *, worker_factory=None, extra_state=None):
        self.agent_config = agent_config
        self.state_dir = Path(state_dir)
        self.path = self.state_dir / self.STATE_FILE
        self.docs_path = self.state_dir / self.DOCS_FILE
        self.native_dir = self.state_dir / self.NATIVE_DIR
        self.worker_factory = worker_factory or self.start_worker
        self.identity = fingerprint({"protocol": self.PROTOCOL, **identity_payload})
        self.state = self.read_state(state_dir)
        if self.state:
            self.check_owned_state(self.state)
            if self.state.get("identity") != self.identity:
                raise ValueError(f"{self.LABEL} saved configuration/history differs; use --force or a new --artifact_root")
        else:
            self.state = {"protocol": self.PROTOCOL, "identity": self.identity, "ready": False,
                          "documents": [], **(extra_state or {})}
        self.worker = None
        self.started = False

    @classmethod
    def read_state(cls, state_dir):
        path = Path(state_dir) / cls.STATE_FILE
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    @classmethod
    def check_owned_state(cls, state):
        if state.get("protocol") != cls.PROTOCOL:
            raise ValueError(f"Refusing to reuse a state not written by the MemoryData {cls.LABEL} adapter")

    def start_worker(self):
        raise NotImplementedError

    def _begin(self):
        if self.started:
            return
        if self.state.get("ready"):
            raise RuntimeError(f"Cannot append history to a completed {self.LABEL} index; rebuild with --force")
        self.discard_partial()
        for leftover in (self.docs_path, self.native_dir):
            if leftover.is_dir():
                shutil.rmtree(leftover)
            elif leftover.exists():
                leftover.unlink()
        self.state["documents"] = []
        atomic_save_json(self.path, self.state)
        self.started = True

    def discard_partial(self):
        """Hook for external state of an unfinished ingestion (e.g. a graph database group)."""

    def add_chunk(self, content, *, source_ids=(), timestamp=None):
        if not str(content).strip():
            return
        self._begin()
        doc_id = f"md-doc-{len(self.state['documents']):08d}"
        record = {"doc_id": doc_id, "text": str(content), "date": timestamp or None}
        self.docs_path.parent.mkdir(parents=True, exist_ok=True)
        with self.docs_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.state["documents"].append({"doc_id": doc_id, "date": record["date"],
                                        "source_ids": list(dict.fromkeys(str(x) for x in source_ids if x))})
        atomic_save_json(self.path, self.state)

    def documents(self):
        docs = read_jsonl(self.docs_path)
        if [d["doc_id"] for d in docs] != [d["doc_id"] for d in self.state["documents"]]:
            raise RuntimeError(f"{self.LABEL} document buffer does not match the state manifest; rebuild with --force")
        return docs

    def open_worker(self, **open_args):
        if self.worker is not None and getattr(self.worker, "closed", False):
            self.worker = None  # stopped by close_all_workers(); reopen on demand
            self.worker_lost()
        if self.worker is None:
            worker = self.worker_factory()
            try:
                worker.call("open", **open_args)
            except Exception:
                worker.close(kill=True)
                raise
            self.worker = worker
        return self.worker

    def worker_lost(self):
        """Hook: forget in-worker state (e.g. a loaded index) after the worker stopped."""

    def mark_ready(self, stats):
        self.state.update({"ready": True, "index_stats": stats})
        atomic_save_json(self.path, self.state)

    def require_ready(self):
        if not self.state.get("ready"):
            raise RuntimeError(f"{self.LABEL} history is incomplete; rebuild before answering")

    def source_ids_for(self, doc_ids):
        sources = {d["doc_id"]: d["source_ids"] for d in self.state["documents"]}
        return list(dict.fromkeys(s for doc in doc_ids for s in sources.get(doc, [])))

    def close(self):
        if self.worker is not None:
            self.worker.close()
            self.worker = None


def resolve_worker_python(agent_config, key, env_key):
    """Interpreter for the method's own environment: an env variable overrides the config."""
    value = os.environ.get(str(agent_config.get(env_key) or "")) if agent_config.get(env_key) else None
    return str(value or agent_config.get(key) or "").strip()


def worker_environment(agent_config, extra=None, environment=None):
    """Environment handed to the worker process only; never serialized."""
    source = os.environ if environment is None else environment
    env = dict(source)
    api_key = source.get(agent_config.get("api_key_env") or "OPENAI_API_KEY")
    base_url = agent_config.get("base_url") or source.get(agent_config.get("base_url_env") or "OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise ValueError("Set OPENAI_API_KEY and OPENAI_BASE_URL (or the configured *_env names) in the project .env")
    env["OPENAI_API_KEY"] = api_key
    env["OPENAI_BASE_URL"] = base_url.rstrip("/")
    embedding_key = source.get(agent_config.get("embedding_api_key_env") or "") or api_key
    embedding_url = (agent_config.get("embedding_base_url")
                     or source.get(agent_config.get("embedding_base_url_env") or "") or base_url)
    env["MEMORYDATA_EMBEDDING_API_KEY"] = embedding_key
    env["MEMORYDATA_EMBEDDING_BASE_URL"] = embedding_url.rstrip("/")
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("PYTHONPATH", None)
    env.update(extra or {})
    return env
