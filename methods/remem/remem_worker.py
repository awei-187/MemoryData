"""REMem worker: runs inside the REMem conda environment, never inside MemoryData's.

Started by ``methods/remem/remem_adapter.py`` as ``<remem python> -I -u remem_worker.py``
with the per-context native directory as working directory. It imports the
unmodified upstream package from ``methods/remem/source/REMem-main/src`` and
exposes four operations: ``open``, ``index``, ``query`` and ``stats``.

Glue added here, all documented in docs/remem.md:
* each MemoryData chunk becomes one single-message session whose message keeps
  ``memorydata_doc`` as metadata (upstream stores message dicts as verbatim
  metadata), so retrieved nodes can be traced back to benchmark source IDs;
* the session chunker treats a session without a date like upstream
  ``chunk_by_session`` would treat a dated one, instead of failing on a missing
  ``date`` key;
* ``FlexibleToolsStrategy._format_result`` is wrapped to additionally report the
  internal node names of the passages the agent visited. Retrieval and answer
  behaviour are unchanged.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "source" / "REMem-main"


def format_session(doc):
    """One MemoryData chunk -> one REMem session (a list with one message dict)."""
    text = str(doc["text"])
    date = doc.get("date")
    content = f"{date}\n{text}" if date else text
    return [{"content": content, "memorydata_doc": doc["doc_id"]}]


def chunk_session(session):
    """Session -> one chunk, like upstream chunk_by_session for a single-date session."""
    if not isinstance(session, list) or not session:
        raise ValueError("REMem worker expects non-empty message-list sessions")
    return [{"content": "\n".join(str(message["content"]) for message in session)}]


def native_node_type(name):
    """Upstream node keys are content hashes prefixed with their store name."""
    for prefix in ("verbatim", "gists", "facts", "entity"):
        if str(name).startswith(prefix + "-"):
            return prefix
    return "other"


def resolve_source_docs(node_names, *, verbatim_doc, neighbors):
    """Map visited node names to MemoryData document IDs, keeping order.

    verbatim nodes carry their document; gist nodes are linked to the verbatim
    nodes of their own chunk; facts are linked to the gists of their chunk.
    Entities connect many chunks and are deliberately not attributed.
    """
    groups = []
    for name in node_names:
        kind = native_node_type(name)
        docs = []
        if kind == "verbatim":
            docs = [verbatim_doc(name)]
        elif kind == "gists":
            docs = [verbatim_doc(n) for n in neighbors(name) if native_node_type(n) == "verbatim"]
        elif kind == "facts":
            gists = [n for n in neighbors(name) if native_node_type(n) == "gists"]
            docs = [verbatim_doc(v) for g in gists for v in neighbors(g) if native_node_type(v) == "verbatim"]
        groups.append(list(dict.fromkeys(d for d in docs if d)))
    return groups


class RememService:
    def __init__(self, source_dir=None):
        self.source_dir = Path(source_dir or os.environ.get("REMEM_SOURCE_DIR") or DEFAULT_SOURCE).resolve()
        self.rag = None
        self.sessions = None

    def _import_native(self):
        sys.path.insert(0, str(self.source_dir / "src"))
        from remem.agent.strategies import flexible_tools_strategy as flexible
        from remem.embedding_model.openai_embedding_client import CacheOpenAIEmbeddingModel
        from remem.remem import ReMem
        from remem.utils.config_utils import BaseConfig

        if not getattr(flexible.FlexibleToolsStrategy, "_memorydata_observed", False):
            original = flexible.FlexibleToolsStrategy._format_result

            def format_result(self, state, return_chunk):
                result = original(self, state, return_chunk)
                result.logs["memorydata_visited_nodes"] = [
                    p.get("node_name") for p in state.retrieved_passages if p.get("node_name")]
                return result

            flexible.FlexibleToolsStrategy._format_result = format_result
            flexible.FlexibleToolsStrategy._memorydata_observed = True
        return ReMem, BaseConfig, CacheOpenAIEmbeddingModel

    def open(self, args):
        ReMem, BaseConfig, Embedding = self._import_native()
        native = dict(args["config"])
        dspy_file = native.pop("rerank_dspy_file", None)
        if dspy_file:
            native["rerank_dspy_file_path"] = str((self.source_dir / dspy_file).resolve())
        config = BaseConfig(llm_base_url=os.environ["OPENAI_BASE_URL"], dataset="memorydata",
                            preprocess_chunk_func="by_session", **native)
        self.rag = ReMem(global_config=config, working_dir=str(Path(args["working_dir"]).resolve()))
        self.rag.text_preprocessor.chunking_func = chunk_session
        self.rag.set_embedding_model(Embedding(
            None, global_config=config, embedding_model_name=config.embedding_model_name,
            api_key=os.environ["MEMORYDATA_EMBEDDING_API_KEY"],
            base_url=os.environ["MEMORYDATA_EMBEDDING_BASE_URL"],
        ))
        self.sessions = [format_session(doc) for doc in args["docs"]]
        return {"opened": True, "sessions": len(self.sessions)}

    def index(self, args):
        # Upstream skips extraction when graph, embeddings and extraction results
        # already exist, so the same call both builds and reopens an index.
        self.rag.index(self.sessions)
        if not self.rag.ready_to_retrieve:
            self.rag.rag_strategy.prepare_retrieval_objects()
        stores = self.rag.episodic_embedding_stores
        stats = {"graph_nodes": self.rag.graph.vcount(), "graph_edges": self.rag.graph.ecount(),
                 **{f"{name}_count": len(store.get_all_ids()) for name, store in stores.items()}}
        # Upstream logs and continues on some extraction failures; never report a
        # degraded index as complete.
        if not stats["graph_nodes"] or not stats.get("verbatim_count") or not stats.get("gists_count"):
            raise RuntimeError(f"REMem index is incomplete: {stats}")
        return stats

    def _verbatim_doc(self, name):
        row = self.rag.episodic_embedding_stores["verbatim"].get_rows([name]).get(name) or {}
        metadata = row.get("metadata") or {}
        return metadata.get("memorydata_doc") if isinstance(metadata, dict) else None

    def _neighbors(self, name):
        graph = self.rag.graph
        try:
            vertex = graph.vs.find(name=name)
        except (ValueError, KeyError):
            return []
        return [graph.vs[i]["name"] for i in graph.neighbors(vertex.index, mode="all")]

    def query(self, args):
        metadata = {"date": args["question_date"]} if args.get("question_date") else None
        _, _, agent_result = self.rag.rag_strategy._rag_each_query(
            self.rag, args["query"], return_chunk="verbatim", question_metadata=metadata)
        logs = agent_result[2] if len(agent_result) > 2 and isinstance(agent_result[2], dict) else {}
        visited = list(dict.fromkeys(logs.get("memorydata_visited_nodes") or []))
        contents = []
        for name in visited:
            row = None
            kind = native_node_type(name)
            if kind in self.rag.episodic_embedding_stores:
                row = self.rag.episodic_embedding_stores[kind].get_rows([name]).get(name)
            contents.append(str((row or {}).get("content", "")))
        return {
            "answer": logs.get("agent_answer") or "",
            "passages": [{"node": n, "type": native_node_type(n), "text": t} for n, t in zip(visited, contents)],
            "source_docs": resolve_source_docs(visited, verbatim_doc=self._verbatim_doc, neighbors=self._neighbors),
            "reasoning": logs.get("agent_observation") or [],
        }

    def handle(self, op, args):
        if op == "open":
            return self.open(args)
        if self.rag is None:
            raise RuntimeError("REMem worker is not open")
        if op == "index":
            return self.index(args)
        if op == "query":
            return self.query(args)
        raise ValueError(f"Unknown REMem worker operation: {op}")


def _protocol():
    spec = importlib.util.spec_from_file_location("memorydata_worker_protocol", HERE.parent / "worker_protocol.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    _protocol().serve(RememService().handle)
