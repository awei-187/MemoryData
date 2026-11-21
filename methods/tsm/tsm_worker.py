"""TSM worker: runs inside the TSM conda environment, never inside MemoryData's.

Started by ``methods/tsm/tsm_adapter.py`` as ``<tsm python> -I -X utf8 -u tsm_worker.py``
with the per-context native directory as working directory. It imports the
unmodified release in ``methods/tsm/source/TSM-main`` and drives the same
functions as its ``main_time.py`` pipeline for one data piece:
``parallel_build_extract`` -> ``build_retriever`` -> ``Retriever.query``, and
for the optional native answer the release's ``QA_prompt_mirix_improved``.

Glue added here, documented in docs/tsm.md:
* each MemoryData chunk becomes one TSM "turn" string ``Date: <timestamp>\\n<text>``
  (TSM's round-based loader builds the same ``Date:`` header from dataset turns);
* ``Retriever._format_results_with_summaries`` is wrapped to additionally record
  the leaf IDs (``leaf_<i>``) it formats, so results map back to chunk ``i``.
  The returned text is unchanged.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "source" / "TSM-main"
CACHE_FILES = ("tree.json", "graph.json", "index.json", "appearance_count.json")


def format_turn(doc):
    """``Date: <timestamp>`` header like the release's round loader.

    GraphitiExtractor._reference_time looks for ``\\bYYYY-MM-DD\\b``; in an ISO
    timestamp the date is followed by ``T``, which defeats the word boundary and
    silently falls back to 2000-01-01, so that ``T`` becomes a space.
    """
    text = str(doc["text"])
    if not doc.get("date"):
        return text
    date = re.sub(r"^(\d{4}-\d{1,2}-\d{1,2})T", r"\1 ", str(doc["date"]))
    return f"Date: {date}\n{text}"


def leaf_positions(chunk_ids):
    """Ordered chunk indices of ``leaf_<i>`` IDs in a {key: [ids]} map; summaries are skipped."""
    positions = []
    for ids in (chunk_ids or {}).values():
        for chunk_id in ids:
            match = re.fullmatch(r"leaf_(\d+)", str(chunk_id))
            if match:
                positions.append(int(match.group(1)))
    return list(dict.fromkeys(positions))


def episode_without_uuid(factory):
    """Wrap an episode constructor so a caller-supplied uuid is ignored."""

    def build(*args, **kwargs):
        kwargs.pop("uuid", None)
        return factory(*args, **kwargs)

    build._memorydata_compat = True
    return build


REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def is_reasoning_model(model):
    name = str(model or "").strip().lower().rsplit("/", 1)[-1]
    return name.startswith(REASONING_MODEL_PREFIXES)


def client_without_reasoning(factory):
    """Wrap an OpenAIClient constructor: no reasoning/verbosity for non-reasoning models."""

    def build(*args, **kwargs):
        config = kwargs.get("config", args[0] if args else None)
        if not is_reasoning_model(getattr(config, "model", None)):
            kwargs.setdefault("reasoning", None)
            kwargs.setdefault("verbosity", None)
        return factory(*args, **kwargs)

    build._memorydata_compat = True
    return build


def outside_running_loop(func):
    """Run ``func`` in a fresh thread when the caller is inside a running event loop.

    TSM's build_dense_index.embed_documents_batch bridges the async embedder with
    new_event_loop().run_until_complete(), but parallel_build_extract calls it from
    inside asyncio.run(); every batch then fails ("Cannot run the event loop while
    another loop is running") and is silently replaced by zero vectors. The same
    function with the same arguments runs unchanged in a thread without a loop,
    which is how graphiti_extractor._run_coroutine handles the same situation.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    def run(*args, **kwargs):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return func(*args, **kwargs)
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(func, *args, **kwargs).result()

    run._memorydata_compat = True
    return run


def rebuild_leaves(cache, text_chunks, original_text=None):
    """One ``leaf_<i>`` per input chunk, as the non-temporal build_tree_clustering.py does.

    The temporal build_tree_clustering_time.py rebinds ``text_chunks`` to each
    month's episode contents and numbers every window's leaves from leaf_0, so
    ``all_caches.update`` keeps only the last window's leaves. Leaf texts follow
    that file's own rule (``full_text`` when ``original_text`` holds turn dicts).
    """
    rebuilt = {key: value for key, value in (cache or {}).items() if not str(key).startswith("leaf_")}
    for position, chunk in enumerate(text_chunks or []):
        text = chunk
        if original_text and position < len(original_text):
            original = original_text[position]
            if isinstance(original, dict) and "full_text" in original:
                text = original["full_text"]
        rebuilt[f"leaf_{position}"] = {"text": text, "children": None, "parent": None}
    return rebuilt


def leaves_for_all_chunks(build):
    """Wrap build_tree_with_clustering_sync: keep its snapshots and index, rebuild the leaves."""

    def run(*args, **kwargs):
        cache, cost, cluster_index = build(*args, **kwargs)
        if kwargs.get("snapshots_only"):
            return cache, cost, cluster_index
        cache = rebuild_leaves(cache, kwargs["text_chunks"], kwargs.get("original_text"))
        tree_path = Path(kwargs["cache_folder"]) / "tree.json"
        tree_path.write_text(json.dumps(cache, ensure_ascii=False, indent=4), encoding="utf-8")
        return cache, cost, cluster_index

    run._memorydata_compat = True
    return run


def dense_index_problem(dense_retriever, leaf_count):
    """Upstream logs embedding failures and stores zero vectors; refuse such an index."""
    index = getattr(dense_retriever, "faiss_index", None)
    if index is None:
        return "no FAISS index"
    if int(index.ntotal) != int(leaf_count):
        return f"{index.ntotal} vectors for {leaf_count} leaves"
    import numpy as np

    vectors = np.asarray(index.reconstruct_n(0, index.ntotal))
    zero = int((np.abs(vectors).sum(axis=1) == 0).sum()) if len(vectors) else 0
    return f"{zero} of {len(vectors)} vectors are all zero (embedding failed)" if zero else None


class TsmService:
    def __init__(self, source_dir=None):
        self.source_dir = Path(source_dir or os.environ.get("TSM_SOURCE_DIR") or DEFAULT_SOURCE).resolve()
        self.main = None
        self.bundle = None
        self.retriever = None

    def _import_native(self):
        if self.main is None:
            sys.path.insert(0, str(self.source_dir))
            import main_time
            import query_refactored

            retriever_class = query_refactored.Retriever
            if not getattr(retriever_class, "_memorydata_observed", False):
                original = retriever_class._format_results_with_summaries

                def observed(self, chunk_ids, selected_snapshots):
                    self.memorydata_last_chunk_ids = {k: list(v) for k, v in (chunk_ids or {}).items()}
                    return original(self, chunk_ids, selected_snapshots)

                retriever_class._format_results_with_summaries = observed
                retriever_class._memorydata_observed = True
            self._graphiti_config_compat()
            self._graphiti_episode_compat()
            self._graphiti_reasoning_compat()
            self._release_defect_patches()
            self.main = main_time
        return self.main

    @staticmethod
    def _graphiti_config_compat():
        """The release passes LLMConfig(request_timeout=...), which its own pin
        graphiti-core==0.19.0 does not accept (TypeError before any extraction).
        Accept and drop that one argument; Graphiti then keeps the OpenAI SDK's
        default request timeout instead of the release's 120 s."""
        import inspect

        import graphiti_extractor

        base = graphiti_extractor.LLMConfig
        if "request_timeout" in inspect.signature(base.__init__).parameters or getattr(base, "_memorydata_compat", False):
            return

        class LLMConfigCompat(base):
            _memorydata_compat = True

            def __init__(self, *args, request_timeout=None, **kwargs):
                super().__init__(*args, **kwargs)

        graphiti_extractor.LLMConfig = LLMConfigCompat

    @staticmethod
    def _graphiti_episode_compat():
        """The release builds RawEpisode(uuid=<stable uuid5>, ...), but graphiti-core
        0.19.0's add_episode_bulk treats a given uuid as an existing episode and
        loads it (NodeNotFoundError in a new group). Drop the uuid so Graphiti
        creates the episode; _read_compatible_graph reads uuids back by the name
        chunk_<i>, so only the release's idempotent re-insert is lost."""
        import inspect

        import graphiti_extractor
        from graphiti_core import Graphiti

        factory = graphiti_extractor.RawEpisode
        if getattr(factory, "_memorydata_compat", False):
            return
        if "get_by_uuid(" not in inspect.getsource(Graphiti.add_episode_bulk):
            return
        graphiti_extractor.RawEpisode = episode_without_uuid(factory)

    @staticmethod
    def _graphiti_reasoning_compat():
        """graphiti-core 0.19.0's OpenAIClient defaults to reasoning='minimal' and
        verbosity='low' (GPT-5 parameters) on every structured call, and the
        release builds it with OpenAIClient(config=..., cache=False). Models
        such as gpt-4o-mini (the release's own default) reject those with 400.
        For non-reasoning models pass None, so the same model is used without them."""
        import inspect

        import graphiti_extractor

        factory = graphiti_extractor.OpenAIClient
        if getattr(factory, "_memorydata_compat", False):
            return
        if not {"reasoning", "verbosity"} <= set(inspect.signature(factory.__init__).parameters):
            return
        graphiti_extractor.OpenAIClient = client_without_reasoning(factory)

    @staticmethod
    def _release_defect_patches():
        """Two logic defects of release 0.1.1 (see docs/tsm.md); upstream files stay unchanged."""
        import build_dense_index
        import build_tree_clustering_time

        if not getattr(build_dense_index.embed_documents_batch, "_memorydata_compat", False):
            build_dense_index.embed_documents_batch = outside_running_loop(build_dense_index.embed_documents_batch)
        # main_time imports this name inside the function body at call time.
        if not getattr(build_tree_clustering_time.build_tree_with_clustering_sync, "_memorydata_compat", False):
            build_tree_clustering_time.build_tree_with_clustering_sync = leaves_for_all_chunks(
                build_tree_clustering_time.build_tree_with_clustering_sync)

    def open(self, args):
        main = self._import_native()
        self.bundle = main.extract_config_bundle(args["config"])
        self.cache_folder = str(Path(args["cache_folder"]).resolve())
        self.group_id = args["group_id"]
        self.turns = [format_turn(doc) for doc in args.get("docs") or []]
        return {"opened": True, "turns": len(self.turns)}

    def index(self, args):
        main = self._import_native()
        Path(self.cache_folder).mkdir(parents=True, exist_ok=True)
        result = asyncio.run(main.parallel_build_extract(
            self.turns, self.bundle, self.cache_folder, self.bundle.cluster_length,
            self.bundle.cluster_overlap, self.bundle.cluster_merge_num, self.group_id))
        tree, graph = result[0], result[1]
        dense = result[2] if len(result) == 3 else None
        self.retriever = main.build_retriever(tree, graph, self.bundle, dense, self.group_id,
                                              cache_folder=self.cache_folder)
        return self._stats()

    def reopen(self, args):
        """Load cached files exactly like main_time.run_pipeline's cached path."""
        main = self._import_native()
        from extract_graph import build_graph

        folder = Path(self.cache_folder)
        missing = [name for name in CACHE_FILES if not (folder / name).exists()]
        if missing:
            raise FileNotFoundError(f"TSM cache files missing: {missing}")
        load = lambda name: json.loads((folder / name).read_text(encoding="utf-8"))
        tree = load("tree.json")
        mapping = load("text_edge_mapping.json") if (folder / "text_edge_mapping.json").exists() else {}
        graph = (build_graph(load("graph.json")), load("index.json"), load("appearance_count.json"), mapping)
        # Upstream passes no prebuilt index on this path; the retriever re-embeds the tree.
        self.retriever = main.build_retriever(tree, graph, self.bundle, None, self.group_id,
                                              cache_folder=self.cache_folder)
        return self._stats()

    def _stats(self):
        """Verify the build like a complete upstream run; several upstream steps log and continue."""
        folder = Path(self.cache_folder)
        missing = [name for name in CACHE_FILES if not (folder / name).exists()]
        snapshots = folder / "snapshots.json"
        clustering = bool(getattr(self.bundle, "enable_entity_clustering", False))
        if clustering and not snapshots.exists():
            missing.append("snapshots.json")
        stats = {"tree_nodes": len(getattr(self.retriever, "cache_tree", {}) or {}),
                 "snapshots": len(json.loads(snapshots.read_text(encoding="utf-8"))) if snapshots.exists() else 0,
                 "dense_retriever": getattr(self.retriever, "dense_retriever", None) is not None}
        dense_wanted = self.bundle.retriever_kwargs.get("enable_dense_retrieval", False)
        if missing or not stats["tree_nodes"] or (dense_wanted and not stats["dense_retriever"]):
            raise RuntimeError(f"TSM build is incomplete: missing={missing}, stats={stats}")
        tree = getattr(self.retriever, "cache_tree", {}) or {}
        stats["leaves"] = sum(1 for key in tree if str(key).startswith("leaf_"))
        if stats["leaves"] != len(self.turns):
            raise RuntimeError(f"TSM tree has {stats['leaves']} leaves for {len(self.turns)} history chunks")
        if dense_wanted:
            dense = self.retriever.dense_retriever
            problem = dense_index_problem(dense, len(getattr(dense, "collapse_tree_ids", None) or tree))
            if problem:
                raise RuntimeError(f"TSM dense index is unusable: {problem}")
        return stats

    def query(self, args):
        if self.retriever is None:
            raise RuntimeError("TSM retriever is not built; call index or reopen first")
        kwargs = dict(self.bundle.retriever_kwargs)
        if args.get("question_date"):
            kwargs["question_date"] = args["question_date"]
        self.retriever.memorydata_last_chunk_ids = {}
        result = self.retriever.query(args["query"], **kwargs)
        if result.get("error"):
            raise RuntimeError(f"TSM retrieval failed: {result['error']}")
        evidence = str(result.get("chunks") or "")
        reply = {"evidence": evidence,
                 "leaf_positions": leaf_positions(getattr(self.retriever, "memorydata_last_chunk_ids", {})),
                 "leaf_texts": {}}
        tree = getattr(self.retriever, "cache_tree", {}) or {}
        for position in reply["leaf_positions"]:
            node = tree.get(f"leaf_{position}") or {}
            reply["leaf_texts"][str(position)] = str(node.get("text", "")) if isinstance(node, dict) else ""
        if args.get("native_answer"):
            reply["answer"] = self._native_answer(args["query"], args.get("question_date"), evidence)
        return reply

    def _native_answer(self, question, question_date, evidence):
        from prompt_dict import Prompts
        from provider.public_llm_provider import call_llm

        llm = self.bundle.llm_api
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": Prompts["QA_prompt_mirix_improved"].format(
                question_date=question_date, question=question, evidence=evidence)},
        ]
        return call_llm(messages=messages, model=llm.get("backend_model", llm["model"]),
                        temperature=llm["temperature"], max_tokens=llm["max_tokens"],
                        base_url=llm.get("base_url"), api_key_env=llm.get("api_key_env", "OPENAI_API_KEY"))

    def reset(self, args):
        """Delete only this context's Graphiti group from Neo4j."""
        self._import_native()
        from graphiti_extractor import GraphitiExtractor

        extractor = GraphitiExtractor(self.bundle)

        async def clear():
            await extractor.clear_group_data(self.group_id)
            if extractor.graphiti is not None:
                await extractor.graphiti.close()

        asyncio.run(clear())
        return {"reset": self.group_id}

    def handle(self, op, args):
        if op == "open":
            return self.open(args)
        if self.bundle is None:
            raise RuntimeError("TSM worker is not open")
        operations = {"index": self.index, "reopen": self.reopen, "query": self.query, "reset": self.reset}
        if op not in operations:
            raise ValueError(f"Unknown TSM worker operation: {op}")
        return operations[op](args)


def _protocol():
    spec = importlib.util.spec_from_file_location("memorydata_worker_protocol", HERE.parent / "worker_protocol.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    _protocol().serve(TsmService().handle)
