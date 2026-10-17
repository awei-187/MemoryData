"""One-unit-at-a-time execution, with content-addressed memory and results."""
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

from .common import PROTOCOL, UPSTREAM, atomic_json, file_digest, fingerprint, require_identity
from .loader import dataset_identity, iter_eval_units
from .templates import format_question, get_template


def release_agent(agent):
    """Release local persistence before moving to another unit on Windows."""
    memory = getattr(agent, "memory", None)
    connection = getattr(getattr(memory, "db", None), "connection", None)
    if connection is not None:
        connection.close()
    vector_client = getattr(getattr(memory, "vector_store", None), "client", None)
    if vector_client is not None and hasattr(vector_client, "close"):
        vector_client.close()
    clients = [getattr(agent, "client", None),
               getattr(getattr(memory, "llm", None), "client", None),
               getattr(getattr(memory, "embedding_model", None), "client", None)]
    seen = set()
    for client in clients:
        if client is not None and id(client) not in seen and hasattr(client, "close"):
            seen.add(id(client))
            client.close()


def run(args, agent_config, dataset_config):
    from utils.initialization import initialize_and_memorize_agent
    from utils.eval_other_utils import metrics_summarization
    from utils.artifact_paths import rebase_legacy_artifact_path, resolve_results_artifact_path

    started_run = time.time()
    identity = dataset_identity(dataset_config)
    module_root = Path(__file__).parent
    source_hashes = {name: file_digest(module_root / name) for name in ("loader.py", "templates.py", "runtime.py")}
    source_hashes["agent.py"] = file_digest(module_root.parents[1] / "utils/agent.py")
    config_identity = fingerprint({"dataset": dataset_config, "agent": agent_config, "protocol": PROTOCOL, "sources": source_hashes})
    run_id = fingerprint({"data": identity, "config": config_identity})
    artifact_override = args.artifact_root or agent_config.get("artifact_root") or ""
    model_namespace = str(agent_config.get("model", "unknown")).replace("/", "_")
    output_dir = rebase_legacy_artifact_path(agent_config.get("output_dir", f"./results/outputs/{model_namespace}"), artifact_override)
    root = Path(output_dir) / "OmniMemEval" / dataset_config["sub_dataset"] / run_id[:16]
    state_root = Path(resolve_results_artifact_path(artifact_override, "agents", "omnimemeval", dataset_config["sub_dataset"], run_id[:16]))
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": PROTOCOL, "upstream": UPSTREAM, "data": identity, "config_fingerprint": config_identity, "run_id": run_id}
    require_identity(root / "manifest.json", manifest)
    output_path = root / "benchmark_results.json"
    payload = {"protocol": PROTOCOL, "identity": manifest, "agent_config": agent_config, "dataset_config": dataset_config, "data": [], "metrics": {}, "averaged_metrics": {}, "time_cost": []}
    if output_path.exists() and not args.force:
        with output_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("identity") != manifest:
            raise ValueError("Result fingerprint mismatch; use a new artifact directory")
    completed = {r["qa_pair_id"]: r for r in payload["data"]}
    query_count = 0
    max_queries = int(args.max_test_queries_ablation or 0)

    def save(record_time=False):
        metrics = defaultdict(list)
        for row in payload["data"]:
            if row.get("status") != "success":
                continue
            for key, value in row.items():
                is_metric = key.startswith("memorydata_aux_") or key == "omni_choice_accuracy" or key in {"input_len", "output_len", "memory_construction_time", "query_time_len"}
                if is_metric and isinstance(value, (int, float)):
                    metrics[key].append(value)
        payload["metrics"] = dict(metrics)
        payload["averaged_metrics"] = {key: sum(values) / len(values) * (1 if "_len" in key or "_time" in key else 100) for key, values in metrics.items() if values}
        if record_time:
            payload["time_cost"].append(time.time() - started_run)
        atomic_json(output_path, payload)

    save()
    for unit_index, unit in enumerate(iter_eval_units(dataset_config)):
        if max_queries and query_count >= max_queries:
            break
        questions = unit["questions"]
        context_id = int(fingerprint([run_id, unit["sample_id"]])[:12], 16)
        if max_queries:
            questions = questions[:max_queries - query_count]
        query_count += len(questions)
        pending = [q for q in questions if q["question_id"] not in completed or (args.retry_failed_queries and completed[q["question_id"]].get("status") == "failed")]
        if not pending:
            print(f"Resume: {unit['sample_id']} already complete", flush=True)
            continue
        state_path = state_root / fingerprint(unit["sample_id"])[:20]
        unit_identity = {"run_id": run_id, "sample_id": unit["sample_id"], "history": fingerprint(unit["context_chunks"])}
        # Existing initializers use a directory's existence as a completion marker.
        # A separate marker proves ingestion finished; partial writes are rebuilt.
        marker = state_path.parent / (state_path.name + ".ready.json")
        reusable = marker.exists() and not args.force
        if reusable:
            with marker.open(encoding="utf-8") as stream:
                if json.load(stream) != unit_identity:
                    raise ValueError("Incompatible unit state; use a new artifact directory")
        unit_config = dict(agent_config, artifact_root=str(artifact_override), output_dir=str(output_dir), _omni_sample_id=unit["sample_id"], record_llm_io=True, zep_local_namespace_prefix="omni_" + run_id[:12])
        print(f"OmniMemEval: ingest/query {unit['sample_id']} ({unit['context_chars']} chars)", flush=True)
        agent = None
        try:
            agent, _ = initialize_and_memorize_agent(unit_config, dataset_config, str(state_path), unit["context_chunks"], context_id, unit_index + 1, force_rebuild=not reusable)
            atomic_json(marker, unit_identity)
            for q in pending:
                metadata = {key: value for key, value in q.items() if key not in {"question", "answer"}}
                metadata["question_text"] = q["question"]
                query = get_template(dataset_config["sub_dataset"], "query").format(question=format_question(q))
                started = time.perf_counter()
                query_id = int(fingerprint([run_id, q["question_id"]])[:12], 16)
                try:
                    # Only timing/task identity reaches the adapter; grading
                    # evidence and rubrics stay in the evaluator's metadata.
                    public_metadata = {key: metadata[key] for key in ("benchmark", "sample_id", "question_time", "protocol") if key in metadata}
                    response = agent.send_message(query, memorizing=False, query_id=query_id, context_id=context_id, eval_metadata=public_metadata)
                    if not isinstance(response, dict) or not str(response.get("output") or "").strip():
                        raise ValueError("Agent returned an empty answer")
                    _, records = metrics_summarization(response, query, q["answer"], dataset_config, defaultdict(list), [], query_id, context_id, q["question_id"], metadata)
                    row = dict(records[0], status="success")
                except Exception as exc:
                    row = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "qa_pair_id": q["question_id"], "query_id": query_id, "context_id": context_id, "output": "", "answer": q["answer"], "query": query, "eval_metadata": metadata}
                row.update(ingested_context_chars=unit["context_chars"], ingested_chunk_count=unit["context_chunk_count"], ingested_session_count=unit["session_count"], query_wall_time_s=time.perf_counter() - started, protocol=PROTOCOL)
                completed[q["question_id"]] = row
                payload["data"] = list(completed.values())
                save(record_time=True)
        finally:
            if agent is not None:
                release_agent(agent)
                del agent
            gc.collect()
    print(f"Results: {output_path}", flush=True)
    return output_path
