"""Real-data smoke validation; never starts or substitutes inference services."""
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark.omnimemeval.common import atomic_json, file_digest
from benchmark.omnimemeval.loader import dataset_identity, iter_eval_units


def main():
    import yaml
    os.chdir(ROOT)
    os.environ.setdefault("OPENAI_API_KEY", "local")
    os.environ["MEM0_TELEMETRY"] = "False"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    output_root = ROOT / "results/omnimemeval_smoke"
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {"services": {}, "datasets": {}, "runs": []}
    endpoints = {9908: "Qwen3-8B", 9909: "Qwen3-8B", 9009: "Qwen3-Embedding-4B"}
    for port, expected in endpoints.items():
        try:
            request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models", headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]})
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=3) as response:
                models = [m["id"] for m in json.load(response)["data"]]
            summary["services"][str(port)] = {"ready": expected in models, "models": models, "expected": expected}
        except Exception as exc:
            summary["services"][str(port)] = {"ready": False, "expected": expected, "error": str(exc)}
    for kind in ("locomo", "longmemeval", "personamem_v2", "halumem", "beam"):
        config_path = ROOT / f"benchmark/omnimemeval/config/{kind}_smoke.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        try:
            identity = dataset_identity(config)
            unit = next(iter_eval_units(config))
            summary["datasets"][kind] = {"ready": True, "identity": identity, "sample_id": unit["sample_id"], "questions": [q["question_id"] for q in unit["questions"]], "context_chars": unit["context_chars"], "chunks": unit["context_chunk_count"]}
            del unit
        except Exception as exc:
            summary["datasets"][kind] = {"ready": False, "error": f"{type(exc).__name__}: {exc}"}
        for method, agent_file, ports in (("long_context", "reference_long_context_agent.yaml", [9908]), ("embedding_rag", "reference_embedding_rag.yaml", [9908, 9009]), ("mem0", "sequential_mem0.yaml", [9909, 9009, 9908])):
            run = {"benchmark": kind, "method": method}
            summary["runs"].append(run)
            missing = [port for port in ports if not summary["services"][str(port)]["ready"]]
            if missing or not summary["datasets"][kind]["ready"]:
                run.update(status="blocked", missing_service_ports=missing, data_ready=summary["datasets"][kind]["ready"])
                continue
            try:
                agent = yaml.safe_load((ROOT / "config" / agent_file).read_text(encoding="utf-8"))
                # Explicit backend ceiling for reliable fitting with local Qwen.
                agent["model_context_window"] = 32768
                agent_path = output_root / f"{method}.yaml"
                agent_path.write_text(yaml.safe_dump(agent, allow_unicode=True), encoding="utf-8")
                artifact_root = output_root / method / kind
                command = [sys.executable, "main.py", "--agent_config", str(agent_path), "--dataset_config", str(config_path), "--artifact_root", str(artifact_root), "--retry_failed_queries"]
                with (output_root / f"{method}_{kind}.log").open("w", encoding="utf-8") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                    results = list(artifact_root.rglob("benchmark_results.json"))
                    if len(results) != 1:
                        raise ValueError("Expected exactly one result file")
                    path = results[0]
                    result = json.loads(path.read_text(encoding="utf-8"))
                    expected = summary["datasets"][kind]["questions"]
                    actual = [row["qa_pair_id"] for row in result["data"]]
                    if actual != expected or any(row.get("status") != "success" for row in result["data"]):
                        raise ValueError("Answer coverage or query status check failed")
                    digest = file_digest(path)
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                    if file_digest(path) != digest:
                        raise ValueError("Resume changed completed answers")
                    scoring = [sys.executable, "-m", "evaluation.omnimemeval.evaluate", str(path), "--judge_config", "evaluation/omnimemeval/config/qwen3_8b.yaml"]
                    subprocess.run(scoring, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                    score_path = path.with_suffix(".omnimemeval_scores.json")
                    digest = file_digest(score_path)
                    subprocess.run(scoring, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                    if file_digest(score_path) != digest:
                        raise ValueError("Resume changed completed scores")
                run.update(status="passed", results=str(path), scores=str(score_path))
            except Exception as exc:
                run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            finally:
                atomic_json(output_root / "summary.json", summary)
    atomic_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(run["status"] == "passed" for run in summary["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
