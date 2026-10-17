"""Construct the three smoke adapters without ingestion or model requests."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("OPENAI_API_KEY", "local")
os.environ["MEM0_TELEMETRY"] = "False"
os.environ["MEM0_DIR"] = str(ROOT / "results/dependency_check/mem0_home")

import yaml
from benchmark.omnimemeval.common import atomic_json
from benchmark.omnimemeval.runtime import release_agent
from methods.embedding_rag.embedding_retriever import TextRetriever, RAGSystem
from utils.agent import AgentWrapper


def main():
    dataset = yaml.safe_load((ROOT / "benchmark/omnimemeval/config/locomo_smoke.yaml").read_text(encoding="utf-8"))
    report = {}
    for name, file in (("long_context", "reference_long_context_agent.yaml"), ("embedding_rag", "reference_embedding_rag.yaml"), ("mem0", "sequential_mem0.yaml")):
        try:
            config = yaml.safe_load((ROOT / "config" / file).read_text(encoding="utf-8"))
            agent = AgentWrapper(config, dataset, str(ROOT / "results/dependency_check" / name))
            release_agent(agent)
            del agent
            report[name] = {"status": "passed"}
        except Exception as exc:
            report[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        print(name + ": " + json.dumps(report[name]), flush=True)
    atomic_json(ROOT / "results/logs/omnimemeval/environment_validation.json", report)
    return 0 if all(row["status"] == "passed" for row in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
