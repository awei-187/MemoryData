"""Validate all records of the five downloaded default data variants."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from benchmark.omnimemeval.common import atomic_json
from benchmark.omnimemeval.loader import iter_eval_units


def main():
    report = {}
    defaults = {"locomo": "default", "longmemeval": "s", "personamem_v2": "32k", "halumem": "medium", "beam": "100k"}
    for kind, variant in defaults.items():
        try:
            config = yaml.safe_load((ROOT / f"benchmark/omnimemeval/config/{kind}_{variant}.yaml").read_text(encoding="utf-8"))
            samples, questions = set(), set()
            chars = chunks = 0
            for unit in iter_eval_units(config):
                if unit["sample_id"] in samples:
                    raise ValueError("Duplicate sample ID: " + unit["sample_id"])
                samples.add(unit["sample_id"])
                chars += unit["context_chars"]
                chunks += unit["context_chunk_count"]
                for question in unit["questions"]:
                    if question["question_id"] in questions:
                        raise ValueError("Duplicate question ID: " + question["question_id"])
                    questions.add(question["question_id"])
            report[kind] = {"status": "passed", "units": len(samples), "questions": len(questions), "context_chars": chars, "chunks": chunks}
        except Exception as exc:
            report[kind] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        print(kind + ": " + json.dumps(report[kind]), flush=True)
    atomic_json(ROOT / "results/logs/omnimemeval/data_validation.json", report)
    return 0 if all(row["status"] == "passed" for row in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
