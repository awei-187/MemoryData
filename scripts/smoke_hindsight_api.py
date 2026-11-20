"""Tiny real-service API smoke test, using synthetic data rather than a benchmark ranking."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_root", default=str(ROOT / "results/hindsight_api_smoke" / str(time.time_ns())))
    parser.add_argument("--agent_config", default=str(ROOT / "config/hybrid_hindsight.yaml"))
    args = parser.parse_args()
    run = Path(args.artifact_root).resolve()
    run.mkdir(parents=True, exist_ok=False)
    history = ("Mira: My preferred beverage is jasmine tea. I do not like coffee.\n"
               + "Assistant: This is a routine acknowledgment; no new personal fact is introduced.\n" * 30)
    sample = [{"sample_id": "synthetic-api-check", "conversation": {"speaker_a": "Mira", "speaker_b": "Assistant",
               "session_1_date_time": "2 January 2024", "session_1": [{"speaker": "Mira", "dia_id": "D1:1", "text": history}]},
               "qa": [{"question": "What is Mira's preferred beverage?", "answer": "jasmine tea", "category": 4, "evidence": ["D1:1"]}]}]
    (run / "input.json").write_text(json.dumps(sample), encoding="utf-8")
    dataset = yaml.safe_load((ROOT / "benchmark/locomo/config/Locomo_qa_4cat_600_dist.yaml").read_text(encoding="utf-8"))
    dataset.update(test_files=str(run / "input.json"), max_test_samples=1, tag="synthetic_api_check")
    (run / "dataset.yaml").write_text(yaml.safe_dump(dataset), encoding="utf-8")
    command = [sys.executable, str(ROOT / "main.py"), "--agent_config", args.agent_config,
               "--dataset_config", str(run / "dataset.yaml"), "--artifact_root", str(run / "artifacts")]
    report = {"mode": "real_service_remote_api_synthetic_input", "status": "running", "run_dir": str(run)}
    summary = run / "summary.json"

    def save():
        summary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save()
    try:
        subprocess.run(command, cwd=ROOT, check=True)
        results = list((run / "artifacts/outputs").rglob("*_results.json"))
        if len(results) != 1:
            raise RuntimeError("Expected exactly one result file")
        result = results[0]
        payload = json.loads(result.read_text(encoding="utf-8"))
        if len(payload["data"]) != 1 or not payload["data"][0].get("output") or payload["data"][0].get("status") == "failed":
            raise RuntimeError("The real-service pipeline did not produce a successful answer")
        digest = hashlib.sha256(result.read_bytes()).hexdigest()
        subprocess.run(command, cwd=ROOT, check=True)
        if hashlib.sha256(result.read_bytes()).hexdigest() != digest:
            raise RuntimeError("No-op answer resume changed the result file")
        report.update(status="passed", result_file=str(result), resume_unchanged=True,
                      answer_model=payload["agent_config"]["model"], answer=payload["data"][0]["output"])
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        save()
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
