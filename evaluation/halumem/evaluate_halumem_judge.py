"""Judge HaluMem answers in MemoryData result files with the official QA prompt.

Example:
    python -m evaluation.halumem.evaluate_halumem_judge results/outputs/.../halumem_..._results.json
"""

import argparse
import json
from pathlib import Path

import dotenv

ROOT = Path(__file__).resolve().parents[2]
dotenv.load_dotenv(ROOT / ".env")

from evaluation.halumem.halumem_judge import (  # noqa: E402
    JUDGE_MODEL_ENV, extract_halumem_rows, judge_result_file, resolve_judge_model,
)
def expand_result_paths(result_files, result_globs):
    """Explicit paths plus recursive globs (``**`` matches nested output folders), deduplicated."""
    import glob
    import os
    paths = [os.path.normpath(path) for path in result_files]
    for pattern in result_globs:
        paths.extend(os.path.normpath(path) for path in glob.glob(pattern, recursive=True))
    return sorted(set(path for path in paths if not path.endswith("_halumem_judge.json")))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("result_files", nargs="*", help="MemoryData *_results.json files from HaluMem runs")
    parser.add_argument("--result_glob", action="append", default=[], help="Glob for result files; repeatable")
    parser.add_argument("--judge-model", default=None,
                        help=f"Judge model; default ${JUDGE_MODEL_ENV} or gpt-4o")
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=3, help="Attempts per answer, including format errors")
    parser.add_argument("--overwrite", action="store_true", help="Ignore previously judged rows")
    parser.add_argument("--dry-run", action="store_true", help="Only count judge items; no API calls")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = expand_result_paths(args.result_files, args.result_glob)
    if not paths:
        raise SystemExit("No result files given")
    judge_model = resolve_judge_model(args.judge_model)
    for path in paths:
        if args.dry_run:
            items = extract_halumem_rows(json.loads(Path(path).read_text(encoding="utf-8")))
            print(json.dumps({"result_file": path, "judge_model": judge_model, "items": len(items),
                              "answer_failed": sum(i["answer_failed"] for i in items)}, ensure_ascii=False))
            continue
        sidecar = judge_result_file(path, judge_model=judge_model, max_concurrency=args.max_concurrency,
                                    max_attempts=args.max_attempts, overwrite=args.overwrite)
        overall = (sidecar["summary"] or {}).get("overall")
        print(json.dumps({"result_file": path, "judge_model": judge_model, "complete": sidecar["complete"],
                          "overall": overall}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
