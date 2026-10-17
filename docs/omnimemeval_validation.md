# Validation record — 2026-10-18 (Asia/Shanghai)

Protocol: `memorydata_omnimemeval_v1`; upstream OmniMemEval `a5c8b4b`.

## Completed

- Created Python 3.11 environment at `D:\conda\envs\memorydata` and installed
  `requirements-omnimemeval.txt`; `pip check` reports no broken requirements.
- Downloaded the five default data variants using upstream preparation scripts.
- Parsed every evaluation unit in all five datasets, preserving stable IDs and
  checking sample/question uniqueness. Original datasets remain in the adjacent
  OmniMemEval repository and are excluded from Git by its existing rules.
- Passed 23 automated tests: schemas, annotation isolation, deterministic options,
  large JSONL lazy reading, valid-score parity with upstream pure functions,
  strict label failures/retries, answer/score resume and invalidation, original
  dataset dispatch/scoring/resume, and all 22 existing method template presets.
- Additional simulated-HTTP integration validation exercises all 15 combinations
  of five small synthetic datasets and the actual Long Context/Embedding RAG/Mem0
  adapters through `main.py`. Only model HTTP responses and tokenization are
  simulated; SDK request/response handling, FAISS/Qdrant, SQLite, scoring and
  answer/score resume execute normally. These are functional checks, not model
  accuracy results. The original LoCoMo main path is tested the same way.
- Original `main.py` imports and the new scoring CLI help both execute.
- All three smoke adapters (Long Context, Embedding RAG and Mem0) construct
  successfully without ingestion or inference requests. This also validates
  the vendored Mem0 distribution-metadata dependency (`mem0ai`).
- `git diff --check` passes.

| Dataset / default variant | Evaluation units | Questions |
| --- | ---: | ---: |
| LoCoMo / categories 1–4 | 10 | 1,540 |
| LongMemEval / S | 500 | 500 |
| PersonaMem v2 / benchmark, 32K | 200 | 5,000 |
| HaluMem / Medium | 20 | 3,467 |
| BEAM / 100k | 20 | 400 |
| Total | 750 | 10,907 |

Full-data parsing is an ingestion-format validation, not a model evaluation.

## Online acceptance blocked

The smoke runner successfully loads complete histories for the first selected
unit of each benchmark. All 15 benchmark × method combinations are marked
`blocked`: direct requests, bypassing proxies, to localhost ports **9908, 9909
and 9009** fail with Windows connection-refused error **10061**.

No real-model answers, judge scores or measured online accuracies are claimed.
Synthetic test answers and simulated judge scores are deliberately excluded
from online acceptance.
No replacement service or model was deployed. Once the configured services are
available, rerun `scripts/smoke_omnimemeval.ps1`; it executes real calls and
checks both answer and scoring resume behavior.

## Machine-readable evidence

- `results/logs/omnimemeval/data_validation.json`: all-record counts.
- `results/logs/omnimemeval/tests.log`: test outcomes.
- `results/logs/omnimemeval/environment_validation.json`: adapter construction.
- `results/logs/omnimemeval/simulated_validation.json`: 15 simulated pipeline cases.
- `results/logs/omnimemeval/verify.status`: interface/integration verification.
- `results/omnimemeval_smoke/summary.json`: input hashes, selected units,
  endpoint readiness and per-combination acceptance status.
- `results/logs/omnimemeval/setup.status` and `prepare.status`: both `success`.

Larger variants are configured but not downloaded or run. Non-smoke methods have
template compatibility checks only. See [usage and boundaries](omnimemeval.md).

## Interface and structure review

The review found and corrected mismatches in the initial extension:

- The public dataset loader now returns the original `{"data": ...}` sample
  schema and works with `ConversationCreator`, including short synthetic
  histories and single-question units. The bounded-memory main path remains.
- Main results retain the six original top-level fields, including `time_cost`.
  Quality averages use the original percentage convention; diagnostic flags
  and IDs are excluded from metrics. Scoring sidecars retain fractional scores.
- Output/state locations honor `output_dir` and `--artifact_root`, using the
  original `outputs/` and `agents/` hierarchy.
- Judge configuration moved to `evaluation/omnimemeval/config/qwen3_8b.yaml` so
  `config/*.yaml` continues to contain exactly the 22 original agent presets.
- Mem0's SQLite history connection is explicitly closed between units, fixing
  the Windows file lock found by actual adapter integration tests.
- Original RAG result mappings receive no new context-budget key unless the
  new benchmark explicitly enables it.

Checks against Git HEAD confirm no original tracked file was deleted or moved,
all 22 original agent presets are unchanged (ignoring checkout line endings),
and representative legacy score outputs match the original source. Nine
original files have integration/documentation hooks; existing benchmark and
method directories retain their locations. New functionality is contained in
new benchmark/evaluation modules and supporting tests, scripts and docs.
