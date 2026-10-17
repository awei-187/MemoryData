# OmniMemEval integration

[中文使用指南](omnimemeval_zh.md) · [Change summary](omnimemeval_changes.md)

Protocol: `memorydata_omnimemeval_v1`. Data/score definitions are adapted from
OmniMemEval commit `a5c8b4b`. This evaluates MemoryData's existing methods under
its own ingestion and answer pipeline; scores do not reproduce the complete
OmniMemEval execution protocol or its published rankings.

## Environment and data

Python 3.11 environment: `D:\conda\envs\memorydata`. The setup script creates
this environment and installs `requirements-omnimemeval.txt`. The preparation
script uses the adjacent OmniMemEval repository's existing preparation scripts.
Defaults download LoCoMo, LongMemEval S, PersonaMem benchmark/32K histories,
HaluMem Medium and BEAM 100k. Data remain in `../OmniMemEval/data`; override
`data_root` in the benchmark YAML when needed. Relative paths resolve against
the MemoryData repository, not the shell's current directory.

Run these single-line commands in Windows PowerShell. Each opens an independent
visible window, runs the operation there and retains the window. Logs and status
files are in `results/logs/omnimemeval`.

```powershell
Start-Process powershell.exe -WindowStyle Normal -ArgumentList '-NoProfile -NoExit -ExecutionPolicy Bypass -File "E:\LLM judge\MemoryData\scripts\setup_omnimemeval.ps1"'
```

```powershell
Start-Process powershell.exe -WindowStyle Normal -ArgumentList '-NoProfile -NoExit -ExecutionPolicy Bypass -File "E:\LLM judge\MemoryData\scripts\prepare_omnimemeval.ps1"'
```

## Running

The dataset configs in `benchmark/omnimemeval/config` use `dataset: OmniMemEval`.
Select `locomo`, `longmemeval`, `personamem_v2`, `halumem` or `beam` smoke YAML
for one complete history and at most three questions (one for LongMemEval).
BEAM smoke includes rubric, event-ordering and a third capability, in source order.

Full configs have `max_test_samples: 0` and `max_queries_per_unit: 0` (unlimited).
`sample_ids` accepts stable IDs, e.g. `conv-26` for LoCoMo or `persona:7` for
PersonaMem. Filtering preserves source order and question IDs. Configs also
cover LongMemEval oracle/M, PersonaMem 128K, HaluMem Long and BEAM 500k/1m/10m;
these variants require their corresponding upstream downloads.

```powershell
cd 'E:\LLM judge\MemoryData'
```

```powershell
& 'D:\conda\envs\memorydata\python.exe' main.py --agent_config config/reference_long_context_agent.yaml --dataset_config benchmark/omnimemeval/config/locomo_smoke.yaml --artifact_root results/omni_runs
```

Results live under the original `outputs/<model>/OmniMemEval/` hierarchy, with
`<sub_dataset>/<run-fingerprint>/` below it. Per-unit memory uses the original
`agents/` hierarchy at `agents/omnimemeval/<sub_dataset>/<run-fingerprint>/`.
The existing agent `output_dir` is honored; `--artifact_root` rebases the
standard `results/` prefix as it does for the original benchmarks. The manifest
records source file SHA-256, the execution configuration, integration code
hashes and upstream version. Changes select a different run directory. Within
a run, incompatible identities are rejected. Partial ingestion is rebuilt;
only a completed ingestion marker permits memory reuse. Answer failures are
saved; use `--retry_failed_queries` to retry them. `--force` rebuilds the run.

## Scoring

```powershell
& 'D:\conda\envs\memorydata\python.exe' -m evaluation.omnimemeval.evaluate 'results/omni_runs/outputs/Qwen3-8B/OmniMemEval/omni_locomo/<run-fingerprint>/benchmark_results.json' --judge_config evaluation/omnimemeval/config/qwen3_8b.yaml
```

Replace the example result path with the one printed by `main.py`. The sidecar
is `<result-stem>.omnimemeval_scores.json`. Successful scores resume only when
their question/answer content and scoring identities match. Use `--overwrite`
to rerun with a different judge configuration. Original answers remain intact.

Answer and embedding clients use `OPENAI_API_KEY` as configured by existing
presets. Set it to your service credential; an unauthenticated local service can
use the value `local`. The smoke script supplies `local` only if the variable
is unset. Judge likewise accepts this placeholder for loopback endpoints.

| Dataset | Primary scoring |
| --- | --- |
| LoCoMo, LongMemEval | Upstream binary Judge Accuracy |
| PersonaMem v2 | Upstream option extraction/Accuracy; no LLM judge |
| HaluMem | QA Judge Accuracy; no extraction/update operation score |
| BEAM | Per-rubric nugget scores; Kendall tau-b × coverage for event ordering |

Scoring-sidecar scores and per-question scores are fractions in [0, 1]. The
main result's `averaged_metrics` follows the original percentage convention
for quality scores and raw units for length/time. The original `time_cost`
field is retained. Auxiliary lexical scores use the
`memorydata_aux_` prefix and retain MemoryData's formulas, not OmniMemEval's F1
definition. No BERTScore, BLEU or METEOR is calculated. Unsupported evidence
mapping does not produce recall metrics. Existing LongMemEval judge remains
available through its original entrypoint.

Prompts are copied unchanged from upstream; parsing is versioned separately as
`omni_strict_v1`. Ambiguous/invalid labels and unparseable BEAM responses fail
scoring, never silently become zero. Judge uses temperature 0, concurrency 2,
one run and up to two retries per call. Sidecars preserve raw successful and
failed responses, per-item/run scores, prompt identity and model configuration.
Summary reports answer success, scoring failure/skip counts, valid coverage and
group scores. The main score's denominator is the number of valid scored answers.

## Execution boundaries and validation

Only conversation messages enter memory. PersonaMem options are visible to the
answerer; correct option labels, rubric, evidence and annotated memory points
are evaluator-only. Each persona/user/conversation has isolated persistent state.
HaluMem uses all sessions before QA, matching this upstream implementation;
it is not a time-cutoff evaluation. Historical timestamps are preserved; missing
times do not receive the execution date. Memory backends can still attach their
own system creation timestamps, which are distinct from event times.

The full selected history is ingested, including for RAG. Answer input may be
trimmed for model limits; results include ingested chars/chunks/sessions,
`answer_context_budget` and `answer_context_truncated`. Measurements of tokens
and runtime retain each MemoryData adapter's counting scope. Retained history,
retrieved top-k and answer budgets are different experimental quantities.

```powershell
Start-Process powershell.exe -WindowStyle Normal -ArgumentList '-NoProfile -NoExit -ExecutionPolicy Bypass -File "E:\LLM judge\MemoryData\scripts\verify_omnimemeval.ps1"'
```

```powershell
Start-Process powershell.exe -WindowStyle Normal -ArgumentList '-NoProfile -NoExit -ExecutionPolicy Bypass -File "E:\LLM judge\MemoryData\scripts\smoke_omnimemeval.ps1"'
```

Smoke runs the 15 combinations of five datasets × Long Context/Embedding
RAG/Mem0, then repeats answer and scoring commands to check resume behavior.
It checks `/v1/models` on 9908 and 9909 for Qwen3-8B and 9009 for
Qwen3-Embedding-4B. Missing services block the affected combinations; the script
does not deploy or replace services. Qwen3-8B is a functional judge smoke test,
not a validated judge for scientific rankings. Each combination's state is
recorded in `results/omnimemeval_smoke/summary.json`.

The separate `verify_omnimemeval.ps1` check needs no model service. It runs 23
tests, including all 15 combinations with in-memory simulated model HTTP
responses, actual method adapters and vector stores, scoring, and resume.
Its machine-readable case report is
`results/logs/omnimemeval/simulated_validation.json`; these synthetic results
do not count as real-model smoke acceptance or model-quality scores.

The remaining method presets receive template compatibility checks, not online
certification. Full-scale runs and published-score comparisons are outside the
smoke acceptance. Source attribution and Apache-2.0 license/notice are retained
under `evaluation/omnimemeval`; dataset terms remain those of their sources.

Additional local checks: `scripts/validate_omnimemeval.py` parses all records of
the downloaded defaults and checks stable-ID uniqueness;
`scripts/check_omnimemeval_environment.py` constructs the three smoke adapters
without ingesting history or calling inference services. See
[the implementation validation record](omnimemeval_validation.md).

`load_eval_data` exposes new benchmarks through the existing `{"data": ...}`
sample schema (context chunks, question/answer arrays and evaluation metadata).
It can be consumed by the original `ConversationCreator`. The main entrypoint
uses a dedicated iterator to avoid materializing all histories. Judge configs
live under `evaluation/omnimemeval/config`, leaving the 22 top-level agent
presets in `config/` unchanged.
