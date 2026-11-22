"""HaluMem question-answering judge.

Uses the official EVALUATION_PROMPT_FOR_QUESTION (see upstream_prompts.py) and
the same parsing rule as upstream eval/llms.py: the first ```json block must
carry "evaluation_result" in Correct | Hallucination | Omission. Ratios are
reported with two denominators, as upstream evaluation.py does:
"all" counts every HaluMem result row (answer failures and unparseable judge
replies count as not Correct), "valid" counts only rows with a judged label.

Results go to a sidecar next to the MemoryData result file; the result file
itself is never modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from evaluation.halumem.upstream_prompts import EVALUATION_PROMPT_FOR_QUESTION

LABELS = ("Correct", "Hallucination", "Omission")
DEFAULT_JUDGE_MODEL = "gpt-4o"
JUDGE_MODEL_ENV = "HALUMEM_JUDGE_MODEL"
SIDECAR_SUFFIX = "_halumem_judge.json"
UPSTREAM_COMMIT = "d653749586bfc7d13f5b68e8851f52155b5b1650"
PROTOCOL = "memorydata_halumem_qa_judge_v1"
# Same extraction as upstream eval/llms.py llm_request_for_json.
JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_judge_model(explicit: str | None = None) -> str:
    return (explicit or os.environ.get(JUDGE_MODEL_ENV) or DEFAULT_JUDGE_MODEL).strip()


def sidecar_path_for(result_path: str | os.PathLike[str]) -> Path:
    path = Path(result_path)
    stem = path.name[:-5] if path.name.endswith(".json") else path.name
    return path.with_name(stem + SIDECAR_SUFFIX)


def build_judge_prompt(question: str, reference_answer: str, key_memory_points: str, response: str) -> str:
    return EVALUATION_PROMPT_FOR_QUESTION.format(
        question=question, reference_answer=reference_answer,
        key_memory_points=key_memory_points, response=response,
    )


def parse_judge_response(text: str) -> tuple[str, str | None]:
    """Return (label, reasoning); raise ValueError when the reply breaks the upstream format."""
    match = JSON_BLOCK.search(text or "")
    if not match:
        raise ValueError("judge reply has no ```json block")
    payload = json.loads(match.group(1))
    label = payload.get("evaluation_result")
    if label not in LABELS:
        raise ValueError(f"judge label {label!r} is not one of {LABELS}")
    return label, payload.get("reasoning")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def extract_halumem_rows(result_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """One judge item per HaluMem question; the last successful answer wins over earlier failures."""
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(result_payload.get("data") or []):
        metadata = row.get("eval_metadata") or {}
        if metadata.get("dataset") != "halumem_qa":
            continue
        key = str(row.get("qa_pair_id") or metadata.get("qa_pair_id"))
        failed = row.get("status") == "failed" or row.get("output") is None
        item = {
            "qa_pair_id": key,
            "source_row_index": index,
            "question": str(metadata.get("question") or ""),
            "reference_answer": str(row.get("answer") if not isinstance(row.get("answer"), list) else row["answer"][0]),
            "key_memory_points": "\n".join(str(e.get("memory_content", "")) for e in metadata.get("evidence") or []),
            "response": "" if failed else str(row.get("output") or ""),
            "question_type": str(metadata.get("question_type") or "unknown"),
            "difficulty": str(metadata.get("difficulty") or "unknown"),
            "answer_failed": failed,
        }
        item["input_sha256"] = _digest([item["question"], item["reference_answer"], item["key_memory_points"], item["response"]])
        previous = by_id.get(key)
        if previous is None or previous["answer_failed"] or not failed:
            by_id[key] = item
    return list(by_id.values())


def _ratios(items: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in items if item.get("label") in LABELS]
    block = {"total": len(items), "valid": len(valid),
             "answer_failed": sum(1 for item in items if item.get("answer_failed")),
             "judge_failed": sum(1 for item in items if not item.get("answer_failed") and item.get("label") not in LABELS)}
    for label in LABELS:
        count = sum(1 for item in valid if item["label"] == label)
        key = label.lower()
        block[f"{key}_count"] = count
        block[f"{key}_ratio(all)"] = count / len(items) if items else None
        block[f"{key}_ratio(valid)"] = count / len(valid) if valid else None
    return block


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, list]] = {"question_type": defaultdict(list), "difficulty": defaultdict(list)}
    for item in items:
        for field in groups:
            groups[field][item[field]].append(item)
    return {
        "overall": _ratios(items),
        "by_question_type": {name: _ratios(group) for name, group in sorted(groups["question_type"].items())},
        "by_difficulty": {name: _ratios(group) for name, group in sorted(groups["difficulty"].items())},
    }


def create_judge_client():
    """OpenAI-compatible client from the project's shared OPENAI_* settings."""
    import httpx
    from openai import OpenAI
    from utils.provider_utils import model_api_request_options, model_api_trust_env, resolve_base_url, resolve_env_value

    api_key = str(resolve_env_value("OPENAI_API_KEY", ["OPENAI_API_KEY"]) or "").strip()
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY (root .env) for the HaluMem judge")
    base_url = resolve_base_url(None, "OPENAI_BASE_URL")
    kwargs = {"api_key": api_key, "http_client": httpx.Client(trust_env=model_api_trust_env()), **model_api_request_options()}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def judge_item(client, judge_model: str, item: dict[str, Any], max_attempts: int = 3,
               sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Judge one answer; API and format errors are retried, then reported as judge_error."""
    prompt = build_judge_prompt(item["question"], item["reference_answer"], item["key_memory_points"], item["response"])
    raw = None
    errors = []
    for attempt in range(1, max_attempts + 1):
        try:
            completion = client.chat.completions.create(
                model=judge_model, messages=[{"role": "user", "content": prompt}], temperature=0,
            )
            raw = completion.choices[0].message.content or ""
            label, reasoning = parse_judge_response(raw)
            return {**item, "label": label, "reasoning": reasoning, "raw_judge_response": raw,
                    "attempts": attempt, "judge_error": None}
        except Exception as exc:  # retried like upstream tenacity, which also retries parse errors
            errors.append(f"{type(exc).__name__}: {exc}")
            if attempt < max_attempts:
                sleep(min(30.0, 2.0 ** attempt))
    return {**item, "label": None, "reasoning": None, "raw_judge_response": raw,
            "attempts": max_attempts, "judge_error": errors[-1] if errors else "unknown"}


def judge_result_file(result_path: str | os.PathLike[str], *, client=None, judge_model: str | None = None,
                      max_concurrency: int = 8, max_attempts: int = 3, overwrite: bool = False,
                      sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Judge every HaluMem answer in one MemoryData result file and write the sidecar."""
    result_path = Path(result_path)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    items = extract_halumem_rows(payload)
    if not items:
        raise ValueError(f"{result_path} contains no HaluMem result rows")
    judge_model = resolve_judge_model(judge_model)
    sidecar_path = sidecar_path_for(result_path)

    previous = {}
    if sidecar_path.exists() and not overwrite:
        existing = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if existing.get("judge_model") == judge_model and existing.get("protocol") == PROTOCOL:
            previous = {row["qa_pair_id"]: row for row in existing.get("rows", [])
                        if row.get("label") in LABELS}

    judged: dict[str, dict[str, Any]] = {}
    pending = []
    for item in items:
        reuse = previous.get(item["qa_pair_id"])
        if item["answer_failed"]:
            judged[item["qa_pair_id"]] = {**item, "label": None, "judge_error": "answer failed; not judged"}
        elif reuse and reuse.get("input_sha256") == item["input_sha256"]:
            judged[item["qa_pair_id"]] = reuse
        else:
            pending.append(item)

    def write():
        rows = [judged[item["qa_pair_id"]] for item in items if item["qa_pair_id"] in judged]
        sidecar = {
            "protocol": PROTOCOL, "judge_model": judge_model, "judge_temperature": 0,
            "prompt_source": f"MemTensor/HaluMem@{UPSTREAM_COMMIT} eval/eval_tools.py EVALUATION_PROMPT_FOR_QUESTION",
            "result_file": str(result_path), "updated_at": utc_now_iso(),
            "complete": len(rows) == len(items),
            "summary": summarize(rows) if len(rows) == len(items) else None,
            "rows": rows,
        }
        temporary = sidecar_path.with_name(sidecar_path.name + ".tmp")
        temporary.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, sidecar_path)
        return sidecar

    if pending:
        client = client or create_judge_client()
        with ThreadPoolExecutor(max_workers=max(1, int(max_concurrency))) as pool:
            futures = [pool.submit(judge_item, client, judge_model, item, max_attempts, sleep) for item in pending]
            for done, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                judged[row["qa_pair_id"]] = row
                if done % 20 == 0:
                    write()
    return write()
