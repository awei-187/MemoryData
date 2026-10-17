"""Score stored answers: python -m evaluation.omnimemeval.evaluate RESULTS --judge_config YAML."""
import argparse
import asyncio
import json
import os
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.omnimemeval.common import PROTOCOL, UPSTREAM, atomic_json, file_digest, fingerprint
from evaluation.omnimemeval import upstream_prompts as prompts
from evaluation.omnimemeval.scoring import PARSER_VERSION, binary_label, extract_choice_answer, nugget_score, ordering_score


def prompt_identity():
    return fingerprint({key: getattr(prompts, key) for key in ("JUDGE_SYSTEM_PROMPT", "JUDGE_PROMPT", "HM_JUDGE_PROMPT", "BEAM_RUBRIC_ITEM_JUDGE_PROMPT", "BEAM_EVENT_ORDERING_JUDGE_PROMPT")})


def build_calls(row):
    meta = row["eval_metadata"]
    kind = meta["benchmark"]
    text, response, answer = meta.get("question", row.get("query", "")), row["output"], row.get("answer", "")
    # question_text is kept as grading metadata, separate from the formatted query.
    text = meta.get("question_text", text)
    if kind == "personamem_v2":
        return []
    if kind != "beam":
        template = prompts.HM_JUDGE_PROMPT if kind == "halumem" else prompts.JUDGE_PROMPT
        return [("binary", [
            {"role": "system", "content": prompts.JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": template.format(question=text, golden_answer=answer, response=response)},
        ])]
    rubric = meta.get("rubric") or [f"LLM response should contain: {answer}"]
    if isinstance(rubric, str):
        try:
            parsed = json.loads(rubric)
        except json.JSONDecodeError:
            parsed = rubric
        rubric = parsed if isinstance(parsed, list) else [str(parsed)]
    if meta.get("dimension") == "event_ordering":
        reference = "\n".join(f"{i + 1}. {item}" for i, item in enumerate(rubric))
        content = prompts.BEAM_EVENT_ORDERING_JUDGE_PROMPT.format(question=text, reference_ordering=reference, response=response)
        return [(f"ordering:{len(rubric)}", [{"role": "system", "content": "You are a precise and fair evaluation judge."}, {"role": "user", "content": content}])]
    return [("nugget", [{"role": "system", "content": "You are a precise and fair evaluation judge."}, {"role": "user", "content": prompts.BEAM_RUBRIC_ITEM_JUDGE_PROMPT.format(question=text, rubric_item=item, response=response)}]) for item in rubric]


def summarize(rows):
    scored = [row for row in rows if row["status"] == "success"]
    total = len(rows)
    groups = {}
    for key in ("category", "question_type", "difficulty", "dimension", "variant"):
        values = {}
        for row in scored:
            value = row.get("eval_metadata", {}).get(key)
            if value is not None:
                values.setdefault(str(value), []).append(row["score"])
        groups[key] = {name: {"count": len(scores), "score": statistics.mean(scores)} for name, scores in values.items()}
    return {"total_questions": total, "successful_answers": sum(row.get("answer_status") == "success" for row in rows), "valid_scores": len(scored), "failed": sum(row["status"] == "failed" for row in rows), "skipped": sum(row["status"] == "skipped" for row in rows), "coverage": len(scored) / total if total else 0, "score": statistics.mean([row["score"] for row in scored]) if scored else None, "by_group": groups}


async def evaluate(result_path, config, overwrite=False, complete=None):
    result_path = Path(result_path)
    with result_path.open(encoding="utf-8") as stream:
        source = json.load(stream)
    if source.get("protocol") != PROTOCOL:
        raise ValueError("Expected MemoryData OmniMemEval results")
    if any("api_key" == key for key in config):
        raise ValueError("Use api_key_env rather than storing credentials in judge config")
    if float(config.get("temperature", 0)) != 0:
        raise ValueError("This scoring protocol requires temperature 0")
    num_runs = int(config.get("num_runs", 1))
    if num_runs < 1:
        raise ValueError("num_runs must be positive")
    resolved_base_url = config.get("base_url") or os.getenv(config.get("base_url_env", "OPENAI_BASE_URL"))
    identity = {"protocol": PROTOCOL, "upstream": UPSTREAM, "parser": PARSER_VERSION, "scoring_code": file_digest(Path(__file__).with_name("scoring.py")), "prompts": prompt_identity(), "judge_config": dict(config, resolved_base_url=resolved_base_url), "run_identity": source["identity"]}
    output = result_path.with_suffix(".omnimemeval_scores.json")
    cached = {}
    if output.exists() and not overwrite:
        with output.open(encoding="utf-8") as stream:
            previous = json.load(stream)
        if previous.get("identity") != identity:
            raise ValueError("Incompatible scoring configuration; use --overwrite or a new run directory")
        cached = {r["question_id"]: r for r in previous["rows"]}
    rows = source["data"]
    results = {}
    client = None
    semaphore = asyncio.Semaphore(max(1, int(config.get("max_concurrency", 2))))

    async def call(messages):
        nonlocal client
        if complete is not None:
            return await complete(messages)
        if client is None:
            from openai import AsyncOpenAI
            base_url = config.get("base_url") or os.getenv(config.get("base_url_env", "OPENAI_BASE_URL"))
            api_key = os.getenv(config.get("api_key_env", "OPENAI_API_KEY"))
            if not api_key and base_url and ("127.0.0.1" in base_url or "localhost" in base_url):
                api_key = "local"
            if not api_key:
                raise RuntimeError("Judge API key environment variable is not set")
            client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=float(config.get("timeout", 120)))
        kwargs = dict(model=config["model"], messages=messages, temperature=0, max_tokens=int(config.get("max_tokens", 2048)))
        if "qwen3" in config["model"].lower():
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        response = await client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    async def score(row):
        key = str(row["qa_pair_id"])
        content_hash = fingerprint({"question": row.get("eval_metadata"), "answer": row.get("answer"), "output": row.get("output"), "status": row.get("status")})
        old = cached.get(key)
        if old and old.get("input_fingerprint") == content_hash and old["status"] == "success":
            return key, old
        item = {"question_id": key, "input_fingerprint": content_hash, "eval_metadata": row.get("eval_metadata", {}), "answer_status": row.get("status", "failed"), "raw_responses": [], "attempts": []}
        if row.get("status") != "success":
            return key, dict(item, status="skipped", reason="answer_not_successful")
        try:
            if row["eval_metadata"]["benchmark"] == "personamem_v2":
                correct, parsed = extract_choice_answer(row["output"], str(row["answer"]))
                return key, dict(item, status="success", score=float(correct), parsed_choice=parsed)
            calls = build_calls(row)
            values = []
            for item_index, (parser, messages) in enumerate(calls):
                for run_index in range(num_runs):
                    for attempt in range(3):
                        raw = None
                        try:
                            async with semaphore:
                                raw = await call(messages)
                            value = float(binary_label(raw)) if parser == "binary" else nugget_score(raw) if parser == "nugget" else ordering_score(raw, int(parser.split(":")[1]))[0]
                            record = {"parser": parser, "item_index": item_index, "run": run_index, "raw": raw, "score": value}
                            if parser.startswith("ordering:"):
                                record["positions"] = ordering_score(raw, int(parser.split(":")[1]))[1]
                            item["raw_responses"].append(record)
                            values.append(value)
                            break
                        except Exception as exc:
                            item["attempts"].append({"parser": parser, "run": run_index, "attempt": attempt + 1, "raw": raw, "error": f"{type(exc).__name__}: {exc}"})
                            if attempt == 2:
                                raise
                            await asyncio.sleep(min(2 ** attempt, 2))
            return key, dict(item, status="success", score=statistics.mean(values), scoring_method="kendall_tau_b" if calls[0][0].startswith("ordering:") else "per_rubric_item" if row["eval_metadata"]["benchmark"] == "beam" else "judge_accuracy")
        except Exception as exc:
            return key, dict(item, status="failed", error=f"{type(exc).__name__}: {exc}")

    def save():
        ordered = [results[str(row["qa_pair_id"])] for row in rows if str(row["qa_pair_id"]) in results]
        atomic_json(output, {"identity": identity, "rows": ordered, "summary": summarize(ordered), "pending": len(rows) - len(ordered)})

    try:
        # Bounded batches: do not create millions of tasks for large results.
        for start in range(0, len(rows), max(1, int(config.get("max_concurrency", 2)))):
            tasks = [asyncio.create_task(score(row)) for row in rows[start:start + max(1, int(config.get("max_concurrency", 2)))]]
            for task in asyncio.as_completed(tasks):
                key, value = await task
                results[key] = value
                save()
        save()
    finally:
        if client is not None:
            await client.close()
    return output, summarize(list(results.values()))


def main():
    import yaml
    from dotenv import load_dotenv
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_files", nargs="+")
    parser.add_argument("--judge_config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    with open(args.judge_config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for path in args.result_files:
        output, summary = asyncio.run(evaluate(path, config, args.overwrite))
        print(json.dumps({"path": str(output), **summary}, ensure_ascii=False))
        if summary["failed"] or summary["skipped"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
