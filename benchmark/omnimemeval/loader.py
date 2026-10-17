"""Lazy readers. Only dialogue content is exposed to memory backends.

Schemas and PersonaMem option permutation adapted from OmniMemEval a5c8b4b.
"""
import ast
import csv
import json
import random
from collections import OrderedDict
from pathlib import Path

from .common import PROTOCOL, REPO_ROOT, file_digest

BEAM_FILES = {"100k": "beam_100k.json", "500k": "beam_500k.json", "1m": "beam_1m.json", "10m": "beam_10m_10m.json"}


def load_omnimemeval_eval_data(config):
    """Original loader contract: data contains canonical MemoryData samples.

    The main runner consumes iter_eval_units directly to retain bounded memory.
    ConversationCreator can consume this compatibility view for existing tools.
    """
    from .templates import format_question

    def samples():
        for unit in iter_eval_units(config):
            questions = unit["questions"]
            metadata = [dict({key: value for key, value in q.items() if key not in {"question", "answer"}}, question_text=q["question"]) for q in questions]
            yield {
                "source": config["sub_dataset"],
                "sample_id": unit["sample_id"],
                "context_chunks": unit["context_chunks"],
                "context_length": unit["context_chars"],
                "questions": [format_question(q) for q in questions],
                "answers": [q["answer"] for q in questions],
                "question_ids": [q["question_id"] for q in questions],
                "qa_pair_ids": [q["question_id"] for q in questions],
                "question_types": [q.get("question_type", q.get("category", q.get("dimension"))) for q in questions],
                "eval_metadata": metadata,
            }
    return {"data": samples()}


def parse_structured(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def dataset_files(config):
    root = resolve_path(config.get("data_root", "../OmniMemEval/data"))
    kind = config["sub_dataset"].removeprefix("omni_")
    if config.get("test_files"):
        primary = resolve_path(config["test_files"])
    elif kind == "locomo":
        primary = root / "locomo/locomo10.json"
    elif kind == "longmemeval":
        variant = config.get("variant", "s")
        primary = root / "longmemeval" / ("longmemeval_oracle.json" if variant == "oracle" else f"longmemeval_{variant}_cleaned.json")
    elif kind == "personamem_v2":
        primary = root / "personamem_v2/benchmark/text" / f"{config.get('split', 'benchmark')}.csv"
    elif kind == "halumem":
        primary = root / "halumem" / f"HaluMem-{config.get('variant', 'medium').title()}.jsonl"
    elif kind == "beam":
        primary = root / "beam" / BEAM_FILES[config.get("variant", "100k")]
    else:
        raise ValueError(f"Unknown OmniMemEval benchmark: {kind}")
    files = [primary]
    if kind == "personamem_v2":
        history_root = resolve_path(config["history_dir"]) if config.get("history_dir") else root / "personamem_v2/data" / f"chat_history_{config.get('variant', '32k')}"
        if not history_root.is_dir():
            raise FileNotFoundError(f"PersonaMem histories missing: {history_root}")
        files.extend(sorted(history_root.glob("*.json")))
        if len(files) == 1:
            raise FileNotFoundError(f"No PersonaMem histories: {history_root}")
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Dataset missing: {path}. Run the corresponding OmniMemEval data preparation script.")
    return kind, files


def dataset_identity(config):
    kind, files = dataset_files(config)
    return {"benchmark": kind, "protocol": PROTOCOL, "files": [{"path": str(p.resolve()), "sha256": file_digest(p)} for p in files]}


def read_array(path):
    import ijson
    with Path(path).open("rb") as stream:
        yield from ijson.items(stream, "item", use_float=True)


def read_lines(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def flatten_beam_session(session):
    # BEAM sessions may be lists, or nested plan -> batches -> turns.
    if isinstance(session, list):
        return [message for turn in session for message in (turn if isinstance(turn, list) else [turn])]
    if isinstance(session, dict) and "content" in session:
        return [session]
    messages = []
    for key in sorted(session):
        for batch in session[key] or []:
            for turn in batch.get("turns", []):
                messages.extend(turn if isinstance(turn, list) else [turn])
    return messages


def dialogue_chunks(sample_id, sessions, chunk_size):
    chunks, lines, sources, timestamps = [], [], [], []
    def flush():
        if lines:
            chunks.append({"text": "\n".join(lines), "source_ids": sources.copy(), "chunk_id": f"{sample_id}:chunk:{len(chunks)}", "timestamp": timestamps[0] if timestamps else None})
            lines.clear()
            sources.clear()
            timestamps.clear()
    for session_index, (session_id, date, messages) in enumerate(sessions):
        flush()
        for turn_index, message in enumerate(messages):
            content = str(message.get("content", message.get("text", "")) or "").replace("<|endoftext|>", "")
            if not content.strip() or message.get("role") == "system":
                continue
            role = message.get("speaker", message.get("role", "user"))
            time = message.get("time_anchor") or date
            prefix = f"Session {session_id}" + (f" | Time: {time}" if time else "")
            if message.get("blip_caption"):
                content += f" [Image: {message['blip_caption']}]"
            header = f"[{prefix}] {role}: "
            width = max(1, chunk_size - len(header))
            for offset in range(0, len(content), width):
                line = header + content[offset:offset + width]
                if lines and sum(map(len, lines)) + len(line) > chunk_size:
                    flush()
                lines.append(line)
                timestamps.append(time)
                sources.append(str(message.get("dia_id") or f"{sample_id}:s{session_index}:t{turn_index}"))
    flush()
    return chunks


def question(sample_id, question_id, text, answer, **metadata):
    return {"question": str(text), "answer": answer, "question_id": str(question_id), "sample_id": sample_id, **metadata}


def convert_locomo(row, index, config):
    sample_id = str(row.get("sample_id", f"locomo:{index}"))
    conv = row["conversation"]
    sessions = [(key, conv.get(key + "_date_time"), conv[key]) for key in sorted((k for k in conv if k.startswith("session_") and k[8:].isdigit()), key=lambda k: int(k[8:]))]
    categories = {str(x) for x in config.get("categories", [1, 2, 3, 4])}
    questions = [question(sample_id, f"{sample_id}:q{i}", q["question"], q.get("answer", q.get("adversarial_answer")), category=str(q["category"]), evidence=q.get("evidence", [])) for i, q in enumerate(row["qa"]) if str(q["category"]) in categories]
    return sample_id, sessions, questions


def convert_longmemeval(row, index, config):
    sample_id = str(row["question_id"])
    dates = row.get("haystack_dates", [])
    ids = row.get("haystack_session_ids", [])
    sessions = [(str(ids[i]) if i < len(ids) else str(i), dates[i] if i < len(dates) else None, messages) for i, messages in enumerate(row["haystack_sessions"])]
    return sample_id, sessions, [question(sample_id, sample_id, row["question"], row["answer"], question_type=row.get("question_type"), question_time=row.get("question_date"), evidence=row.get("answer_session_ids", []))]


def persona_options(row, index):
    values = [(True, row["correct_answer"])] + [(False, value) for value in parse_structured(row["incorrect_answers"])]
    if len(values) != 4:
        raise ValueError(f"PersonaMem row {index}: expected four options")
    random.Random(index).shuffle(values)
    options = {label: text for label, (_, text) in zip("abcd", values)}
    answer = next(f"({label})" for label, (correct, _) in zip("abcd", values) if correct)
    return options, answer


def persona_rows(path, files, config):
    histories = {}
    for p in files[1:]:
        histories[p.stem.rsplit("_persona", 1)[-1]] = p
    groups = OrderedDict()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            groups.setdefault(str(row["persona_id"]), []).append((index, row))
    selected = config.get("sample_ids") or []
    if isinstance(selected, str):
        selected = [selected]
    for persona_id, rows in groups.items():
        sample_id = f"persona:{persona_id}"
        if selected and sample_id not in selected:
            continue
        if persona_id not in histories:
            raise FileNotFoundError(f"Missing history for persona {persona_id}")
        with histories[persona_id].open(encoding="utf-8") as stream:
            history = json.load(stream)
        messages = history["chat_history"]
        questions = []
        for index, row in rows:
            options, answer = persona_options(row, index)
            text = row["user_query"]
            try:
                parsed = parse_structured(text)
                if isinstance(parsed, dict):
                    text = parsed.get("content", text)
            except (ValueError, SyntaxError):
                pass
            questions.append(question(sample_id, f"persona:row:{index}", text, answer, options=options, row_index=index, category=row.get("pref_type"), updated=row.get("updated"), sensitive_info=row.get("sensitive_info"), topic=row.get("topic_query")))
        yield sample_id, [(persona_id, None, messages)], questions


def convert_halumem(row, index, config):
    sample_id = str(row["uuid"])
    sessions, questions = [], []
    for si, session in enumerate(row["sessions"]):
        sessions.append((str(si), session.get("start_time"), session["dialogue"]))
        for qi, q in enumerate(session.get("questions", [])):
            questions.append(question(sample_id, f"{sample_id}:s{si}:q{qi}", q["question"], q["answer"], session_index=si, question_type=q.get("question_type"), category=q.get("question_type"), difficulty=q.get("difficulty"), evidence=q.get("evidence"), history_protocol="all_user_sessions_before_qa"))
    return sample_id, sessions, questions


def convert_beam(row, index, config):
    sample_id = str(row["conversation_id"])
    sessions = [(str(i), None, flatten_beam_session(session)) for i, session in enumerate(row["chat"])]
    questions = []
    for dimension, values in parse_structured(row["probing_questions"]).items():
        for q in values:
            gold = next((q[k] for k in ("answer", "ideal_response", "ideal_answer", "ideal_summary", "expected_compliance") if k in q and q[k] is not None), "")
            questions.append(question(sample_id, f"{sample_id}:q{len(questions)}", q["question"], gold, dimension=dimension, category=dimension, rubric=q.get("rubric", []), difficulty=q.get("difficulty"), question_time=q.get("time_anchor")))
    if config.get("smoke_diverse_questions"):
        first = next((q for q in questions if q["dimension"] != "event_ordering"), None)
        ordering = next((q for q in questions if q["dimension"] == "event_ordering"), None)
        other = next((q for q in questions if first and q["dimension"] not in {first["dimension"], "event_ordering"}), None)
        chosen = {q["question_id"] for q in (first, ordering, other) if q}
        questions = [q for q in questions if q["question_id"] in chosen]
    return sample_id, sessions, questions


def iter_eval_units(config):
    kind, files = dataset_files(config)
    if kind == "personamem_v2":
        converted = persona_rows(files[0], files, config)
    else:
        rows = read_lines(files[0]) if kind in {"halumem", "beam"} else read_array(files[0])
        converter = {"locomo": convert_locomo, "longmemeval": convert_longmemeval, "halumem": convert_halumem, "beam": convert_beam}[kind]
        converted = (converter(row, index, config) for index, row in enumerate(rows))
    selected = config.get("sample_ids") or []
    if isinstance(selected, str):
        selected = [selected]
    count = 0
    for sample_id, sessions, questions in converted:
        if selected and sample_id not in selected:
            continue
        if config.get("max_queries_per_unit"):
            questions = questions[:int(config["max_queries_per_unit"])]
        chunks = dialogue_chunks(sample_id, sessions, int(config.get("chunk_size", 4096)))
        if not chunks or not questions:
            raise ValueError(f"Empty history or questions in {kind}/{sample_id}")
        for q in questions:
            q.update(dataset="omnimemeval", benchmark=kind, variant=config.get("variant"), protocol=PROTOCOL)
        yield {"sample_id": sample_id, "context_chunks": chunks, "questions": questions, "context_chars": sum(len(c["text"]) for c in chunks), "context_chunk_count": len(chunks), "session_count": len(sessions)}
        count += 1
        if config.get("max_test_samples") and count >= int(config["max_test_samples"]):
            break
    if not count:
        raise ValueError("No OmniMemEval samples matched the configuration")
