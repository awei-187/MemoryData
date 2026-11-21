"""HaluMem loader (https://github.com/MemTensor/HaluMem), question-answering task.

One HaluMem user is one MemoryData context. Each session is split at dialogue
turn boundaries into structured chunks that never cross sessions. Every
question records ``ask_after_chunk``: the number of history chunks that must
be memorized before it is asked, which is the end of its own session. This
follows the official protocol (eval/eval_memzero.py), where a session's
questions are answered right after that session is added.

Only dialogue text enters memory. ``persona_info``, ``memory_points``, answers
and evidence stay in evaluation metadata.
"""

import json
import re
from datetime import datetime
from pathlib import Path

HALUMEM_DATASET = "halumem_qa"
VARIANT_FILES = {"medium": "HaluMem-Medium.jsonl", "long": "HaluMem-Long.jsonl"}
# Same pattern upstream uses to name the user (eval/eval_memzero.py).
NAME_PATTERN = re.compile(r"Name:\s*(.*?); Gender:")
TIME_FORMAT = "%b %d, %Y, %H:%M:%S"


def load_halumem_eval_data(dataset_config):
    """Load HaluMem users as pre-chunked contexts with session-scheduled questions."""
    dataset_path = resolve_halumem_dataset_path(dataset_config)
    chunk_size = int(dataset_config.get("chunk_size", 4096))
    if chunk_size <= 0:
        raise ValueError("HaluMem chunk_size must be positive")
    selected_ids = _normalize_user_ids(dataset_config.get("halumem_user_ids"))
    max_samples = dataset_config.get("max_test_samples")
    max_samples = int(max_samples) if max_samples else None

    records = []
    with open(dataset_path, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            user = json.loads(line)
            if selected_ids is not None and str(user.get("uuid")) not in selected_ids:
                continue
            records.append(convert_halumem_user(user, chunk_size, line_number=line_number))
            if max_samples is not None and len(records) >= max_samples:
                break
    if selected_ids is not None:
        missing = selected_ids - {record["sample_id"] for record in records}
        if missing and (max_samples is None or len(records) < max_samples):
            raise ValueError(f"HaluMem user ids not found in {dataset_path}: {sorted(missing)}")
    return {"data": records}


def resolve_halumem_dataset_path(dataset_config):
    test_file = str(dataset_config.get("test_files") or "").strip()
    if test_file:
        return Path(test_file)
    variant = str(dataset_config.get("halumem_variant", "medium")).lower()
    if variant not in VARIANT_FILES:
        raise ValueError(f"Unknown halumem_variant {variant!r}; expected one of {sorted(VARIANT_FILES)}")
    return Path(__file__).resolve().parents[2] / "datasets" / "HaluMem" / VARIANT_FILES[variant]


def parse_halumem_time(value):
    """Convert HaluMem's 'Sep 04, 2025, 18:42:18' to ISO 8601 without inventing a timezone."""
    try:
        return datetime.strptime(str(value).strip(), TIME_FORMAT).isoformat()
    except ValueError as exc:
        raise ValueError(f"Unsupported HaluMem timestamp {value!r}; expected e.g. 'Sep 04, 2025, 18:42:18'") from exc


def user_name_from_persona(persona_info):
    match = NAME_PATTERN.search(str(persona_info or ""))
    return match.group(1).strip() if match and match.group(1).strip() else "User"


def convert_halumem_user(user, chunk_size, line_number=None):
    uuid = str(user.get("uuid") or f"halumem_line_{line_number}")
    user_name = user_name_from_persona(user.get("persona_info"))
    context_chunks = []
    questions, answers, qa_pair_ids, question_types, question_dates, eval_metadata = [], [], [], [], [], []

    for session_index, session in enumerate(user.get("sessions") or []):
        context_chunks.extend(_session_chunks(uuid, user_name, session_index, session, chunk_size))
        # Upstream ingests generated QA sessions but never evaluates them.
        if session.get("is_generated_qa_session"):
            continue
        ask_after_chunk = len(context_chunks)
        question_date = parse_halumem_time(session["end_time"])
        for question_index, qa in enumerate(session.get("questions") or []):
            qa_pair_id = f"{uuid}:s{session_index}:q{question_index}"
            evidence = [
                {"memory_content": str(item.get("memory_content", "")), "memory_type": item.get("memory_type")}
                for item in qa.get("evidence") or []
            ]
            questions.append(str(qa.get("question", "")))
            answers.append(str(qa.get("answer", "")))
            qa_pair_ids.append(qa_pair_id)
            question_types.append(str(qa.get("question_type", "")))
            question_dates.append(question_date)
            eval_metadata.append({
                "dataset": HALUMEM_DATASET,
                "sample_id": uuid,
                "uuid": uuid,
                "user_name": user_name,
                "qa_pair_id": qa_pair_id,
                "question_id": qa_pair_id,
                "session_index": session_index,
                "ask_after_chunk": ask_after_chunk,
                "question": str(qa.get("question", "")),
                "question_type": str(qa.get("question_type", "")),
                "difficulty": str(qa.get("difficulty", "")),
                "evidence": evidence,
                "question_date": question_date,
            })

    return {
        "source": HALUMEM_DATASET,
        "sample_id": uuid,
        "context_chunks": context_chunks,
        "context_length": sum(len(chunk["text"]) for chunk in context_chunks),
        "questions": questions,
        "answers": answers,
        "qa_pair_ids": qa_pair_ids,
        "question_ids": list(qa_pair_ids),
        "question_types": question_types,
        "question_dates": question_dates,
        "eval_metadata": eval_metadata,
    }


def _session_chunks(uuid, user_name, session_index, session, chunk_size):
    start, end = session.get("start_time", ""), session.get("end_time", "")
    header = f"Session {session_index + 1} ({start} - {end})"
    lines, timestamps = [], []
    for turn in session.get("dialogue") or []:
        content = str(turn.get("content", "") or "").strip()
        if not content:
            continue
        speaker = user_name if turn.get("role") == "user" else "Assistant"
        lines.append(f"[{turn.get('timestamp', '')}] {speaker}: {content}")
        timestamps.append(turn.get("timestamp") or start)

    chunks, current, current_times = [], [], []
    for line, timestamp in zip(lines, timestamps):
        candidate = len(header) + sum(len(item) + 1 for item in current) + 1 + len(line)
        if current and candidate > chunk_size:
            chunks.append(_make_chunk(uuid, session_index, len(chunks), header, current, current_times[0]))
            current, current_times = [], []
        current.append(line)
        current_times.append(timestamp)
    if current:
        chunks.append(_make_chunk(uuid, session_index, len(chunks), header, current, current_times[0]))
    return chunks


def _make_chunk(uuid, session_index, chunk_index, header, lines, first_timestamp):
    return {
        "text": header + "\n" + "\n".join(lines),
        "chunk_id": f"{uuid}:s{session_index}:c{chunk_index}",
        "session_index": session_index,
        "timestamp": parse_halumem_time(first_timestamp),
    }


def _normalize_user_ids(raw_value):
    if raw_value in (None, "", []):
        return None
    values = raw_value if isinstance(raw_value, (list, tuple, set)) else str(raw_value).split(",")
    normalized = {str(value).strip() for value in values if str(value).strip()}
    return normalized or None
