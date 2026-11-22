"""AgentWrapper glue for memory methods that run behind a worker (REMem, TSM).

Kept out of utils/agent.py so the wrapper only gains small dispatch branches.
The flow matches the Hindsight handler: memorizing sends the cleaned chunk with
its LoCoMo source IDs and timestamp; querying retrieves through the native
method and, by default, lets MemoryData's answer model answer from the
retrieved memories with the shared templates. ``*_answer_mode: native`` uses the
method's own answer instead (REMem's agent answer, TSM's release QA prompt).
"""

from __future__ import annotations

import time

from utils.locomo_utils import parse_locomo_source_ids

WORKER_METHODS = ("remem", "tsm")


def worker_method(wrapper):
    return next((name for name in WORKER_METHODS if wrapper._is_agent_type(name)), None)


def initialize(wrapper, agent_config, dataset_config, method):
    from methods.worker_bridge import close_all_workers

    if dataset_config.get("interleave_queries"):
        # Checked before any worker starts or any chunk is buffered, so such a
        # run stops at construction instead of at its first question.
        raise ValueError(
            f"{method} builds its index once from the complete history (save()); questions "
            "interleaved with history chunks (interleave_queries, e.g. HaluMem's session "
            "protocol) are not supported")

    # Contexts run one after another: stop the previous context's worker first.
    close_all_workers()
    if method == "remem":
        from methods.remem import RememAdapter as Adapter
    else:
        from methods.tsm import TsmAdapter as Adapter
    wrapper.retrieve_num = int(agent_config.get("retrieve_num", 20))
    wrapper.chunk_size = agent_config.get("agent_chunk_size", dataset_config.get("chunk_size"))
    wrapper.context = ""
    wrapper.agent_start_time = time.time()
    wrapper.worker_memory = Adapter(agent_config, dataset_config, wrapper.agent_save_to_folder)
    wrapper.client = wrapper._create_oai_client()


def remember_timestamp(wrapper, message):
    timestamp = message.get("timestamp") if isinstance(message, dict) else None
    wrapper._worker_chunk_timestamp = timestamp.isoformat() if hasattr(timestamp, "isoformat") else timestamp


def handle(wrapper, message, memorizing, query_id, context_id):
    adapter = wrapper.worker_memory
    if memorizing:
        adapter.add_chunk(wrapper._prepare_memory_chunk_for_storage(message),
                          source_ids=parse_locomo_source_ids(message),
                          timestamp=getattr(wrapper, "_worker_chunk_timestamp", None))
        return "Memorized"

    memory_time = time.time() - wrapper.agent_start_time
    metadata = wrapper._current_eval_metadata or {}
    native = adapter.answer_mode == "native"
    native_answer, entries = adapter.retrieve(message, query_timestamp=metadata.get("question_date"),
                                              native_answer=native)
    raw_contexts = [entry["text"] for entry in entries]
    groups = [entry["source_ids"] for entry in entries]
    method = worker_method(wrapper)
    if native:
        memories = "\n".join(raw_contexts)
        response = native_answer
        # The native answer call runs inside the worker; its token usage is not
        # reported, so input length counts the question and retrieved memories.
        input_tokens = wrapper._count_tokens(message + "\n" + memories)
        output_tokens = wrapper._count_tokens(response)
    else:
        prompt_override = wrapper.agent_config.get(f"{method}_memory_answer_prompt")
        contexts = wrapper._fit_memories_for_answer(message, raw_contexts, prompt_override=prompt_override)
        memories = "\n".join(contexts) or "(No retrieved memories found.)"
        response, input_tokens, output_tokens = wrapper._generate_answer_from_memories(
            message, memories, prompt_override=prompt_override)
        input_tokens += wrapper._count_tokens(memories)
    output = wrapper._create_standard_response(
        response, input_tokens, output_tokens, memory_time, time.time() - wrapper.agent_start_time - memory_time)
    wrapper._maybe_save_longmemeval_backfill_debug(
        query_id=query_id, context_id=context_id, retrieved_paragraphs=raw_contexts,
        query_text=message, response_text=response, retrieved_source_id_groups=groups,
    )
    wrapper.agent_start_time = time.time()
    return wrapper._attach_locomo_recall_metadata(output, groups)
