"""Task requirements shared by all MemoryData method adapters."""
SYSTEM = "You are a helpful assistant. Answer using the supplied dialogue history."


def get_template(sub_dataset, name):
    kind = sub_dataset.removeprefix("omni_")
    instruction = (
        "Choose the most personalized response. Reply with exactly one option: (a), (b), (c), or (d)."
        if kind == "personamem_v2" else
        "Answer completely using the history, including all supported facts needed by the question. Preserve event order and follow remembered instructions."
        if kind == "beam" else
        "Answer concisely using the history. Resolve dates using the history timestamps. Use the latest information unless asked about an earlier state. Say unknown when the history does not contain an answer."
    )
    return {
        "system": SYSTEM,
        "memorize": "{context}",
        "query": instruction + "\n\n{question}",
        "fact_extraction": "Extract factual memories from the conversation, preserving speakers, event dates, preferences, instructions, negations, and changes over time. Do not invent facts.",
        "memory_answer": instruction + "\n\nRetrieved history:\n{memories}\n\n{question}",
    }[name]


def format_question(q):
    text = q["question"]
    if q.get("options"):
        text += "\n" + "\n".join(f"({label}) {value}" for label, value in q["options"].items())
    if q.get("question_time"):
        text = f"Question time: {q['question_time']}\n" + text
    return text
