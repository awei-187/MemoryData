"""Pure scoring helpers; derived from OmniMemEval a5c8b4b (Apache-2.0).

See UPSTREAM_LICENSE and UPSTREAM_NOTICE. Parsing failures raise, rather than
silently producing a wrong-answer score. Formulae retain the upstream semantics.
"""
import json
import math
import re

PARSER_VERSION = "omni_strict_v1"


def json_object(text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("No valid JSON object in judge response")


def binary_label(text):
    text = str(text or "").strip()
    tokens = set(re.findall(r"\b(?:CORRECT|WRONG)\b", text.upper()))
    if len(tokens) > 1:
        raise ValueError("Conflicting judge labels")
    try:
        label = json_object(text).get("label")
    except ValueError:
        label = None
    if label is not None:
        label = str(label).strip().upper()
        if label not in {"CORRECT", "WRONG"}:
            raise ValueError("Invalid JSON label")
    else:
        match = re.search(r"(?:^|\s)(CORRECT|WRONG)[.!]?\s*$", text, re.IGNORECASE)
        if not match:
            raise ValueError("Missing unambiguous final judge label")
        label = match[1].upper()
    return label == "CORRECT"


def extract_choice_answer(predicted_answer, correct_answer):
    # Same extraction and correctness semantics as pm_responses.py upstream.
    def options(text):
        text = text.lower()
        found = re.findall(r"\(([a-d])\)", text)
        return set(found) if found else set(re.findall(r"\b([a-d])\b", text))
    full_response = str(predicted_answer)
    prediction = full_response.strip()
    if "<final_answer>" in prediction:
        prediction = prediction.split("<final_answer>")[-1].strip()
    if prediction.endswith("</final_answer>"):
        prediction = prediction[:-len("</final_answer>")].strip()
    correct = correct_answer.lower().strip("() ")
    return options(prediction) == {correct} or options(full_response) == {correct}, prediction


def nugget_score(text):
    try:
        value = float(json_object(text)["score"])
    except (ValueError, KeyError, TypeError):
        match = re.search(r'"?score"?\s*[:=]\s*([0-9.]+)', text)
        if not match:
            raise ValueError("No nugget score")
        value = float(match[1])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Nugget score outside [0, 1]")
    return 1.0 if value >= .75 else .5 if value >= .25 else 0.0


def kendall_tau_b(ref_order, pred_order):
    n = len(ref_order)
    if n < 2:
        return 1.0
    concordant = discordant = ties_ref = ties_pred = 0
    for i in range(n):
        for j in range(i + 1, n):
            rd, pd = ref_order[i] - ref_order[j], pred_order[i] - pred_order[j]
            if rd == 0 and pd == 0:
                ties_ref += 1
                ties_pred += 1
            elif rd == 0:
                ties_ref += 1
            elif pd == 0:
                ties_pred += 1
            elif (rd > 0) == (pd > 0):
                concordant += 1
            else:
                discordant += 1
    pairs = n * (n - 1) // 2
    denominator = ((pairs - ties_ref) * (pairs - ties_pred)) ** .5
    return (concordant - discordant) / denominator if denominator else 0.0


def ordering_score(text, count):
    if count < 1:
        raise ValueError("Empty event ordering rubric")
    positions = json_object(text)["positions"]
    if not isinstance(positions, list):
        raise ValueError("positions must be a list")
    detected = [entry.get("position", -1) for entry in positions[:count]]
    if any(not isinstance(pos, (int, float)) or not math.isfinite(pos) for pos in detected):
        raise ValueError("Invalid event position")
    detected.extend([-1] * (count - len(detected)))
    found = [(i, pos) for i, pos in enumerate(detected) if pos > 0]
    coverage = len(found) / count
    if len(found) < 2:
        score = coverage * .5
    else:
        tau = kendall_tau_b([i for i, _ in found], [pos for _, pos in found])
        score = max(0.0, (tau + 1) / 2) * coverage
    return score, detected
