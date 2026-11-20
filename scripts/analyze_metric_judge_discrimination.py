"""Offline comparison of saved judges and actual MemoryData rule metrics; no API calls."""
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.eval_other_utils import calculate_metrics, post_process

OUT = ROOT / 'results/metric_judge_analysis_20261024'
SOURCES = {
    'mem0': Path('E:/LLM judge/mem0_locomo_dual_eval/self_review_147/results_with_original_judge.jsonl'),
    'hindsight': Path('E:/LLM judge/hindsight_locomo_dual_eval/self_review_147/results_with_original_judge.jsonl')}
FIELDS = ('exact_match', 'substring_exact_match', 'f1', 'rougeL_f1', 'rougeL_recall')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def metrics(answer, gold):
    raw = calculate_metrics(answer, gold)
    processed, info = post_process({'output': answer}, gold,
        {'dataset': 'LoCoMo', 'sub_dataset': 'locomo_qa'})
    return {'raw': {key: float(raw[key]) for key in FIELDS},
            'framework': {key: float(processed[key]) for key in FIELDS},
            'parsed_output': info.get('parsed_output')}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, summaries, by_system = [], {}, {}
    for system, path in SOURCES.items():
        original = load_lines(path)
        assert len(original) == 147
        converted = []
        for item in original:
            state = item['state']
            converted.append({'system': system, 'question_id': item['question_id'], **state,
                'old_label': item['original_judge']['label'],
                'old_reason': item['original_judge'].get('reasoning'),
                'gpt_label': item['gpt_label'], 'gpt_reason': item.get('gpt_reason'),
                'jev_label': item['jev_label'], 'review_label': item['self_label'],
                'review_reason': item.get('self_reason'), **metrics(state['candidate_answer'], state['gold_answer'])})
        by_system[system] = {item['question_id']: item for item in converted}
        summary = {'answers': len(converted), 'old_labels': dict(Counter(item['old_label'] for item in converted))}
        for field in ('gpt_label', 'jev_label', 'review_label'):
            summary[field] = dict(Counter(item[field] for item in converted))
            summary['old_CORRECT_to_' + field] = dict(Counter(item[field] for item in converted if item['old_label'] == 'CORRECT'))
        summary['EM_zero_but_new_GPT_correct'] = sum(item['framework']['exact_match'] == 0 and item['gpt_label'] == 'correct' for item in converted)
        summary['new_GPT_vs_Jev_agreement'] = sum(item['gpt_label'] == item['jev_label'] for item in converted)
        summary['rule_metric_means_percent'] = {key: sum(item['framework'][key] for item in converted) / 147 * 100 for key in FIELDS}
        summaries[system] = summary
        rows.extend(converted)
    assert by_system['mem0'].keys() == by_system['hindsight'].keys()
    pairs = []
    for qid in by_system['mem0']:
        a, b = by_system['mem0'][qid], by_system['hindsight'][qid]
        assert a['query'] == b['query'] and a['gold_answer'] == b['gold_answer'], qid
        same = {key: abs(a['framework'][key] - b['framework'][key]) < 1e-12 for key in FIELDS}
        differs = a['gpt_label'] != b['gpt_label']
        inverted = False
        if (a['gpt_label'] == 'correct') != (b['gpt_label'] == 'correct'):
            correct, other = (a, b) if a['gpt_label'] == 'correct' else (b, a)
            inverted = correct['framework']['f1'] < other['framework']['f1'] - 1e-12
        pairs.append({'question_id': qid, 'new_gpt_labels_differ': differs,
                      'same_metrics': same, 'all_five_metrics_equal': all(same.values()),
                      'F1_inverted_relative_to_new_GPT_full_correct': inverted,
                      'mem0_gpt': a['gpt_label'], 'hindsight_gpt': b['gpt_label'],
                      'all_three_new_judges_agree_within_each_system':
                          len({a['gpt_label'], a['jev_label'], a['review_label']}) == 1 and
                          len({b['gpt_label'], b['jev_label'], b['review_label']}) == 1})
    different = [p for p in pairs if p['new_gpt_labels_differ']]
    paired = {'same_question_pairs': 147, 'different_new_GPT_labels': len(different),
        'metric_ties_among_different_labels': {key: sum(p['same_metrics'][key] for p in different) for key in FIELDS},
        'all_five_metrics_equal_with_different_GPT_labels': [p['question_id'] for p in different if p['all_five_metrics_equal']],
        'all_five_equal_and_three_new_judges_agree_per_system': [p['question_id'] for p in different if p['all_five_metrics_equal'] and p['all_three_new_judges_agree_within_each_system']],
        'F1_inversions_relative_to_new_GPT_labels': [p['question_id'] for p in pairs if p['F1_inverted_relative_to_new_GPT_full_correct']]}
    synthetic = []
    for name, query, gold, answer, expected in [
        ('negation_correct', 'Which city hosted the meeting?', 'Paris', 'Paris, not Lyon.', 'correct'),
        ('negation_wrong', 'Which city hosted the meeting?', 'Paris', 'Lyon, not Paris.', 'incorrect'),
        ('date_equivalent', 'On what date?', 'May 7, 2023', '7 May 2023', 'correct'),
        ('date_wrong_year', 'On what date?', 'May 7, 2023', 'May 7, 2024', 'incorrect'),
        ('AND_list_partial', 'List both instruments.', ['clarinet', 'violin'], 'clarinet', 'partial'),
        ('AND_list_full', 'List both instruments.', ['clarinet', 'violin'], 'clarinet and violin', 'correct')]:
        item = {'name': name, 'query': query, 'gold': gold, 'candidate': answer,
                'expected_by_stated_semantics_NOT_model_output': expected, **metrics(answer, gold)}
        if name.startswith('AND_list'):
            item['eventqa_all_elements_recall'] = int(all(value.lower() in answer.lower() for value in gold))
        synthetic.append(item)
    # Choice A is stripped as an English article by the shared normalizer.
    for answer in ('A', 'D', 'I cannot determine the option.'):
        choice_metrics, choice_info = post_process({'output': answer}, ['A'],
            {'dataset': 'MemBench', 'sub_dataset': 'membench_simple'})
        synthetic.append({'name': 'MCQ_A_normalization', 'gold': ['A'],
            'candidate': answer, 'framework': choice_metrics,
            'parsed_output': choice_info['parsed_output'],
            'note': 'Diagnostic control, not a new model answer or judge call'})
    full = json.loads((ROOT / 'results/full_first_units_20261023/summary.json').read_text(encoding='utf-8'))
    unique = {}
    for row in load_lines(ROOT / 'results/full_first_units_20261023/per_question.jsonl'):
        unique[(row['task'], row['qa_pair_id'])] = row['frozen_gold']
    gold_shapes = Counter('list_len_' + str(len(g)) if isinstance(g, list) else type(g).__name__ for g in unique.values())
    summary = {'generated_at': datetime.now().astimezone().isoformat(), 'external_api_calls': 0,
        'main_run_answers': full['successful_answers'], 'main_run_llm_judge_calls': 0,
        'main_unique_questions': len(unique), 'main_gold_shapes': dict(gold_shapes),
        'historical_dataset_scope': 'edited LoCoMo first 147 queries; fixed saved answers; separate from raw 199-Q main evaluation',
        'sources': {system: {'path': str(path), 'sha256': digest(path)} for system, path in SOURCES.items()},
        'main_sources': {str(path.relative_to(ROOT)): digest(path) for path in (
            ROOT / 'results/full_first_units_20261023/summary.json',
            ROOT / 'results/full_first_units_20261023/per_question.jsonl')},
        'source_metric_code_sha256': digest(ROOT / 'utils/eval_other_utils.py'),
        'historical_systems': summaries, 'paired': paired,
        'note': 'Judge disagreements are not independent human ground truth; synthetic expected labels are not actual Judge outputs'}
    for name, value in (('analysis_summary.json', summary), ('paired_comparisons.json', pairs), ('synthetic_controls.json', synthetic)):
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'historical_294_metrics.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    interesting = {'conv0_q7', 'conv0_q11', 'conv0_q38', 'conv0_q57', 'conv0_q126', *paired['all_five_metrics_equal_with_different_GPT_labels'], *paired['F1_inversions_relative_to_new_GPT_labels']}
    examples = [row for row in rows if row['question_id'] in interesting]
    (OUT / 'selected_real_examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
