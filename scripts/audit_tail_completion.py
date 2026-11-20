"""Verify the final case and prove previously successful outputs were preserved."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/full_first_units_20261023'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    current = read(RUN / 'suite.status.json')
    assert current['phase'] == 'success' and len(current['cases']) == 22
    by_id = {case['id']: case for case in current['cases']}
    baseline = read(RUN / 'suite.attempt_011.status.json')
    preserved = []
    for old in baseline['cases']:
        if old['phase'] == 'success':
            now = by_id[old['id']]
            assert old['result_sha256'] == now['result_sha256'] == digest(Path(now['result_file']))
            preserved.append(old['id'])
    assert len(preserved) == 21
    case = by_id['mem0_eventqa']
    assert case['phase'] == 'success' and case['completed'] == 100
    rows = read(Path(case['result_file']))['data']
    assert len(rows) == 100 and {row['query_id'] for row in rows} == set(range(100))
    assert all(row.get('status') != 'failed' and str(row.get('output') or '').strip() for row in rows)
    previous_files = list((RUN / 'attempt_history/attempt_13/artifacts/eventqa/mem0/outputs').rglob('*_results.json'))
    assert len(previous_files) == 1
    previous = {row['query_id']: row for row in read(previous_files[0])['data']}
    preserved_queries = [row['query_id'] for row in rows if previous[row['query_id']].get('status') != 'failed']
    assert len(preserved_queries) == 99
    assert all(row['output'] == previous[row['query_id']]['output'] for row in rows if row['query_id'] in preserved_queries)
    validation = read(RUN / 'mem0_checkpoint_production_validation.json')
    assert validation['complete'] and validation['committed_chunks'] == validation['total_chunks'] == 131
    result = {'all_cases_success': True, 'case_count': 22, 'expected_answers': 1130,
              'completed_answers': sum(case['completed'] for case in current['cases']),
              'preserved_prior_success_cases': preserved, 'preserved_prior_success_case_count': 21,
              'preserved_valid_eventqa_answers_before_final_retry': 99,
              'completed_eventqa_answers': 100, 'eventqa_result_sha256': digest(Path(case['result_file'])),
              'eventqa_metrics': case['averaged_metrics'], 'eventqa_memory_complete': True,
              'source_comparison_result_sha256': digest(previous_files[0])}
    assert result['completed_answers'] == 1130
    (RUN / 'tail_completion_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
