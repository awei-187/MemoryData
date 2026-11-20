import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_full_first_units import expected_documents, transient_failed_queries, verify_result


class FullCoverageTests(unittest.TestCase):
    def test_resume_preserves_multi_answer_gold_and_scores(self):
        from utils.initialization import load_existing_results
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.json'
            path.write_text(json.dumps({'data': [{'query_id': 0, 'context_id': 0, 'qa_pair_id': 'id0',
                'query': 'frozen query', 'answer': ['alpha', 'beta'], 'output': 'beta',
                'input_len': 1, 'output_len': 1, 'query_time_len': 1}]}))
            metrics, rows, contexts, skipped = load_existing_results(path,
                {'dataset': 'Accurate_Retrieval', 'sub_dataset': 'longmemeval_s*', 'debug': False},
                [[('frozen query', ['alpha', 'beta'], 'id0')]], retry_failed_queries=True)
            self.assertEqual(rows[0]['answer'], ['alpha', 'beta'])
            self.assertEqual(rows[0]['f1'], 1)
            self.assertEqual(skipped, {0})
    def test_transient_recovery_excludes_successes_and_does_not_retry_policy_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            good = {'query_id': 0, 'output': 'already valid'}
            failed = {'query_id': 1, 'output': '', 'status': 'failed', 'error': 'Request timed out.'}
            path.write_text(json.dumps({'data': [good, failed]}))
            self.assertEqual(transient_failed_queries(path), [1])
            path.write_text(json.dumps({'data': [good, {**failed, 'error': '400 content_filter'}]}))
            self.assertEqual(transient_failed_queries(path), [])

    def test_prefix_is_not_accepted_as_full_and_failed_rows_are_not_zero_scores(self):
        qa = [['question one', ['answer'], 'id0'], ['question two', ['answer'], 'id1']]
        rows = [{'query_id': i, 'context_id': 0, 'qa_pair_id': item[2], 'query': item[0], 'output': 'answer'}
                for i, item in enumerate(qa)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.json'
            def write(data):
                path.write_text(json.dumps({'data': data, 'averaged_metrics': {'exact_match': 100}}))
            write(rows[:1])
            with self.assertRaisesRegex(ValueError, 'coverage'):
                verify_result(path, qa)
            write([{**rows[0], 'status': 'failed'}, rows[1]])
            with self.assertRaisesRegex(ValueError, 'Failed/empty'):
                verify_result(path, qa)
            write(rows)
            self.assertEqual(verify_result(path, qa)['completed'], 2)
            old_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            write([{**rows[0], 'output': 'changed'}, rows[1]])
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                verify_result(path, qa, old_hash)

    def test_question_identity_and_text_must_both_match(self):
        qa = [['frozen query', ['answer'], 'id0']]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'results.json'
            path.write_text(json.dumps({'data': [{'query_id': 0, 'context_id': 0, 'qa_pair_id': 'id0',
                                                'query': 'different query', 'output': 'answer'}],
                                        'averaged_metrics': {}}))
            with self.assertRaisesRegex(ValueError, 'identity/content'):
                verify_result(path, qa)

    def test_acknowledged_document_ids_use_native_normalized_text(self):
        from methods.hindsight.hindsight_adapter import fingerprint
        self.assertEqual(expected_documents(['  Alpha\r\n\r\n\r\nBeta  ', 'Second']),
                         {f'md-chunk-00000000-{fingerprint("Alpha" + chr(10) * 2 + "Beta")[:16]}': [],
                          f'md-chunk-00000001-{fingerprint("Second")[:16]}': []})
