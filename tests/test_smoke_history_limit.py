import unittest
from unittest.mock import patch

from utils.conversation_creator import ConversationCreator


class SmokeHistoryLimitTests(unittest.TestCase):
    def creator(self, limit=None):
        config = {'dataset': 'MemBench', 'sub_dataset': 'test', 'max_test_samples': 1,
                  'context_max_length': 200000, 'chunk_size': 4096}
        if limit is not None:
            config['smoke_max_history_chunks'] = limit
        with patch.object(ConversationCreator, '_load_and_process_dataset'):
            return ConversationCreator({'agent_name': 'Structure_rag_mem0'}, config)

    def test_limit_preserves_provenance_and_original_context(self):
        creator = self.creator(2)
        original = [{'text': 'x' * 2500, 'source_ids': [f'D1:{i}']} for i in range(3)]
        creator.contexts = [original]
        result = creator.get_chunks()
        self.assertEqual(result, [original[:2]])
        self.assertEqual(len(original), 3)
        self.assertEqual(result[0][1]['source_ids'], ['D1:1'])

    def test_no_limit_retains_all_history(self):
        creator = self.creator()
        original = [{'text': 'x' * 2500} for _ in range(3)]
        creator.contexts = [original]
        self.assertEqual(creator.get_chunks(), [original])

    def test_invalid_limits_fail_before_loading(self):
        for value in (0, -1, True, '2'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'positive integer'):
                self.creator(value)
