"""Regression checks for literal MemoryAgentBench source IDs on Windows."""
import copy
import pathlib
import tempfile
import unittest
from unittest.mock import patch

from utils.artifact_paths import generate_agent_save_folder_path, sanitize_artifact_component


class WindowsArtifactPathTests(unittest.TestCase):
    def test_mem0_initialization_creates_real_qdrant_collection_for_literal_star(self):
        from utils.agent import AgentWrapper
        from qdrant_client import QdrantClient, models
        import mem0.memory.main as memory_module
        wrapper = AgentWrapper.__new__(AgentWrapper)
        wrapper.sub_dataset = 'longmemeval_s*'
        wrapper.agent_name = 'Structure_rag_mem0'
        wrapper.model = 'offline-model'
        wrapper.temperature = 0
        captured = {}

        def local_storage(config):
            captured.update(config['vector_store']['config'])
            client = QdrantClient(path=captured['path'])
            client.create_collection(captured['collection_name'],
                                     vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE))
            return client

        with tempfile.TemporaryDirectory() as directory:
            wrapper.agent_save_to_folder = directory
            config = {'retrieve_num': 1, 'embedding_dim': 3, 'mem0_fact_extraction_prompt': 'Extract facts'}
            with patch.object(wrapper, '_create_oai_client', return_value=None), \
                 patch.object(wrapper, '_resolve_base_url', return_value=''), \
                 patch.object(wrapper, '_resolve_llm_api_key', return_value=''), \
                 patch.object(memory_module.Memory, 'from_config', side_effect=local_storage), \
                 patch.dict('os.environ', {'MEM0_DIR': directory}):
                wrapper._initialize_mem0_agent(config, {'chunk_size': 4096})
            try:
                self.assertEqual(captured['collection_name'], 'mem0_longmemeval_s%2A')
                self.assertEqual(wrapper.sub_dataset, 'longmemeval_s*')
                self.assertEqual(wrapper.memory.get_collection(captured['collection_name']).points_count, 0)
            finally:
                wrapper.memory.close()

    def test_literal_source_star_is_encoded_for_both_memory_methods(self):
        dataset = {'dataset': 'Accurate_Retrieval', 'sub_dataset': 'longmemeval_s*', 'chunk_size': 4096}
        before = copy.deepcopy(dataset)
        with tempfile.TemporaryDirectory() as directory:
            for method in ('Structure_rag_mem0', 'Agentic_memory_hindsight'):
                agent = {'agent_name': method, 'model': 'gpt-4o-mini', 'agent_chunk_size': 4096}
                path = pathlib.Path(generate_agent_save_folder_path(agent, dataset, 0, directory))
                self.assertNotIn('*', str(path))
                self.assertIn('longmemeval_s%2A', str(path))
                path.mkdir(parents=True)
                self.assertTrue(path.is_dir())
        self.assertEqual(dataset, before)

    def test_result_filename_is_writable_and_preserves_filter(self):
        from utils.initialization import _create_output_path
        dataset = {'dataset': 'Accurate_Retrieval', 'sub_dataset': 'longmemeval_s*', 'chunk_size': 4096,
                   'context_max_length': 150000, 'generation_max_length': 50, 'max_test_samples': 1, 'shots': 0}
        with tempfile.TemporaryDirectory() as directory:
            agent = {'agent_name': 'Structure_rag_mem0', 'model': 'gpt-4o-mini', 'output_dir': directory,
                     'agent_chunk_size': 4096, 'retrieve_num': 100}
            path = pathlib.Path(_create_output_path(agent, dataset))
            path.write_text('{}', encoding='utf-8')
            self.assertIn('longmemeval_s%2A', path.name)
        self.assertEqual(dataset['sub_dataset'], 'longmemeval_s*')

    def test_existing_safe_names_do_not_change(self):
        name = 'Structure_rag_mem0_locomo_qa_chunk4096_modelgpt-4o-mini'
        self.assertEqual(sanitize_artifact_component(name), name)

    def test_literal_encoding_and_illegal_character_do_not_collide(self):
        self.assertNotEqual(sanitize_artifact_component('source*'), sanitize_artifact_component('source%2A'))
        self.assertEqual(sanitize_artifact_component('source*'), 'source%2A')


if __name__ == '__main__':
    unittest.main()
