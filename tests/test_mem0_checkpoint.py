import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import httpx
from openai import BadRequestError

import utils.initialization as initialization  # Registers the vendored Mem0 source.
from mem0.memory.storage import SQLiteManager
from mem0.vector_stores.qdrant import Qdrant
from utils.mem0_checkpoint import Mem0Checkpoint, digest, fingerprint
from utils.model_api_diagnostics import error_details


class LocalAgent:
    calls = []
    fail_on = None

    def __init__(self, config, dataset, load_agent_from):
        folder = Path(load_agent_from)
        runtime = folder / 'mem0_runtime'
        runtime.mkdir(parents=True, exist_ok=True)
        self.folder = folder
        self.mem0_source_map = {}
        self.memory = SimpleNamespace(enable_graph=False,
            config=SimpleNamespace(history_db_path=str(runtime / 'history.db'),
                vector_store=SimpleNamespace(config=SimpleNamespace(path=str(runtime / 'qdrant')))),
            vector_store=Qdrant('checkpoint_test', 3, path=str(runtime / 'qdrant'), on_disk=True),
            db=SQLiteManager(str(runtime / 'history.db')))

    def send_message(self, chunk, **kwargs):
        self.calls.append(chunk)
        memory_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk))
        self.memory.vector_store.insert([[1., 0., 0.]], [{'data': chunk}], [memory_id])
        self.memory.db.add_history(memory_id, None, chunk, 'ADD')
        self.mem0_source_map[memory_id] = [chunk]
        if chunk == self.fail_on:
            raise RuntimeError('interrupted AFTER partial database writes')

    def save_agent(self):
        (self.folder / 'mem0_source_map.json').write_text(json.dumps(self.mem0_source_map))

    def load_agent(self):
        self.mem0_source_map = json.loads((self.folder / 'mem0_source_map.json').read_text())

    def count(self):
        return self.memory.vector_store.client.get_collection('checkpoint_test').points_count

    def close(self):
        self.memory.vector_store.client.close()
        self.memory.db.connection.close()


class Mem0CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.config = {'agent_name': 'Structure_rag_mem0', 'mem0_checkpoint_enabled': True}
        self.dataset = {'dataset': 'Accurate_Retrieval', 'sub_dataset': 'checkpoint_test'}
        LocalAgent.calls = []
        LocalAgent.fail_on = None

    def test_framework_rolls_back_partial_writes_then_resumes_without_duplicate_history(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(initialization, 'AgentWrapper', LocalAgent):
            folder = Path(directory)
            LocalAgent.fail_on = 'second'
            with self.assertRaisesRegex(RuntimeError, 'partial database writes'):
                initialization.initialize_and_memorize_agent(self.config, self.dataset, directory,
                                                             ['first', 'second', 'third'], 0, 1)
            checkpoint = Mem0Checkpoint(directory, ['first', 'second', 'third'], self.config, self.dataset, 0)
            self.assertEqual(checkpoint.validate()['next_chunk_index'], 1)
            self.assertFalse((folder / 'mem0_source_map.json').exists())
            LocalAgent.fail_on = None
            agent, _ = initialization.initialize_and_memorize_agent(self.config, self.dataset, directory,
                                                                   ['first', 'second', 'third'], 0, 1)
            try:
                self.assertEqual(agent.count(), 3)
                self.assertEqual(len(agent.mem0_source_map), 3)
                for chunk in ('first', 'second', 'third'):
                    memory_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk))
                    self.assertEqual(len(agent.memory.db.get_history(memory_id)), 1)
                self.assertEqual(LocalAgent.calls, ['first', 'second', 'second', 'third'])
                checkpoint.validate()
                self.assertTrue(checkpoint.complete)
                self.assertEqual(len(checkpoint.write_times), 3)
            finally:
                agent.close()
            # A finished checkpoint reloads the whole state and performs no writes.
            agent, _ = initialization.initialize_and_memorize_agent(self.config, self.dataset, directory,
                                                                   ['first', 'second', 'third'], 0, 1)
            try:
                self.assertEqual(agent.count(), 3)
                self.assertEqual(len(LocalAgent.calls), 4)
            finally:
                agent.close()

    def test_input_config_and_snapshot_corruption_are_rejected_before_live_state_is_moved(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = LocalAgent({}, {}, directory)
            checkpoint = Mem0Checkpoint(directory, ['first'], self.config, self.dataset, 0)
            checkpoint.capture(agent, 0, [])
            agent.close()
            live = Path(directory) / 'mem0_runtime/history.db'
            before = digest(live)
            for chunks, cfg in ((['changed'], self.config), (['first'], dict(self.config, model='changed'))):
                with self.assertRaisesRegex(ValueError, 'identity'):
                    Mem0Checkpoint(directory, chunks, cfg, self.dataset, 0).restore_before_open()
                self.assertEqual(digest(live), before)
            (checkpoint.generation / 'source_map.json').write_text('{"tampered": []}')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                checkpoint.restore_before_open()
            self.assertEqual(digest(live), before)

    def _checkpoint_in(self, repo, chunks=('first',)):
        folder = repo / 'results' / 'agents' / 'exp_0'
        # output_dir mixes separators, as main.py's os.path.join rebasing produces.
        config = dict(self.config, artifact_root=str(repo / 'results'),
                      output_dir=os.path.join(str(repo / 'results'), 'outputs/api'))
        dataset = dict(self.dataset, test_files=str(repo / 'data' / 'input.json'))
        return folder, Mem0Checkpoint(folder, list(chunks), config, dataset, 0)

    def _committed(self, repo):
        with patch('utils.repo_paths.REPO_ROOT', repo.resolve()):
            folder, checkpoint = self._checkpoint_in(repo)
            agent = LocalAgent({}, {}, folder)
            checkpoint.capture(agent, 0, [])
            agent.close()
        return folder

    def test_moving_the_whole_repository_keeps_checkpoint_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory) / 'old checkout', Path(directory) / 'new'
            self._committed(old)
            shutil.move(str(old), str(new))
            with patch('utils.repo_paths.REPO_ROOT', new.resolve()):
                self.assertEqual(self._checkpoint_in(new)[1].validate()['next_chunk_index'], 0)
                with self.assertRaisesRegex(ValueError, 'identity'):
                    self._checkpoint_in(new, ['changed'])[1].validate()

    def test_absolute_path_legacy_checkpoint_needs_declared_former_root_and_is_not_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory) / 'old checkout', Path(directory) / 'new'
            folder = self._committed(old)
            # Identity exactly as written before repository-relative fingerprints.
            repo = Path(__file__).resolve().parents[1]
            legacy = fingerprint({'checkpoint_protocol': 1, 'chunks': ['first'],
                                  'agent': dict(self.config, artifact_root=str(old / 'results'),
                                                output_dir=os.path.join(str(old / 'results'), 'outputs/api')),
                                  'dataset': dict(self.dataset, test_files=str(old / 'data' / 'input.json')),
                                  'context_id': 0, 'source_hashes': {name: digest(repo / name) for name in (
                                      'methods/mem0/source/mem0/memory/main.py',
                                      'benchmark/memoryagentbench/prompts/benchmark_templates.py')}})
            pointer = json.loads((folder / 'mem0_ingestion_checkpoint.json').read_text())
            manifest_path = folder / 'mem0_checkpoints' / pointer['generation'] / 'manifest.json'
            manifest_path.write_text(json.dumps(dict(json.loads(manifest_path.read_text()), identity=legacy)))
            pointer['manifest_sha256'] = digest(manifest_path)
            (folder / 'mem0_ingestion_checkpoint.json').write_text(json.dumps(pointer))
            with patch('utils.repo_paths.REPO_ROOT', old.resolve()):
                self._checkpoint_in(old)[1].validate()  # Old checkpoints stay valid where they are.
            shutil.move(str(old), str(new))
            moved = new / 'results' / 'agents' / 'exp_0'
            before = {p: digest(p) for p in moved.rglob('*') if p.is_file()}
            with patch('utils.repo_paths.REPO_ROOT', new.resolve()):
                with patch.dict(os.environ, {'MEMORYDATA_LEGACY_REPO_ROOTS': ''}), \
                        self.assertRaisesRegex(ValueError, 'identity'):
                    self._checkpoint_in(new)[1].validate()
                with patch.dict(os.environ, {'MEMORYDATA_LEGACY_REPO_ROOTS': str(old)}):
                    self._checkpoint_in(new)[1].validate()
                    with self.assertRaisesRegex(ValueError, 'identity'):
                        self._checkpoint_in(new, ['changed'])[1].validate()
            self.assertEqual(before, {p: digest(p) for p in moved.rglob('*') if p.is_file()})

    def test_interrupted_pointer_publish_keeps_previous_generation_authoritative(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = LocalAgent({}, {}, directory)
            checkpoint = Mem0Checkpoint(directory, ['first'], self.config, self.dataset, 0)
            checkpoint.capture(agent, 0, [])
            agent.send_message('first')
            import os
            original = os.replace
            def interrupt(source, destination):
                if Path(destination) == checkpoint.pointer:
                    raise OSError('simulated crash before atomic publish')
                return original(source, destination)
            try:
                with patch('utils.mem0_checkpoint.os.replace', side_effect=interrupt):
                    with self.assertRaisesRegex(OSError, 'atomic publish'):
                        checkpoint.capture(agent, 1, [.1])
            finally:
                agent.close()
            checkpoint.restore_before_open()
            self.assertEqual(checkpoint.next_chunk_index, 0)
            agent = LocalAgent({}, {}, directory)
            try:
                self.assertEqual(agent.count(), 0)
                self.assertEqual(agent.memory.db.connection.execute('SELECT count(*) FROM history').fetchone()[0], 0)
            finally:
                agent.close()

    def test_diagnostic_headers_exclude_credentials_and_do_not_guess_provider(self):
        response = httpx.Response(400, request=httpx.Request('POST', 'https://offline.invalid/v1'),
            headers={'x-oneapi-request-id': 'diagnostic-id', 'authorization': 'secret', 'set-cookie': 'secret'})
        error = BadRequestError('policy rejected', response=response, body={'code': 'content_filter',
            'message': 'policy rejected (request id: gateway-123)',
            'secret': 'never persist', 'innererror': {'code': 'ResponsibleAIPolicyViolation',
            'content_filter_result': {'violence': {'filtered': True, 'severity': 'high', 'secret': 'never persist'}}}})
        result = error_details(error)
        self.assertEqual(result['response_headers'], {'x-oneapi-request-id': 'diagnostic-id'})
        self.assertEqual(result['filter_results']['violence'], {'filtered': True, 'severity': 'high'})
        self.assertNotIn('secret', json.dumps(result))
        self.assertNotIn('provider', result)
        self.assertEqual(result['message_request_id'], 'gateway-123')

    def test_environment_opt_in_works_and_implicit_reopen_is_refused(self):
        config = {'agent_name': 'Structure_rag_mem0'}
        with tempfile.TemporaryDirectory() as directory, patch.object(initialization, 'AgentWrapper', LocalAgent):
            with patch.dict('os.environ', {'MEMORYDATA_MEM0_CHECKPOINT_ENABLED': '1'}):
                agent, _ = initialization.initialize_and_memorize_agent(config, self.dataset, directory, ['one'], 0, 1)
                agent.close()
            with patch.dict('os.environ', {'MEMORYDATA_MEM0_CHECKPOINT_ENABLED': '0'}):
                with self.assertRaisesRegex(ValueError, 'explicitly enable'):
                    initialization.initialize_and_memorize_agent(config, self.dataset, directory, ['one'], 0, 1)


if __name__ == '__main__':
    unittest.main()
