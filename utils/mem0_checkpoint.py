"""Opt-in, closed-storage Mem0 chunk checkpoints; incomplete writes are rolled back."""
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import uuid

from utils.repo_paths import legacy_variants, portable


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class Mem0Checkpoint:
    def __init__(self, folder, chunks, agent_config, dataset_config, context_id):
        self.folder = Path(folder).resolve()
        self.root = self.folder / 'mem0_checkpoints'
        self.pointer = self.folder / 'mem0_ingestion_checkpoint.json'
        self._inside(self.root)
        if agent_config.get('mem0_graph_store'):
            raise ValueError('Chunk checkpoints support local vector-only Mem0, not an external graph store')
        repo = Path(__file__).resolve().parents[1]
        sources = ('methods/mem0/source/mem0/memory/main.py',
                   'benchmark/memoryagentbench/prompts/benchmark_templates.py')
        source_hashes = {name: digest(repo / name) for name in sources}

        # Persist only hashes, never credential-bearing configuration values.
        def identity(agent, dataset, **location):
            return fingerprint({'checkpoint_protocol': 1, 'chunks': chunks, 'agent': agent, 'dataset': dataset,
                                'context_id': context_id, 'source_hashes': source_hashes, **location})

        # In-repository paths (artifact_root, output_dir, data files) count
        # relative to the checkout, so moving the whole repository keeps
        # checkpoints valid. Older absolute-path identities are accepted as is.
        self.identity = identity(portable(agent_config), portable(dataset_config), location='repo-relative')
        self.accepted_identities = {self.identity} | {identity(*value) for value in
                                                      legacy_variants([agent_config, dataset_config])}
        self.chunks = chunks
        self.manifest = None
        self.generation = None
        self.complete = False

    def _inside(self, path, parent=None):
        parent = (parent or self.folder).resolve()
        resolved = path.resolve()
        if resolved == parent or not resolved.is_relative_to(parent):
            raise ValueError('Checkpoint path escapes its owner folder: ' + str(path))
        return resolved

    @property
    def next_chunk_index(self):
        return self.manifest['next_chunk_index'] if self.manifest else 0

    @property
    def write_times(self):
        return list(self.manifest['write_times']) if self.manifest else []

    def validate(self):
        pointer = json.loads(self.pointer.read_text(encoding='utf-8'))
        if pointer.get('version') != 1:
            raise ValueError('Unsupported checkpoint version')
        name = pointer['generation']
        if not re.fullmatch(r'generation-[0-9a-f]{32}', name):
            raise ValueError('Invalid checkpoint generation name')
        generation = self._inside(self.root / name, self._inside(self.root))
        manifest_path = self._inside(generation / 'manifest.json', generation)
        if digest(manifest_path) != pointer['manifest_sha256']:
            raise ValueError('Checkpoint manifest hash mismatch')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('version') != 1:
            raise ValueError('Unsupported checkpoint manifest version')
        if manifest['identity'] not in self.accepted_identities or manifest['total_chunks'] != len(self.chunks):
            raise ValueError('Checkpoint input/config/source identity differs; refusing partial reuse')
        index, times = manifest['next_chunk_index'], manifest['write_times']
        if type(index) is not int or not 0 <= index <= len(self.chunks) or len(times) != index:
            raise ValueError('Invalid checkpoint progress')
        if any(not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0 for t in times):
            raise ValueError('Invalid checkpoint timing')
        actual = {p.relative_to(generation).as_posix() for p in generation.rglob('*') if p.is_file() and p != manifest_path}
        if actual != set(manifest['files']):
            raise ValueError('Checkpoint file inventory differs')
        for relative, expected in manifest['files'].items():
            path = self._inside(generation / relative, generation)
            if digest(path) != expected:
                raise ValueError('Checkpoint file hash mismatch: ' + relative)
        if pointer.get('complete') and index != len(self.chunks):
            raise ValueError('Incomplete history cannot be marked complete')
        self.manifest, self.generation, self.complete = manifest, generation, bool(pointer.get('complete'))
        return manifest

    def restore_before_open(self):
        if not self.pointer.exists():
            runtime = self.folder / 'mem0_runtime'
            if runtime.exists() and any(runtime.iterdir()) and not (self.folder / 'mem0_source_map.json').exists():
                raise ValueError('Partial Mem0 state has no safe checkpoint; archive it before enabling recovery')
            return
        self.validate()  # Verify all identity/integrity checks before changing live files.
        self.root.mkdir(parents=True, exist_ok=True)
        uncertain = self._inside(self.root / ('uncertain-' + uuid.uuid4().hex), self.root)
        uncertain.mkdir()
        for name in ('mem0_runtime', 'mem0_source_map.json'):
            live = self._inside(self.folder / name)
            if live.exists():
                shutil.move(str(live), str(uncertain / name))
        shutil.copytree(self.generation / 'runtime', self.folder / 'mem0_runtime')
        if self.complete:
            shutil.copy2(self.generation / 'source_map.json', self.folder / 'mem0_source_map.json')

    def attach_provenance(self, agent):
        if self.manifest:
            agent.mem0_source_map = json.loads((self.generation / 'source_map.json').read_text(encoding='utf-8'))

    def close_storage(self, agent):
        memory = agent.memory
        if memory.enable_graph:
            raise ValueError('Cannot snapshot an external graph')
        memory.vector_store.client.close()
        memory.db.connection.commit()
        memory.db.connection.close()

    def reopen_storage(self, agent):
        from qdrant_client import QdrantClient
        from mem0.memory.storage import SQLiteManager
        agent.memory.vector_store.client = QdrantClient(path=str(self.folder / 'mem0_runtime/qdrant'))
        agent.memory.db = SQLiteManager(str(self.folder / 'mem0_runtime/history.db'))

    def capture(self, agent, next_chunk_index, write_times):
        expected = self.next_chunk_index + 1 if self.manifest else 0
        if next_chunk_index != expected or len(write_times) != next_chunk_index:
            raise ValueError('A checkpoint can only commit the next fully processed chunk')
        runtime = self._inside(self.folder / 'mem0_runtime')
        if Path(agent.memory.config.history_db_path).resolve() != runtime / 'history.db' or \
                Path(agent.memory.config.vector_store.config.path).resolve() != runtime / 'qdrant':
            raise ValueError('Checkpoint storage differs from this agent local runtime')
        self.root.mkdir(parents=True, exist_ok=True)
        generation = self._inside(self.root / ('generation-' + uuid.uuid4().hex), self.root)
        generation.mkdir()
        previous = self.generation
        self.close_storage(agent)
        try:
            shutil.copytree(runtime, generation / 'runtime')
            atomic_json(generation / 'source_map.json', agent.mem0_source_map)
            files = {p.relative_to(generation).as_posix(): digest(p) for p in generation.rglob('*') if p.is_file()}
            manifest = {'version': 1, 'identity': self.identity, 'total_chunks': len(self.chunks),
                        'next_chunk_index': next_chunk_index, 'write_times': list(write_times), 'files': files,
                        'created_at': datetime.now().astimezone().isoformat()}
            atomic_json(generation / 'manifest.json', manifest)
            atomic_json(self.pointer, {'version': 1, 'generation': generation.name,
                                      'manifest_sha256': digest(generation / 'manifest.json'), 'complete': False})
            self.manifest, self.generation, self.complete = manifest, generation, False
        finally:
            self.reopen_storage(agent)
        # Keep current and previous committed generations. Validate every deletion's
        # resolved path against this exact checkpoint root before recursive removal.
        for old in self.root.glob('generation-*'):
            if old not in (generation, previous) and re.fullmatch(r'generation-[0-9a-f]{32}', old.name):
                shutil.rmtree(self._inside(old, self.root))

    def record_failure(self, agent, chunk_index, error):
        from utils.model_api_diagnostics import error_details
        atomic_json(self.root / ('failure-' + uuid.uuid4().hex + '.json'), {
            'time': datetime.now().astimezone().isoformat(), 'identity': self.identity,
            'failed_chunk_index': chunk_index, 'committed_chunks': self.next_chunk_index,
            'chunk_sha256': fingerprint(self.chunks[chunk_index]), **error_details(error)})
        # Partial storage is left for forensics; resume restores the immutable
        # committed generation before opening either database.
        self.close_storage(agent)

    def finish(self):
        if self.next_chunk_index != len(self.chunks):
            raise ValueError('Cannot finish an incomplete Mem0 checkpoint')
        pointer = json.loads(self.pointer.read_text(encoding='utf-8'))
        pointer['complete'] = True
        atomic_json(self.pointer, pointer)
        self.complete = True
