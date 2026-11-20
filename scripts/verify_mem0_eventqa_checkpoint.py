"""Verify the production checkpoint on an isolated copy; no model/API calls."""
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from utils.artifact_paths import rebase_legacy_artifact_path
from utils.mem0_checkpoint import Mem0Checkpoint
from utils.provider_utils import resolve_model_config


def main():
    load_dotenv(ROOT / '.env')
    run = ROOT / 'results/full_first_units_20261023'
    os.environ['OPENAI_API_TRUST_ENV'] = 'true'
    config = resolve_model_config(yaml.safe_load((run / 'configs/mem0_eventqa.yaml').read_text(encoding='utf-8')))
    config['output_dir'] = rebase_legacy_artifact_path(config['output_dir'], config['artifact_root'])
    dataset = yaml.safe_load((run / 'configs/eventqa.yaml').read_text(encoding='utf-8'))
    chunks = json.loads((run / 'input/eventqa.json').read_text(encoding='utf-8'))['chunks']
    folders = list((run / 'artifacts/eventqa/mem0/agents').glob('*/exp_0'))
    if len(folders) != 1:
        raise ValueError('Expected exactly one production Mem0 context')
    production = Mem0Checkpoint(folders[0], chunks, config, dataset, 0)
    production.validate()
    target = run / 'checkpoint_validation' / uuid.uuid4().hex
    shutil.copytree(folders[0], target)
    copied = Mem0Checkpoint(target, chunks, config, dataset, 0)
    copied.restore_before_open()
    assert copied.next_chunk_index == production.next_chunk_index
    assert (target / 'mem0_source_map.json').exists() == copied.complete
    client = QdrantClient(path=str(target / 'mem0_runtime/qdrant'))
    try:
        points = client.get_collection('mem0_eventqa_full').points_count
    finally:
        client.close()
    import sqlite3
    with sqlite3.connect(target / 'mem0_runtime/history.db') as database:
        integrity = database.execute('PRAGMA integrity_check').fetchone()[0]
        assert integrity == 'ok'
        history = database.execute('SELECT count(*) FROM history').fetchone()[0]
    result = {'production_folder': str(folders[0]), 'validated_copy': str(target),
              'production_untouched': True, 'committed_chunks': copied.next_chunk_index,
              'total_chunks': len(chunks), 'complete': copied.complete, 'snapshot_hashes_valid': True,
              'restored_copy_qdrant_points': points, 'restored_copy_history_rows': history,
              'restored_copy_sqlite_integrity': integrity, 'model_api_calls': 0}
    (run / 'mem0_checkpoint_production_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
