"""Complete first-context evaluation with audited full-history snapshot reuse."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_benchmark_pilot import now, save, wait_child, stop_server


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_result(path, qa, expected_hash=None):
    path = Path(path)
    if expected_hash and digest(path) != expected_hash:
        raise ValueError('Previously verified result SHA-256 changed')
    data = json.loads(path.read_text(encoding='utf-8'))
    rows = data['data']
    if len(rows) != len(qa) or {r['query_id'] for r in rows} != set(range(len(qa))):
        raise ValueError('Full question coverage is incomplete or duplicated')
    for row, expected in zip(sorted(rows, key=lambda x: x['query_id']), qa):
        if row.get('status') == 'failed' or not str(row.get('output') or '').strip():
            raise ValueError('Failed/empty answer in full evaluation')
        if row['context_id'] != 0 or row.get('qa_pair_id') != expected[2] or row['query'] != expected[0]:
            raise ValueError('Question identity/content differs from frozen full input')
    return {'result_file': str(path.resolve()), 'result_sha256': digest(path), 'completed': len(rows),
            'failed_questions': 0, 'averaged_metrics': data['averaged_metrics']}


def transient_failed_queries(path):
    """Only repeat failed model requests, never valid answers or content-policy errors."""
    rows = json.loads(Path(path).read_text(encoding='utf-8'))['data']
    failed = [row for row in rows if row.get('status') == 'failed' or not str(row.get('output') or '').strip()]
    if failed and all(str(row.get('error', '')).strip().lower() in ('request timed out.', 'connection error.')
                      for row in failed):
        return [row['query_id'] for row in failed]
    return []


def expected_documents(chunks):
    from methods.hindsight.hindsight_adapter import fingerprint
    from utils.locomo_utils import parse_locomo_source_ids, strip_locomo_metadata
    import re
    result = {}
    for chunk in chunks:
        raw = str(chunk.get('storage_text') or chunk.get('text') or '') if isinstance(chunk, dict) else chunk
        text = strip_locomo_metadata(raw).replace('\r\n', '\n').replace('\r', '\n').strip()
        text = re.sub(r'\n{3,}', '\n\n', text)
        if text:
            result[f'md-chunk-{len(result):08d}-{fingerprint(text)[:16]}'] = parse_locomo_source_ids(raw)
    return result


def prepare(run):
    from utils.artifact_paths import generate_agent_save_folder_path
    from utils.initialization import _create_output_path
    from methods.hindsight.hindsight_adapter import fingerprint
    source_root = ROOT / 'results/other_benchmarks_small_20261022'
    source = json.loads((source_root / 'manifest.json').read_text(encoding='utf-8'))
    selection = json.loads((run / 'network_selection.json').read_text(encoding='utf-8'))
    if selection['selected_trust_env'] is None:
        raise RuntimeError('No model route passed preflight')
    cases = []
    (run / 'configs').mkdir(exist_ok=True)
    (run / 'input').mkdir(exist_ok=True)
    for original in source['units']:
        name = original['name']
        payload_path = source_root / f'input/{name}_selected_history_and_qa.json'
        payload = json.loads(payload_path.read_text(encoding='utf-8'))
        if fingerprint(payload['chunks']) != original['history_sha256']:
            raise ValueError(f'Frozen complete history changed: {name}')
        if len(payload['chunks']) != original['chunk_count'] or len(payload['qa']) != original['available_questions']:
            raise ValueError('Full input counts differ from manifest')
        shutil.copy2(payload_path, run / f'input/{name}.json')
        dataset = yaml.safe_load(Path(original['dataset_config']).read_text(encoding='utf-8'))
        assert not dataset.get('smoke_max_history_chunks') and dataset['max_test_samples'] == 1
        # Retain the original selection/tag for memory identity; question count is unlimited.
        dataset_path = run / f'configs/{name}.yaml'
        dataset_path.write_text(yaml.safe_dump(dataset, allow_unicode=True, sort_keys=False), encoding='utf-8')
        for method in ('mem0', 'hindsight'):
            key = f'{method}_{name}'
            old_status = json.loads((source_root / f'{key}.status.json').read_text(encoding='utf-8'))
            config = yaml.safe_load(Path(source['agents'][method]).read_text(encoding='utf-8'))
            config['model_api_trust_env'] = selection['selected_trust_env']
            artifact = run / f'artifacts/{name}/{method}'
            config['artifact_root'] = str(artifact)
            config_path = run / f'configs/{key}.yaml'
            case = {'id': key, 'method': method, 'task': name, 'phase': 'prepared',
                    'expected_questions': len(payload['qa']), 'chunk_count': len(payload['chunks']),
                    'context_characters': original['context_characters'], 'history_sha256': original['history_sha256'],
                    'input_file': str(run / f'input/{name}.json'), 'dataset_config': str(dataset_path),
                    'agent_config': str(config_path), 'artifact_root': str(artifact), 'reuse_kind': 'fresh'}
            if old_status.get('phase') == 'success':
                old_file = Path(old_status['result_file'])
                count = old_status['completed']
                checked = verify_result(old_file, payload['qa'][:count], old_status['result_sha256'])
                if count == len(payload['qa']):
                    case.update(phase='success', reuse_kind='previous_full_result', **checked)
                    case['previous_status'] = str(source_root / f'{key}.status.json')
                    cases.append(case)
                    continue
                case['seed_file'] = str(old_file)
                case['seed_sha256'] = checked['result_sha256']
                case['seed_answers'] = count
            old_dirs = list((source_root / f'artifacts/{name}/{method}/agents').glob('*/exp_0'))
            if len(old_dirs) == 1:
                old_dir = old_dirs[0]
                marker = old_dir / ('hindsight_state.json' if method == 'hindsight' else 'mem0_source_map.json')
                can_reuse = method == 'mem0' and old_status.get('phase') == 'success' and marker.exists()
                if method == 'hindsight' and marker.exists():
                    state = json.loads(marker.read_text(encoding='utf-8'))
                    can_reuse = state.get('documents') == expected_documents(payload['chunks'])
                    if can_reuse:
                        config['hindsight_snapshot_origin'] = {'state_dir': str(old_dir.resolve()), 'state_sha256': digest(marker)}
                        case['reuse_kind'] = 'saved_complete' if state['ready'] else 'ingested_pending'
                if can_reuse:
                    dest = Path(generate_agent_save_folder_path(config, dataset, 0, str(artifact)))
                    if dest.exists():
                        raise FileExistsError(f'Refusing to overwrite memory snapshot: {dest}')
                    shutil.copytree(old_dir, dest)
                    case.update(memory_dir=str(dest), source_memory_dir=str(old_dir),
                                source_marker_sha256=digest(marker))
                    if method == 'mem0':
                        case['reuse_kind'] = 'saved_complete'
                    case['server_state_dir'] = str(run / 'native_reuse_service') if method == 'hindsight' else None
            if method == 'hindsight' and case['reuse_kind'] == 'fresh':
                config['hindsight_settle_timeout_seconds'] = 36000
                case['server_state_dir'] = str(run / f'native_services/{key}')
            config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
            if case.get('seed_file'):
                from utils.artifact_paths import rebase_legacy_artifact_path
                output_config = {**config, 'output_dir': rebase_legacy_artifact_path(config['output_dir'], str(artifact))}
                output = Path(_create_output_path(output_config, dataset))
                shutil.copy2(case['seed_file'], output)
                case['seed_destination'] = str(output)
            cases.append(case)
    locomo = ROOT / 'results/locomo_first_original_20261021'
    locomo_summary = json.loads((locomo / 'summary.json').read_text(encoding='utf-8'))
    for method, result in locomo_summary['methods'].items():
        path = Path(result['result_file'])
        if digest(path) != result['result_sha256']:
            raise ValueError('LoCoMo complete result changed')
        data = json.loads(path.read_text(encoding='utf-8'))
        rows = data['data']
        assert len(rows) == 199 and {row['query_id'] for row in rows} == set(range(199))
        assert all(row.get('status') != 'failed' and str(row.get('output') or '').strip() for row in rows)
        cases.append({'id': f'{method}_locomo', 'method': method, 'task': 'locomo', 'phase': 'success',
                      'reuse_kind': 'previous_full_result', 'expected_questions': 199, 'completed': 199,
                      'result_file': str(path), 'result_sha256': digest(path), 'averaged_metrics': data['averaged_metrics']})
    rank = {'previous_full_result': 0, 'saved_complete': 1, 'fresh': 2, 'ingested_pending': 3}
    cases.sort(key=lambda c: (rank[c['reuse_kind']], c.get('context_characters', 0), c['method']))
    manifest = {'scope': 'First COMPLETE context/scenario per dataset slice, ALL questions; no history truncation, no independent LLM judge',
                'expected_questions_per_method': 565, 'expected_answers': 1130,
                'source_manifest': str(source_root / 'manifest.json'), 'sources': source['sources'],
                'mem0_fact_limit_per_chunk': 2, 'cases': cases,
                'longbench_note': source['longbench_note'],
                'trust_env': selection['selected_trust_env']}
    assert len(cases) == 22 and sum(c['expected_questions'] for c in cases) == 1130
    save(run / 'manifest.json', manifest)
    # Clone the stopped native PostgreSQL cluster, preserving all old artifacts.
    old_server = source_root / 'hindsight_server'
    old_state = json.loads((old_server / 'status.json').read_text(encoding='utf-8'))
    if old_state['phase'] != 'stopped' or (old_server / 'database/postmaster.pid').exists():
        raise RuntimeError('Cannot snapshot a running Hindsight database')
    source_db, target_db = old_server / 'database', run / 'native_reuse_service/database'
    shutil.copytree(source_db, target_db)
    hashes = {}
    for file in source_db.rglob('*'):
        if file.is_file():
            relative = file.relative_to(source_db)
            source_hash = digest(file)
            if digest(target_db / relative) != source_hash:
                raise ValueError('Native database snapshot copy differs')
            hashes[str(relative)] = source_hash
    save(run / 'native_reuse_service/snapshot_manifest.json', {'source': str(source_db), 'files': hashes})
    return manifest


def resume_ingestion(case):
    from methods.hindsight.hindsight_adapter import HindsightAdapter, fingerprint
    from utils.provider_utils import resolve_model_config
    config = resolve_model_config(yaml.safe_load(Path(case['agent_config']).read_text(encoding='utf-8')))
    dataset = yaml.safe_load(Path(case['dataset_config']).read_text(encoding='utf-8'))
    payload = json.loads(Path(case['input_file']).read_text(encoding='utf-8'))
    config['_hindsight_history_fingerprint'] = fingerprint(payload['chunks'])
    adapter = HindsightAdapter(config, dataset, case['memory_dir'])
    adapter.settle_timeout = 36000  # Wait budget only; does not change the original identity contract.
    before = adapter.resume_completed_ingestion(expected_documents(payload['chunks']))
    save(Path(case['artifact_root']) / 'settlement_recovery.json', {
        'phase': 'settled', 'new_retain_calls': 0, 'verified_documents': len(adapter.state['documents']),
        'native_stats_before': before, 'native_stats_after': adapter.state['settled_stats'], 'ended_at': now()})


def start_server(case, env, deadline):
    state_dir = Path(case['server_state_dir'])
    state_dir.mkdir(parents=True, exist_ok=True)
    output = (state_dir / 'launcher.log').open('a', encoding='utf-8')
    server_started_wall = time.time()
    server = subprocess.Popen([r'D:\conda\envs\hindsight\python.exe', '-u', str(ROOT / 'scripts/run_hindsight_server.py'),
        '--agent_config', case['agent_config'], '--state_dir', str(state_dir), '--port', '8889'],
        cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
    start_deadline = min(deadline, time.monotonic() + 180)
    try:
        while time.monotonic() < start_deadline:
            if server.poll() is not None:
                raise RuntimeError('Hindsight launcher exited during startup')
            status_file = state_dir / 'status.json'
            if status_file.exists():
                status = json.loads(status_file.read_text(encoding='utf-8'))
                # A previous stopped/ready file alone is not a new readiness event.
                if status.get('phase') == 'ready' and status_file.stat().st_mtime >= server_started_wall:
                    return server, output
            time.sleep(0.5)
        raise TimeoutError('Hindsight startup timeout')
    except Exception:
        stop_server(server, state_dir)
        output.close()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=Path, required=True)
    parser.add_argument('--prepare_only', action='store_true')
    parser.add_argument('--resume_ingestion_only', default='')
    parser.add_argument('--resume', action='store_true', help='Continue a stopped suite, retaining successful cases and answers')
    parser.add_argument('--retry_blocked_case', action='append', default=[],
                        help='Explicitly retry a named blocked case on resume, preserving successes')
    parser.add_argument('--mem0_checkpoint', action='store_true',
                        help='Opt in to verified, closed-storage Mem0 chunk recovery')
    args = parser.parse_args()
    if args.retry_blocked_case and not args.resume:
        parser.error('--retry_blocked_case requires --resume')
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    load_dotenv(ROOT / '.env')
    manifest_file = run / 'manifest.json'
    manifest = json.loads(manifest_file.read_text(encoding='utf-8')) if manifest_file.exists() else prepare(run)
    if args.prepare_only:
        print(json.dumps({'phase': 'prepared', 'cases': len(manifest['cases']), 'answers': manifest['expected_answers']}), flush=True)
        return 0
    if args.resume_ingestion_only:
        case = next(c for c in manifest['cases'] if c['id'] == args.resume_ingestion_only)
        resume_ingestion(case)
        return 0
    previous_cases = {}
    attempt_number = 1
    if (run / 'suite.status.json').exists():
        if not args.resume:
            raise FileExistsError('Suite already started; use explicit --resume to retain previous attempts')
        previous = json.loads((run / 'suite.status.json').read_text(encoding='utf-8'))
        if previous['phase'] == 'running':
            raise RuntimeError('Refusing to start a second coordinator for a running suite')
        archive_number = len(list(run.glob('suite.attempt_*.status.json'))) + 1
        shutil.copy2(run / 'suite.status.json', run / f'suite.attempt_{archive_number:03d}.status.json')
        previous_cases = {case['id']: case for case in previous['cases']}
        for case_id in args.retry_blocked_case:
            if previous_cases.get(case_id, {}).get('phase') != 'blocked':
                raise ValueError('Explicit blocked retry must name a previously blocked case: ' + case_id)
        archive = run / 'attempt_history' / f'attempt_{archive_number}'
        for previous_case in previous['cases']:
            if previous_case['phase'] != 'failed' and previous_case['id'] not in args.retry_blocked_case:
                continue
            files = [run / f'{previous_case["id"]}.log', run / f'{previous_case["id"]}.settlement.log',
                     run / f'{previous_case["id"]}.status.json']
            files.extend((Path(previous_case['artifact_root']) / 'outputs').rglob('*.json'))
            for source_file in files:
                if source_file.is_file():
                    destination = archive / source_file.relative_to(run)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, destination)
            if previous_case['method'] == 'mem0':
                artifact = Path(previous_case['artifact_root']).resolve()
                markers = list((artifact / 'agents').glob('*/exp_0/mem0_source_map.json'))
                checkpoints = list((artifact / 'agents').glob('*/exp_0/mem0_ingestion_checkpoint.json'))
                if checkpoints and not args.mem0_checkpoint:
                    raise ValueError('Partial chunk checkpoints exist; --mem0_checkpoint is required to resume them safely')
                if artifact.exists() and not markers and not checkpoints:
                    if not artifact.is_relative_to(run):
                        raise ValueError('Refusing to move a partial memory outside the current run')
                    destination = archive / 'partial_artifacts' / previous_case['id']
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if destination.exists():
                        raise FileExistsError('Partial memory archive already exists')
                    shutil.move(str(artifact), str(destination))
        attempt_number = archive_number + 1
    source_hashes = {}
    for relative in ('scripts/run_full_first_units.py', 'scripts/run_benchmark_pilot.py', 'scripts/run_hindsight_server.py',
                     'main.py', 'utils/initialization.py', 'utils/agent.py', 'utils/artifact_paths.py',
                     'utils/conversation_creator.py', 'utils/eval_other_utils.py', 'utils/provider_utils.py',
                     'utils/mem0_checkpoint.py', 'utils/model_api_diagnostics.py',
                     'methods/hindsight/hindsight_adapter.py', 'methods/hindsight/api_config.py',
                     'methods/mem0/source/mem0/llms/openai.py',
                     'methods/mem0/source/mem0/embeddings/openai.py',
                     'methods/mem0/source/mem0/memory/main.py', 'benchmark/memoryagentbench/prompts/benchmark_templates.py'):
        target = run / ('source_snapshot' if attempt_number == 1 else f'source_snapshot_attempt_{attempt_number}') / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        source_hashes[relative] = digest(target)
    save(run / ('code_sha256.json' if attempt_number == 1 else f'code_sha256_attempt_{attempt_number}.json'), source_hashes)
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1', MEM0_TELEMETRY='false',
               OPENAI_API_TRUST_ENV=str(manifest['trust_env']).lower(), HINDSIGHT_API_URL='http://127.0.0.1:8889')
    env.update(OPENAI_API_CONNECT_TIMEOUT_SECONDS='30', OPENAI_API_MAX_RETRIES='10')
    env['MEMORYDATA_API_DIAGNOSTICS_DIR'] = str(run / f'api_diagnostics_attempt_{attempt_number}')
    if args.mem0_checkpoint:
        env['MEMORYDATA_MEM0_CHECKPOINT_ENABLED'] = '1'
    if not manifest['trust_env']:
        env = {k: v for k, v in env.items() if k.upper() not in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY')}
    state = {'phase': 'running', 'started_at': now(), 'scope': manifest['scope'], 'cases': [], 'attempt_number': attempt_number,
             'explicit_blocked_retries': args.retry_blocked_case,
             'mem0_chunk_checkpoint_enabled': args.mem0_checkpoint,
             'memorydata_request_policy': {'connect_timeout_seconds': 30, 'sdk_max_retries': 10,
                                          'read_timeout_seconds': 600,
                                          'applies_to': 'MemoryData answer, native Mem0 clients and native Hindsight embeddings; Hindsight LLM retains its own retry loop'}}
    save(run / 'suite.status.json', state)
    stop = lambda: (run / 'stop.request').exists()
    abort = False

    def event(value):
        print(json.dumps({'time': now(), **value}, ensure_ascii=False), flush=True)

    for case in manifest['cases']:
        record = dict(case)
        status_file = run / f'{case["id"]}.status.json'
        previous = previous_cases.get(case['id'])
        if previous and (previous['phase'] == 'success' or
                         previous['phase'] == 'blocked' and case['id'] not in args.retry_blocked_case):
            if previous['phase'] == 'success' and digest(previous['result_file']) != previous['result_sha256']:
                raise ValueError('Successful prior result changed during resume')
            save(status_file, previous)
            state['cases'].append(previous)
            save(run / 'suite.status.json', state)
            continue
        if case['phase'] == 'success':
            record['imported_at'] = now()
            save(status_file, record)
            state['cases'].append(record)
            save(run / 'suite.status.json', state)
            continue
        if abort or stop():
            record.update(phase='not_run', reason='User stop or systemic connection/shutdown failure')
            save(status_file, record)
            state['cases'].append(record)
            save(run / 'suite.status.json', state)
            continue
        record.update(phase='running', started_at=now())
        save(status_file, record)
        event({'event': 'started', 'id': case['id'], 'reuse_kind': case['reuse_kind'],
               'history_chunks': case['chunk_count'], 'questions': case['expected_questions']})
        deadline = time.monotonic() + 43200  # Independent 12-hour guard for large complete histories.
        server = output = None
        try:
            if case['method'] == 'hindsight':
                server, output = start_server(case, env, deadline)
                record['server_launcher_pid'] = server.pid
                save(status_file, record)
                if case['reuse_kind'] == 'ingested_pending':
                    with (run / f'{case["id"]}.settlement.log').open('w', encoding='utf-8') as log:
                        child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--run_dir', str(run),
                            '--resume_ingestion_only', case['id']], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                        record.update(stage='settling_existing_ingestion', pid=child.pid)
                        save(status_file, record)
                        code = wait_child(child, deadline, stop)
                        if code:
                            raise RuntimeError('Verified complete-ingestion recovery failed; see settlement log')
            command = [sys.executable, '-u', str(ROOT / 'main.py'), '--agent_config', case['agent_config'],
                       '--dataset_config', case['dataset_config'], '--artifact_root', case['artifact_root'],
                       '--max_test_queries_ablation', '0', '--retry_failed_queries']
            with (run / f'{case["id"]}.log').open('w', encoding='utf-8') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                record.update(stage='evaluating', pid=child.pid, command=command)
                save(status_file, record)
                record['exit_code'] = wait_child(child, deadline, stop)
            if record['exit_code']:
                raise RuntimeError('main.py exited with error')
            files = list((Path(case['artifact_root']) / 'outputs').rglob('*_results.json'))
            if len(files) != 1:
                raise ValueError('Expected exactly one full result file')
            # The framework records individual SDK errors and continues. Give
            # transient failed queries two bounded recovery passes without
            # rebuilding memory or re-asking already successful questions.
            for retry_pass in range(2):
                failed_ids = transient_failed_queries(files[0])
                if not failed_ids:
                    break
                record.setdefault('query_retry_passes', []).append({'pass': retry_pass + 1, 'query_ids': failed_ids})
                save(status_file, record)
                with (run / f'{case["id"]}.log').open('a', encoding='utf-8') as log:
                    log.write(f'\nRetrying transient failed query IDs only: {failed_ids}\n')
                    log.flush()
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    record['pid'] = child.pid
                    save(status_file, record)
                    if wait_child(child, deadline, stop):
                        raise RuntimeError('Transient query recovery process failed')
            qa = json.loads(Path(case['input_file']).read_text(encoding='utf-8'))['qa']
            record.update(phase='success', **verify_result(files[0], qa))
        except Exception as error:
            text = '\n'.join(p.read_text(encoding='utf-8', errors='replace') for p in
                             (run / f'{case["id"]}.log', run / f'{case["id"]}.settlement.log') if p.exists())
            category = ('content_filter' if 'content_filter' in text else 'model_network_failure'
                        if 'APIConnectionError' in text or 'APITimeoutError' in text else type(error).__name__)
            record.update(phase='blocked' if category == 'content_filter' else 'failed',
                          failure_category=category, error_type=type(error).__name__, error=str(error))
            if category == 'model_network_failure':
                abort = True
            event({'event': 'attention', 'id': case['id'], 'failure_category': category, 'error': str(error)})
        finally:
            if server is not None:
                try:
                    record['shutdown'] = stop_server(server, Path(case['server_state_dir']))
                except Exception as error:
                    record.update(phase='failed', shutdown_error=str(error))
                    abort = True
                    event({'event': 'attention', 'id': case['id'], 'shutdown_error': str(error)})
            if output:
                output.close()
            record['ended_at'] = now()
            save(status_file, record)
            state['cases'].append(record)
            save(run / 'suite.status.json', state)
        event({'event': 'finished', 'id': case['id'], 'phase': record['phase']})
    state.update(phase='success' if all(c['phase'] == 'success' for c in state['cases']) else 'partial', ended_at=now())
    save(run / 'suite.status.json', state)
    event({'event': 'suite_finished', 'phase': state['phase']})
    return 0 if state['phase'] == 'success' else 1


if __name__ == '__main__':
    raise SystemExit(main())
