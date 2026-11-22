"""Bounded pilot runner. Every Hindsight task owns a separate service/database.

Aborting a client never authorizes a new task to share unfinished backend work.
All source data and previous evaluation artifacts remain untouched.
"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def save(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def stop_child(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def wait_child(process, deadline, stop_requested=lambda: False):
    """Wait on an owned child; enforce deadline/cancellation and always reap it."""
    try:
        while True:
            if stop_requested():
                raise InterruptedError('Pilot stop requested')
            if time.monotonic() >= deadline:
                raise TimeoutError('Pilot task/suite time budget exhausted')
            try:
                return process.wait(timeout=min(0.5, max(0.01, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
    finally:
        stop_child(process)


def stop_server(server, state_dir):
    """Let the launcher stop its native worker and database; never orphan it."""
    (state_dir / 'stop.request').touch()
    try:
        server.wait(timeout=60)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f'Owned Hindsight launcher PID {server.pid} did not stop; no further tasks may start')
    status_path = state_dir / 'status.json'
    status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
    if status.get('phase') not in ('stopped', 'failed'):
        raise RuntimeError('Hindsight launcher exited without terminal shutdown state')
    return {'launcher_exit_code': server.returncode, 'service_status': status}


def prepare(source_path, run_dir, trust_env, chunk_limit, query_limit, settle_timeout):
    source = json.loads(source_path.read_text(encoding='utf-8'))
    agents = {}
    for method, path in source['agents'].items():
        config = yaml.safe_load(pathlib.Path(path).read_text(encoding='utf-8'))
        config.update(model_api_trust_env=trust_env)
        config['agent_variant_tag'] += '-smoke-history2-qa2'
        if method == 'hindsight':
            config['hindsight_settle_timeout_seconds'] = settle_timeout
        dest = run_dir / f'configs/{method}.yaml'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
        agents[method] = str(dest)
    units = []
    for original in source['units']:
        config = yaml.safe_load(pathlib.Path(original['dataset_config']).read_text(encoding='utf-8'))
        config.update(smoke_max_history_chunks=chunk_limit, tag='truncated_history_smoke_20261022')
        dest = run_dir / f'configs/{original["name"]}.yaml'
        dest.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
        original_input = source_path.parent / f'input/{original["name"]}_selected_history_and_qa.json'
        payload = json.loads(original_input.read_text(encoding='utf-8'))
        chunks, qa = payload['chunks'][:chunk_limit], payload['qa'][:query_limit]
        save(run_dir / f'input/{original["name"]}_selected_history_and_qa.json', {'chunks': chunks, 'qa': qa})
        units.append({**original, 'dataset_config': str(dest), 'original_chunk_count': original['chunk_count'],
                      'original_context_characters': original['context_characters'], 'chunk_count': len(chunks),
                      'context_characters': sum(len(x['text'] if isinstance(x, dict) else x) for x in chunks),
                      'expected_questions': len(qa),
                      'history_sha256': hashlib.sha256(json.dumps(chunks, ensure_ascii=False).encode()).hexdigest()})
    manifest = {'scope': 'TRUNCATED HISTORY SMOKE ONLY; first context, first chunks and questions; not comparable to full benchmark scores',
                'source_manifest': str(source_path), 'history_chunk_limit': chunk_limit, 'question_limit': query_limit,
                'trust_env': trust_env, 'agents': agents, 'units': units}
    save(run_dir / 'manifest.json', manifest)
    return manifest


def select_units(units, names='', order='smallest-first'):
    """Allow an explicit sequence without changing the default small-first pilot."""
    requested = list(dict.fromkeys(filter(None, names.split(','))))
    by_name = {unit['name']: unit for unit in units}
    if names and (not requested or set(requested) - by_name.keys()):
        raise ValueError('Unknown or empty task selection')
    selected = [by_name[name] for name in requested] if requested else list(units)
    if order == 'selection':
        return selected
    return sorted(selected, key=lambda unit: unit['context_characters'])


def validate_result(run_dir, unit, method):
    files = list((run_dir / f'artifacts/{unit["name"]}/{method}/outputs').rglob('*_results.json'))
    if len(files) != 1:
        raise ValueError(f'Expected one result, found {len(files)}')
    result = json.loads(files[0].read_text(encoding='utf-8'))
    rows = result['data']
    expected = json.loads((run_dir / f'input/{unit["name"]}_selected_history_and_qa.json').read_text(encoding='utf-8'))['qa']
    if len(rows) != len(expected) or {r['query_id'] for r in rows} != set(range(len(expected))):
        raise ValueError('Incomplete or duplicate QA coverage')
    for row, qa in zip(sorted(rows, key=lambda r: r['query_id']), expected):
        if row.get('status') == 'failed' or not str(row.get('output') or '').strip():
            raise ValueError('Failed or empty answer')
        if row['context_id'] != 0 or row.get('qa_pair_id') != qa[2]:
            raise ValueError('Input/answer identity mismatch')
    if result['dataset_config'].get('smoke_max_history_chunks') != 2:
        raise ValueError('Smoke history limit missing from recorded result')
    return {'result_file': str(files[0]), 'result_sha256': hashlib.sha256(files[0].read_bytes()).hexdigest(),
            'completed': len(rows), 'averaged_metrics': result['averaged_metrics']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_manifest', type=pathlib.Path, required=True)
    parser.add_argument('--run_dir', type=pathlib.Path, required=True)
    parser.add_argument('--task_timeout', type=float, default=360)
    parser.add_argument('--suite_timeout', type=float, default=1800)
    parser.add_argument('--settle_timeout', type=float, default=180)
    parser.add_argument('--methods', choices=('both', 'mem0', 'hindsight'), default='both')
    parser.add_argument('--only', default='', help='Comma-separated dataset unit names')
    parser.add_argument('--task_order', choices=('smallest-first', 'selection'), default='smallest-first')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'suite.status.json').exists():
        raise FileExistsError('Choose a new run directory; previous runs are never overwritten')
    selection = json.loads((run / 'network_selection.json').read_text(encoding='utf-8'))
    trust_env = selection['selected_trust_env']
    if type(trust_env) is not bool:
        raise RuntimeError('No route passed chat+embedding probes; refusing to start evaluation')
    manifest = prepare(args.source_manifest.resolve(), run, trust_env, 2, 2, args.settle_timeout)
    selected_methods = ('mem0', 'hindsight') if args.methods == 'both' else (args.methods,)
    manifest['units'] = select_units(manifest['units'], args.only, args.task_order)
    manifest['selected_methods'] = list(selected_methods)
    save(run / 'manifest.json', manifest)
    source_hashes = {}
    for relative in ('main.py', 'scripts/run_benchmark_pilot.py', 'scripts/run_hindsight_server.py',
                     'utils/artifact_paths.py', 'utils/conversation_creator.py', 'utils/initialization.py',
                     'utils/agent.py', 'utils/provider_utils.py', 'methods/hindsight/hindsight_adapter.py',
                     'methods/mem0/source/mem0/memory/main.py', 'methods/mem0/source/mem0/llms/openai.py',
                     'methods/mem0/source/mem0/embeddings/openai.py'):
        data = (ROOT / relative).read_bytes()
        target = run / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        source_hashes[relative] = hashlib.sha256(data).hexdigest()
    save(run / 'code_sha256.json', source_hashes)
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUTF8='1',
               OPENAI_API_TRUST_ENV=str(trust_env).lower(), HINDSIGHT_API_URL='http://127.0.0.1:8889')
    if not trust_env:
        env = {k: v for k, v in env.items() if k.upper() not in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY')}
    deadline = time.monotonic() + args.suite_timeout
    state = {'phase': 'running', 'started_at': now(), 'tasks': [], 'task_timeout_seconds': args.task_timeout,
             'suite_timeout_seconds': args.suite_timeout, 'settle_timeout_seconds': args.settle_timeout,
             'task_order': args.task_order, 'selected_tasks': [unit['name'] for unit in manifest['units']],
             'smoke_only': True, 'trust_env': trust_env}
    save(run / 'suite.status.json', state)

    def event(value):
        value['time'] = now()
        print(json.dumps(value, ensure_ascii=False), flush=True)

    stop = lambda: (run / 'stop.request').exists()
    abort = False
    # No concurrent model clients across tasks. Selection may request a fixed order.
    for method in selected_methods:
        for unit in manifest['units']:
            record = {'method': method, 'task': unit['name'], 'phase': 'running', 'started_at': now(),
                      'expected_questions': unit['expected_questions'], 'history_chunks': unit['chunk_count']}
            path = run / f'{method}_{unit["name"]}.status.json'
            if abort or stop() or time.monotonic() >= deadline:
                record.update(phase='not_run', reason='Earlier systemic failure, user stop or suite time budget')
                save(path, record)
                state['tasks'].append(record)
                continue
            task_deadline = min(deadline, time.monotonic() + args.task_timeout)
            server = None
            server_log = None
            server_dir = run / f'hindsight_servers/{unit["name"]}'
            save(path, record)
            event({'event': 'started', **record})
            try:
                if method == 'hindsight':
                    server_dir.mkdir(parents=True, exist_ok=True)
                    server_log = (server_dir / 'launcher.log').open('w', encoding='utf-8')
                    server = subprocess.Popen([r'D:\conda\envs\hindsight\python.exe', '-u', str(ROOT / 'scripts/run_hindsight_server.py'),
                        '--agent_config', manifest['agents'][method], '--state_dir', str(server_dir), '--port', '8889'],
                        cwd=ROOT, env=env, stdout=server_log, stderr=subprocess.STDOUT)
                    record['server_launcher_pid'] = server.pid
                    save(path, record)
                    startup_deadline = min(task_deadline, time.monotonic() + 150)
                    while True:
                        if stop() or time.monotonic() >= startup_deadline:
                            raise TimeoutError('Hindsight startup interrupted or deadline exceeded')
                        if server.poll() is not None:
                            raise RuntimeError('Hindsight launcher failed at startup')
                        status_path = server_dir / 'status.json'
                        if status_path.exists() and json.loads(status_path.read_text(encoding='utf-8')).get('phase') == 'ready':
                            break
                        time.sleep(0.5)
                command = [sys.executable, '-u', str(ROOT / 'main.py'), '--agent_config', manifest['agents'][method],
                           '--dataset_config', unit['dataset_config'], '--artifact_root', str(run / f'artifacts/{unit["name"]}/{method}'),
                           '--max_test_queries_ablation', '2']
                with (run / f'{method}_{unit["name"]}.log').open('w', encoding='utf-8') as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    record['pid'] = child.pid
                    save(path, record)
                    record['exit_code'] = wait_child(child, task_deadline, stop)
                if record['exit_code']:
                    raise RuntimeError('Evaluation child failed; inspect task log')
                record.update(validate_result(run, unit, method), phase='success')
            except Exception as error:
                record.update(phase='interrupted' if isinstance(error, InterruptedError) else 'failed',
                              error_type=type(error).__name__, error=str(error))
                log_path = run / f'{method}_{unit["name"]}.log'
                log_text = log_path.read_text(encoding='utf-8', errors='replace') if log_path.exists() else ''
                if 'content_filter' in log_text:
                    record['failure_category'] = 'content_filter'
                elif 'APIConnectionError' in log_text or 'APITimeoutError' in log_text:
                    record['failure_category'] = 'model_network_failure'
                    abort = True  # Do not repeat a broken network route across remaining datasets.
                elif 'consolidation did not settle' in log_text:
                    record['failure_category'] = 'consolidation_timeout'
                elif '[WinError 123]' in log_text:
                    record['failure_category'] = 'windows_path'
                else:
                    record['failure_category'] = type(error).__name__
                event({'event': 'attention', **record})
            finally:
                if server is not None:
                    try:
                        record['shutdown'] = stop_server(server, server_dir)
                    except Exception as error:
                        record.update(phase='failed', shutdown_error=str(error))
                        abort = True
                        event({'event': 'attention', 'method': method, 'task': unit['name'], 'shutdown_error': str(error)})
                if server_log:
                    server_log.close()
                record['ended_at'] = now()
                save(path, record)
                state['tasks'].append(record)
                save(run / 'suite.status.json', state)
            event({'event': 'finished', 'method': method, 'task': unit['name'], 'phase': record['phase']})
    state.update(phase='success' if all(r['phase'] == 'success' for r in state['tasks']) else 'partial', ended_at=now())
    save(run / 'suite.status.json', state)
    event({'event': 'suite_finished', 'phase': state['phase']})
    return 0 if state['phase'] == 'success' else 1


if __name__ == '__main__':
    raise SystemExit(main())
