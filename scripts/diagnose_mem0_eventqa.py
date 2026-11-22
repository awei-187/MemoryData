"""Reproduce one previously rejected fact-extraction request without memory writes."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'methods/mem0/source'))
import yaml
from dotenv import load_dotenv
from utils.provider_utils import resolve_model_config
from benchmark.memoryagentbench.prompts.benchmark_templates import get_template
from mem0.memory.utils import parse_messages
from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.openai import OpenAILLM
from utils.model_api_diagnostics import error_details, response_headers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_number', type=int, default=85)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    load_dotenv(ROOT / '.env', override=False)
    run = ROOT / 'results/full_first_units_20261023'
    path = args.output or run / 'mem0_eventqa_reproduction_20261024.json'
    if path.exists():
        raise FileExistsError('Preserve the first reproduction record; do not overwrite')
    cfg = resolve_model_config(yaml.safe_load((run / 'configs/mem0_eventqa.yaml').read_text(encoding='utf-8')))
    dataset = yaml.safe_load((run / 'configs/eventqa.yaml').read_text(encoding='utf-8'))
    frozen = json.loads((run / 'input/eventqa.json').read_text(encoding='utf-8'))
    if not 1 <= args.chunk_number <= len(frozen['chunks']):
        raise ValueError('Invalid chunk number')
    chunk = frozen['chunks'][args.chunk_number - 1]
    assert isinstance(chunk, str)
    agent, sub = cfg['agent_name'], dataset['sub_dataset']
    runtime_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    memorize_template = get_template(sub, 'memorize', agent)
    formatted = memorize_template.format(context=chunk,
        **({'time_stamp': runtime_timestamp} if '{time_stamp}' in memorize_template else {}))
    memory_messages = [
        {'role': 'system', 'content': get_template(sub, 'system', agent)},
        {'role': 'user', 'content': formatted}]
    fact_prompt = cfg.get('mem0_fact_extraction_prompt') or get_template(sub, 'fact_extraction', agent)
    assert fact_prompt
    messages = [{'role': 'system', 'content': fact_prompt},
                {'role': 'user', 'content': 'Input:\n' + parse_messages(memory_messages)}]
    os.environ['OPENAI_API_TRUST_ENV'] = str(cfg['model_api_trust_env']).lower()
    os.environ['OPENAI_API_CONNECT_TIMEOUT_SECONDS'] = '30'
    # A single diagnostic: SDK does not retry a 400. Network retries also disabled.
    os.environ['OPENAI_API_MAX_RETRIES'] = '0'
    llm = OpenAILLM(BaseLlmConfig(model=cfg['model'], temperature=cfg['temperature'],
        api_key=os.environ[cfg.get('api_key_env', 'OPENAI_API_KEY')],
        openai_base_url=cfg.get('base_url') or os.environ[cfg.get('base_url_env', 'OPENAI_BASE_URL')]))
    request = {'model': llm.config.model, 'messages': messages, 'temperature': llm.config.temperature,
               'max_tokens': llm.config.max_tokens, 'top_p': llm.config.top_p,
               'response_format': {'type': 'json_object'}}
    result = {'time': datetime.now().astimezone().isoformat(), 'case_id': 'mem0_eventqa',
        'chunk_index': args.chunk_number - 1, 'chunk_number': args.chunk_number, 'total_chunks': len(frozen['chunks']),
        'model': llm.config.model, 'temperature': llm.config.temperature,
        'input_sha256': hashlib.sha256((run / 'input/eventqa.json').read_bytes()).hexdigest(),
        'chunk_sha256': hashlib.sha256(chunk.encode('utf-8')).hexdigest(),
        'request_sha256': hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest(),
        'memory_writes': False, 'runtime_timestamp': runtime_timestamp,
        'request_changes': 'same model, prompt templates and original chunk; normal runtime timestamp refreshed; SDK network retries disabled for one-call diagnostic'}
    try:
        raw = llm.client.with_raw_response.chat.completions.create(**request)
        response = raw.parse()
        result.update(status='success', http_status=raw.status_code,
                      response_headers=response_headers(raw.headers),
                      response_sha256=hashlib.sha256((response.choices[0].message.content or '').encode('utf-8')).hexdigest())
    except Exception as error:
        result.update(status='rejected' if getattr(error, 'status_code', None) == 400 else 'error',
                      **error_details(error))
    finally:
        llm.client.close()
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
