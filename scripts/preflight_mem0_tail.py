"""Check the authorized route before resuming the final chunk; no benchmark data."""
from datetime import datetime
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
import httpx
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
assert urlsplit(os.environ['OPENAI_BASE_URL']).hostname == 'api.relayrouter.ai'
assert os.environ.get('OPENAI_MODEL', 'gpt-4o-mini') == 'gpt-4o-mini'
assert os.environ.get('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small') == 'text-embedding-3-small'
records = []
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url=os.environ['OPENAI_BASE_URL'],
                http_client=httpx.Client(trust_env=True), timeout=httpx.Timeout(60, connect=30), max_retries=2)
try:
    for kind in ('chat', 'embedding'):
        row = {'time': datetime.now().astimezone().isoformat(), 'kind': kind, 'trust_env': True}
        try:
            if kind == 'chat':
                result = client.chat.completions.create(model='gpt-4o-mini',
                    messages=[{'role': 'user', 'content': 'Reply with OK.'}], max_tokens=4)
                assert result.choices[0].message.content
            else:
                result = client.embeddings.create(model='text-embedding-3-small', input='Connection test.', dimensions=1536)
                assert len(result.data[0].embedding) == 1536
            row['status'] = 'success'
        except Exception as error:
            row.update(status='failed', error_type=type(error).__name__, http_status=getattr(error, 'status_code', None))
        records.append(row)
        print(json.dumps(row), flush=True)
finally:
    client.close()
path = ROOT / 'results/full_first_units_20261023' / ('tail_preflight_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')
path.write_text(json.dumps(records, indent=2), encoding='utf-8')
raise SystemExit(0 if all(row['status'] == 'success' for row in records) else 1)
