"""Small allowlisted API error records, without credentials or input text."""
from datetime import datetime
import hashlib
import json
import os
import re
from pathlib import Path
import uuid

SAFE_HEADERS = {'x-request-id', 'request-id', 'apim-request-id', 'x-ms-request-id',
                'x-ms-region', 'x-oneapi-request-id', 'x-newapi-request-id', 'server',
                'x-provider', 'x-upstream-provider', 'x-model', 'x-served-by'}


def response_headers(headers):
    return {key.lower(): value for key, value in headers.items() if key.lower() in SAFE_HEADERS}


def error_details(error):
    response = getattr(error, 'response', None)
    body = getattr(error, 'body', None)
    if isinstance(body, dict) and isinstance(body.get('error'), dict):
        body = body['error']
    result = {'error_type': type(error).__name__, 'http_status': getattr(error, 'status_code', None),
              'request_id': getattr(error, 'request_id', None),
              'response_headers': response_headers(response.headers) if response is not None else {}}
    if isinstance(body, dict):
        result['error_fields'] = {key: body[key] for key in ('code', 'type', 'param') if key in body}
        message_id = re.search(r'\brequest id:\s*([A-Za-z0-9_-]+)', str(body.get('message', '')), re.IGNORECASE)
        if message_id:
            result['message_request_id'] = message_id.group(1)
        inner = body.get('innererror', body.get('inner_error'))
        if isinstance(inner, dict):
            result['inner_error_code'] = inner.get('code')
            filters = inner.get('content_filter_result', inner.get('content_filter_results'))
            if isinstance(filters, dict):
                result['filter_results'] = {category: {key: item[key] for key in ('filtered', 'severity', 'detected') if key in item}
                                            for category, item in filters.items() if isinstance(item, dict)}
    return result


def record_api_error(error, request):
    directory = os.environ.get('MEMORYDATA_API_DIAGNOSTICS_DIR')
    if not directory:
        return
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    result = {'time': datetime.now().astimezone().isoformat(), 'model': request.get('model'),
              'request_sha256': hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
              **error_details(error)}
    (path / (uuid.uuid4().hex + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
