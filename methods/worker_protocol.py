"""Worker side of the JSON-lines bridge (standard library only).

Worker scripts run in a method's own interpreter with ``-I`` and load this file
by path, so the MemoryData ``methods`` directory is never put on ``sys.path``
(its package names, such as ``remem``, would shadow the native packages).
"""

from __future__ import annotations

import json
import os
import sys
import traceback


def serve(handler):
    """Answer requests until stdin closes or a ``shutdown`` request arrives.

    The original stdout is kept for protocol replies; afterwards file
    descriptor 1 points at stderr so native ``print`` calls cannot corrupt it.
    """
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    for raw in sys.stdin:
        if not raw.strip():
            continue
        request = json.loads(raw)
        request_id = request.get("id")
        op = request.get("op")
        try:
            if op == "shutdown":
                result = {"stopped": True}
            else:
                result = handler(op, request.get("args") or {})
            reply = {"id": request_id, "ok": True, "result": result}
        except Exception as exc:  # reported to the parent, which raises
            traceback.print_exc(file=sys.stderr)
            reply = {"id": request_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        protocol.write(json.dumps(reply, ensure_ascii=False, default=str) + "\n")
        protocol.flush()
        if op == "shutdown":
            break
