"""Retain/recall integration against Hindsight's public v0.10.1 REST API.

Contract checked against E:/AML/hindsight at 1e427025b4d4c01e8ad6385dd04e02102886a5c4.
No server code or generated SDK is imported into MemoryData.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from utils.repo_paths import legacy_variants, portable

PROTOCOL = "memorydata_hindsight_retain_recall_v1"
STATE_FILE = "hindsight_state.json"
LEGACY_IDENTITY_KEYS = ("retrieve_num", "hindsight_recall_max_tokens", "hindsight_recall_budget",
                        "hindsight_recall_types", "hindsight_settle_timeout_seconds")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def normalize_query_timestamp(value):
    """Translate benchmark dates to the native ISO contract without inventing a timezone."""
    if value is None or value == '':
        return None
    if not isinstance(value, str):
        raise ValueError('Hindsight query_timestamp must be a string')
    value = value.strip()
    try:
        datetime.fromisoformat(value.replace('Z', '+00:00'))
        return value
    except ValueError:
        pass
    # MemBench question times look like "'2024-10-04 08:49' Friday".
    membench = re.fullmatch(r"'(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})'(?:\s+[A-Za-z]+)?", value)
    if membench:
        return datetime.strptime(" ".join(membench.groups()), '%Y-%m-%d %H:%M').isoformat()
    # MemoryAgentBench LongMemEval dates include an English weekday. Strip it
    # explicitly rather than depending on the Windows process locale (%a).
    compact = re.sub(r'(?<=\d)\s+\([A-Za-z]{3}\)(?=\s+\d{2}:\d{2}$)', '', value)
    try:
        return datetime.strptime(compact, '%Y/%m/%d %H:%M').isoformat()
    except ValueError as exc:
        raise ValueError('Unsupported Hindsight query_timestamp; expected ISO, YYYY/MM/DD (Day) HH:MM '
                         "or 'YYYY-MM-DD HH:MM' Day") from exc


_MONTHS = {name: index for index, name in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
# Time markers that benchmark loaders write into chunk text. Each match starts
# at its header, so text before it means the chunk began mid-session.
_TIME_MARKERS = (
    # LoCoMo session header: "Session 3 (1:56 pm on 8 May, 2023)"
    (re.compile(r"Session \d+ \((\d{1,2}):(\d{2})\s*([ap]m) on (\d{1,2}) ([A-Za-z]+),? (\d{4})\)", re.I), "locomo"),
    # MemBench turn header: "Session 1 - Turn 1 [time: '2024-10-01 08:00' Tuesday; ..."
    (re.compile(r"(?:Session \d+ - Turn \d+ )?\[time: '(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2})'"), "membench"),
    # MemoryAgentBench LongMemEval: "Chat Time: 2022/11/17 (Thu) 12:04"
    (re.compile(r"Chat Time: (\d{4})/(\d{2})/(\d{2}) \([A-Za-z]{3}\) (\d{2}):(\d{2})"), "longmemeval"),
)


def _marker_time(kind, groups):
    if kind == "locomo":
        hour, minute, meridiem, day, month, year = groups
        month_number = _MONTHS.get(month.lower())
        if month_number is None:
            return None
        hour = int(hour) % 12 + (12 if meridiem.lower() == "pm" else 0)
        return datetime(int(year), month_number, int(day), hour, int(minute))
    return datetime(*(int(value) for value in groups))


def find_time_markers(text):
    """[(start, iso_time)] for every recognized benchmark time marker in text, in order."""
    markers = []
    for pattern, kind in _TIME_MARKERS:
        for match in pattern.finditer(text or ""):
            try:
                moment = _marker_time(kind, match.groups())
            except ValueError:
                moment = None
            if moment is not None:
                markers.append((match.start(), moment.isoformat()))
    return sorted(markers)


class ChunkTimeTracker:
    """Timestamps for history chunks from in-text session/turn time markers.

    A chunk takes the time of its first marker, or the time carried over from
    earlier chunks when it starts mid-session (content before its first
    marker) or has no marker. Chunks must be fed in history order.
    """

    def __init__(self):
        self.last = None

    def timestamp_for(self, text):
        markers = find_time_markers(text)
        if not markers:
            return self.last
        start, first = markers[0]
        starts_mid_session = bool(re.sub(r"[\s\[\]'\",]", "", text[:start]))
        timestamp = self.last if (starts_mid_session and self.last) else first
        self.last = markers[-1][1]
        return timestamp


def render_recall_entry(entry):
    """Memory text with the native dates Hindsight returned, for the answer prompt."""
    dates = []
    start, end = entry.get("occurred_start"), entry.get("occurred_end")
    if start:
        dates.append(f"occurred: {start}" + (f" to {end}" if end and end != start else ""))
    if entry.get("mentioned_at"):
        dates.append(f"mentioned: {entry['mentioned_at']}")
    return f"[{'; '.join(dates)}] {entry['text']}" if dates else entry["text"]


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class HindsightAdapter:
    """Own one isolated remote bank and a local, credential-free state manifest."""

    def __init__(self, agent_config, dataset_config, state_dir, *, transport=None):
        self.path = Path(state_dir) / STATE_FILE
        self.base_url = str(os.environ.get(agent_config.get("hindsight_base_url_env") or "HINDSIGHT_API_URL")
                            or agent_config.get("hindsight_base_url") or "http://127.0.0.1:8888").rstrip("/")
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("hindsight_base_url must be an HTTP(S) service root without credentials, query or fragment")
        self.api_key = os.environ.get(agent_config.get("hindsight_api_key_env") or "HINDSIGHT_API_KEY")
        self.timeout = float(agent_config.get("hindsight_timeout_seconds", 300))
        self.top_k = int(agent_config.get("retrieve_num", 20))
        self.max_tokens = int(agent_config.get("hindsight_recall_max_tokens", 4096))
        self.budget = agent_config.get("hindsight_recall_budget", "mid")
        self.types = agent_config.get("hindsight_recall_types", ["world", "experience", "observation"])
        self.bank_config = dict(agent_config.get("hindsight_bank_config") or {})
        self.wait_for_consolidation = bool(agent_config.get("hindsight_wait_for_consolidation", True))
        self.settle_timeout = float(agent_config.get("hindsight_settle_timeout_seconds", 300))
        # Off by default so earlier configs and saved banks keep their meaning.
        self.infer_timestamps = bool(agent_config.get("hindsight_infer_timestamps", False))
        self.recall_include_dates = bool(agent_config.get("hindsight_recall_include_dates", False))
        self.transport = transport
        if self.budget not in {"low", "mid", "high"} or min(self.top_k, self.max_tokens, self.timeout, self.settle_timeout) <= 0:
            raise ValueError("Hindsight needs positive retrieve_num/token budget/timeout and a low/mid/high recall budget")
        if not isinstance(self.types, list) or not self.types or any(t not in {"world", "experience", "observation"} for t in self.types):
            raise ValueError("hindsight_recall_types must be a nonempty list of world, experience, observation")
        identity_state_dir = str(self.path.parent.resolve())
        origin = agent_config.get('hindsight_snapshot_origin')
        if origin:
            # Explicitly fork a stopped database/local state snapshot. Keep the
            # original identity contract; all other config/history fields still
            # have to match. The original manifest is never modified.
            source_dir = Path(origin['state_dir']).resolve()
            source_manifest = source_dir / STATE_FILE
            if hashlib.sha256(source_manifest.read_bytes()).hexdigest() != origin['state_sha256']:
                raise ValueError('Hindsight snapshot origin manifest changed')
            identity_state_dir = str(source_dir)
        dataset = {k: v for k, v in dataset_config.items() if k != "max_test_queries"}
        history = agent_config.get("_hindsight_history_fingerprint")
        base = {"protocol": PROTOCOL, "base_url": self.base_url, "bank_config": self.bank_config, "history": history}

        # The identity covers only what determines the stored memory: endpoint,
        # bank config, data, history and whether ingestion waited for
        # consolidation. Recall parameters act at query time and the settle
        # timeout is only a wait budget, so changing them reuses the bank.
        def identity(dataset_value, state_dir_value):
            value = {**base, "identity_version": 2, "dataset": dataset_value,
                     "state_dir": state_dir_value, "location": "repo-relative",
                     "wait_for_consolidation": self.wait_for_consolidation}
            if self.infer_timestamps:
                # Retain timestamps then come from in-text markers, so banks
                # built without them must not be reused.
                value["retain_timestamps"] = "text-markers-v1"
            return fingerprint(value)

        # Paths inside the repository are fingerprinted relative to it, so
        # moving the whole checkout keeps saved states valid. Paths outside it
        # stay absolute; copying a state elsewhere still requires an explicit
        # hindsight_snapshot_origin.
        self.identity = identity(portable(dataset), portable(identity_state_dir))
        self.state = self.read_state(state_dir)
        if self.state:
            self._check_owned_state(self.state)
            stored = self.state.get("identity")
            # Older identities also hashed recall/settle values and are reused
            # as is, never rewritten, so snapshot origins keep their SHA-256.
            legacy = set() if self.infer_timestamps else self._legacy_identities(
                agent_config, base, dataset, identity_state_dir)
            if stored != self.identity and stored not in legacy:
                raise ValueError("Hindsight saved configuration/history differs; use --force or a new --artifact_root")
        else:
            # UUID ownership avoids adopting/deleting an unrelated pre-existing bank.
            self.state = {"protocol": PROTOCOL, "identity": self.identity, "base_url": self.base_url,
                          "bank_id": "md-hs-" + uuid.uuid4().hex, "ready": False, "documents": {}}
        self.prepared = False
        self.chunk_index = 0

    def _legacy_identities(self, agent_config, base, dataset, state_dir):
        """Identities written before identity_version 2, which also hashed query-time settings.

        They match with the current recall/settle values, or with values declared
        in hindsight_legacy_identity_params (a dict or list of dicts using the
        ordinary config keys) when a saved bank is reused under new ones.
        """
        declared = agent_config.get("hindsight_legacy_identity_params") or []
        if isinstance(declared, dict):
            declared = [declared]
        if not isinstance(declared, list) or any(not isinstance(item, dict) for item in declared):
            raise ValueError("hindsight_legacy_identity_params must be a dict or a list of dicts")
        candidates = []
        for item in [{}] + declared:
            unknown = set(item) - set(LEGACY_IDENTITY_KEYS)
            if unknown:
                raise ValueError(f"Unsupported hindsight_legacy_identity_params keys: {sorted(unknown)}")
            settings = {key: item.get(key, current) for key, current in zip(LEGACY_IDENTITY_KEYS, (
                self.top_k, self.max_tokens, self.budget, self.types, self.settle_timeout))}
            candidates.append({
                "recall": [int(settings["retrieve_num"]), int(settings["hindsight_recall_max_tokens"]),
                           settings["hindsight_recall_budget"], list(settings["hindsight_recall_types"])],
                "settle": [self.wait_for_consolidation, float(settings["hindsight_settle_timeout_seconds"])],
            })
        identities = set()
        for params in candidates:
            # Repository-relative identities (2026-11-19) and absolute-path ones before them.
            identities.add(fingerprint({**base, **params, "dataset": portable(dataset),
                                        "state_dir": portable(state_dir), "location": "repo-relative"}))
            for dataset_value, state_dir_value in legacy_variants([dataset, state_dir]):
                identities.add(fingerprint({**base, **params, "dataset": dataset_value, "state_dir": state_dir_value}))
        return identities

    @staticmethod
    def read_state(state_dir):
        path = Path(state_dir) / STATE_FILE
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    @staticmethod
    def _check_owned_state(state):
        if state.get("protocol") != PROTOCOL or not re.fullmatch(r"md-hs-[a-f0-9]{32}", str(state.get("bank_id", ""))):
            raise ValueError("Refusing to operate on a bank not owned by the MemoryData Hindsight adapter")

    def _request(self, method, suffix="", payload=None, *, missing_ok=False):
        headers = {"Authorization": "Bearer " + self.api_key} if self.api_key else {}
        # Bypass proxy variables only for this API client, which also permits LAN servers.
        with httpx.Client(timeout=self.timeout, headers=headers, trust_env=False, transport=self.transport) as client:
            response = client.request(method, f"{self.base_url}/v1/default/banks/{self.state['bank_id']}{suffix}", json=payload)
        if missing_ok and response.status_code == 404:
            return None
        response.raise_for_status()
        if response.status_code == 204:
            return {}
        value = response.json()
        if not isinstance(value, dict):
            raise ValueError("Hindsight returned a non-object JSON response")
        return value

    def prepare(self):
        if self.prepared:
            return
        if self.state.get("ready"):
            self.load()
            return
        # Persist ownership before the first write. A failed ingestion is rebuilt
        # in its entirety on the next attempt; synchronous retain is never retried.
        existed = self.path.exists()
        _save(self.path, self.state)
        if existed:
            self._request("DELETE", missing_ok=True)
        self.state["documents"] = {}
        self._request("PUT", payload={"name": "MemoryData isolated evaluation bank", **self.bank_config})
        self.prepared = True

    @staticmethod
    def document_id(index, content):
        return f"md-chunk-{index:08d}-{fingerprint(content)[:16]}"

    @staticmethod
    def _normalize_source_ids(source_ids):
        return list(dict.fromkeys(str(x) for x in source_ids if x))

    @classmethod
    def expected_documents(cls, items):
        """Document IDs/source mappings add_chunk records for (content, source_ids) items, in order."""
        documents = {}
        for content, source_ids in items:
            if content.strip():
                documents[cls.document_id(len(documents), content)] = cls._normalize_source_ids(source_ids)
        return documents

    def awaits_consolidation_only(self, expected_documents):
        """True when every chunk was acknowledged but readiness was never confirmed (e.g. a settle timeout)."""
        return (not self.state.get("ready") and bool(expected_documents)
                and self.state.get("documents") == expected_documents)

    def add_chunk(self, content, *, source_ids=(), timestamp=None):
        if not content.strip():
            return
        self.prepare()
        if self.state.get("ready"):
            raise RuntimeError("Cannot append history to a completed Hindsight bank; rebuild with --force")
        document_id = self.document_id(self.chunk_index, content)
        self.chunk_index += 1
        source_ids = self._normalize_source_ids(source_ids)
        item = {"content": content, "document_id": document_id, "update_mode": "replace",
                "timestamp": timestamp or "unset", "metadata": {"memorydata_source_ids": json.dumps(source_ids)}}
        response = self._request("POST", "/memories", {"items": [item], "async": False})
        if response.get("success") is not True or response.get("async") is not False or response.get("items_count") != 1:
            raise RuntimeError("Hindsight retain did not confirm synchronous completion of one item")
        self.state["documents"][document_id] = source_ids
        _save(self.path, self.state)

    def settle(self):
        """Make the history retained so far answerable without completing the bank.

        Used at interleaved question points (HaluMem): waits for consolidation
        like save(), but retain may continue afterwards. retrieve() accepts the
        bank until the next retain, then requires another settle().
        """
        if self.state.get("ready"):
            return
        self.prepare()
        if self.wait_for_consolidation:
            self.state["last_settled_stats"] = self._wait_for_consolidation()
        self.state["settled_documents"] = len(self.state["documents"])
        _save(self.path, self.state)

    def _answerable(self):
        settled = self.state.get("settled_documents")
        return bool(self.state.get("ready")) or (settled is not None and settled == len(self.state["documents"]))

    def save(self):
        self.prepare()
        if self.wait_for_consolidation and not self.state.get("ready"):
            self.state["settled_stats"] = self._wait_for_consolidation()
        self.state["ready"] = True
        _save(self.path, self.state)

    def _wait_for_consolidation(self):
        """Poll native stats until no background work is pending; return the final stats."""
        deadline = time.monotonic() + self.settle_timeout
        while True:
            stats = self._request("GET", "/stats?refresh=true")
            self.state['last_consolidation_stats'] = stats
            if not {"pending_operations", "failed_operations", "pending_consolidation", "operations_by_status"} <= stats.keys():
                raise ValueError("Hindsight stats lack consolidation readiness fields required by this adapter")
            if stats["failed_operations"]:
                _save(self.path, self.state)
                raise RuntimeError("Hindsight reports failed background operations; ingestion is not complete")
            busy = (stats["pending_operations"] or stats["pending_consolidation"]
                    or stats["operations_by_status"].get("pending", 0)
                    or stats["operations_by_status"].get("processing", 0))
            if not busy:
                return stats
            if time.monotonic() >= deadline:
                _save(self.path, self.state)
                raise TimeoutError(
                    f"Hindsight consolidation did not settle within {self.settle_timeout:g}s; "
                    f"pending_operations={stats['pending_operations']}, "
                    f"pending_consolidation={stats['pending_consolidation']}. "
                    "All acknowledged chunks are kept; rerun the same command (a larger "
                    "hindsight_settle_timeout_seconds is allowed) to keep waiting without retaining again."
                )
            print("Waiting for Hindsight background memory consolidation...", flush=True)
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))

    def resume_completed_ingestion(self, expected_documents):
        """Resume only acknowledged full ingestion; never resend retain or mark ready directly."""
        if not expected_documents or self.state.get('documents') != expected_documents:
            raise ValueError('Cannot resume: acknowledged document IDs/source mappings do not match complete history')
        stats = self._request('GET', '/stats?refresh=true')
        if stats.get('total_documents') != len(expected_documents) or stats.get('failed_operations') != 0:
            raise ValueError('Cannot resume: native document count differs or background operations failed')
        self.prepared = True
        self.save()  # The ordinary native readiness checks still decide completion.
        return stats

    def resume_interleaved_prefix(self, expected_documents):
        """Reopen a bank whose acknowledged documents are exactly a history prefix (interleaved runs).

        Used after a settle() timeout or a failed question at an interleaved
        question point: retain continues after the prefix instead of deleting
        the bank. The caller settles again before answering; nothing is resent.
        """
        if self.state.get("ready") or not expected_documents or self.state.get("documents") != expected_documents:
            raise ValueError("Cannot resume: acknowledged document IDs/source mappings do not match this history prefix")
        stats = self._request("GET", "/stats?refresh=true")
        if stats.get("total_documents") != len(expected_documents) or stats.get("failed_operations") != 0:
            raise ValueError("Cannot resume: native document count differs or background operations failed")
        self.prepared = True
        self.chunk_index = len(expected_documents)
        return stats

    def load(self):
        if not self.state.get("ready"):
            raise RuntimeError("Hindsight history is incomplete; rebuild before answering")
        try:
            bank = self._request("GET", "/profile", missing_ok=True)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 410:
                raise
            # Newer native servers retire /profile with 410 and expose the
            # bank through /config. Retain compatibility with older servers.
            bank = self._request("GET", "/config", missing_ok=True)
        if bank is None:
            raise RuntimeError("Saved Hindsight bank no longer exists; rerun with --force")
        self.prepared = True

    def retrieve(self, query, *, query_timestamp=None):
        if not self._answerable():
            raise RuntimeError("Save the complete Hindsight history (or settle() at an interleaved "
                               "question point, with no retain since) before querying")
        query_timestamp = normalize_query_timestamp(query_timestamp)
        self.prepare()
        payload = {"query": query, "budget": self.budget, "max_tokens": self.max_tokens,
                   "types": self.types, "include": {"entities": None, "source_facts": {"max_tokens": self.max_tokens}}}
        if query_timestamp:
            payload["query_timestamp"] = query_timestamp
        response = self._request("POST", "/memories/recall", payload)
        results = response.get("results")
        if not isinstance(results, list) or any(not isinstance(r, dict) or not isinstance(r.get("text"), str) for r in results):
            raise ValueError("Hindsight recall response must contain a results list with text fields")
        source_facts = response.get("source_facts") or {}

        def sources(item, seen):
            ids = list(self.state["documents"].get(item.get("document_id"), []))
            for fact_id in item.get("source_fact_ids") or []:
                if fact_id not in seen and isinstance(source_facts.get(fact_id), dict):
                    ids.extend(sources(source_facts[fact_id], seen | {fact_id}))
            return list(dict.fromkeys(ids))

        # The API budgets by tokens, not by top_k. Apply MemoryData's k cap
        # after native ranking without sending an unsupported top_k parameter.
        entries = [{"text": r["text"], "source_ids": sources(r, set()), "id": r.get("id"),
                    "document_id": r.get("document_id"), "occurred_start": r.get("occurred_start"),
                    "occurred_end": r.get("occurred_end"), "mentioned_at": r.get("mentioned_at")}
                   for r in results if r["text"].strip()][:self.top_k]
        for entry in entries:
            entry["display_text"] = render_recall_entry(entry) if self.recall_include_dates else entry["text"]
        return entries

    @classmethod
    def reset_saved(cls, agent_config, state_dir):
        state = cls.read_state(state_dir)
        if not state:
            return
        cls._check_owned_state(state)
        # Reset only the UUID bank recorded in this context's local manifest.
        adapter = cls.__new__(cls)
        adapter.base_url = str(os.environ.get(agent_config.get("hindsight_base_url_env") or "HINDSIGHT_API_URL")
                               or agent_config.get("hindsight_base_url") or "http://127.0.0.1:8888").rstrip("/")
        if adapter.base_url != state.get("base_url"):
            raise ValueError("Hindsight endpoint changed; use the old endpoint to reset or select a new --artifact_root")
        adapter.api_key = os.environ.get(agent_config.get("hindsight_api_key_env") or "HINDSIGHT_API_KEY")
        adapter.timeout = float(agent_config.get("hindsight_timeout_seconds", 300))
        adapter.transport = None
        adapter.state = state
        adapter._request("DELETE", missing_ok=True)
