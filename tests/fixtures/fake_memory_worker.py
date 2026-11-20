"""Offline stand-in for remem_worker.py / tsm_worker.py speaking the real JSON-lines protocol.

Run by tests with the MemoryData interpreter in a real subprocess. It never
calls a model: it stores documents in its working directory, "retrieves" all
of them and answers "Tea". FAKE_WORKER_KIND selects the REMem or TSM reply
shape; FAKE_GRAPH_DIR emulates TSM's shared Neo4j database (one file per group).
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("memorydata_worker_protocol", ROOT / "methods" / "worker_protocol.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)

KIND = os.environ.get("FAKE_WORKER_KIND", "remem")
STATE = {}


def log(op, **extra):
    with open("calls.jsonl", "a", encoding="utf-8") as stream:
        stream.write(json.dumps({"op": op, **extra}) + "\n")


def group_file():
    return Path(os.environ["FAKE_GRAPH_DIR"]) / f"{STATE['group_id']}.json"


def handle(op, args):
    print("native noise on stdout must not break the protocol")
    if op == "open":
        if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_BASE_URL"):
            raise RuntimeError("credentials were not passed through the environment")
        if KIND == "tsm" and not os.environ.get("NEO4J_PASSWORD"):
            raise RuntimeError("Neo4j password missing")
        STATE.update(args)
        if os.environ.get("FAKE_MP_CHILD"):
            # Like TSM's Timer: an mp.Manager server process that inherits the cwd.
            import multiprocessing
            STATE["manager"] = multiprocessing.Manager()
            STATE["times"] = STATE["manager"].dict()
        log("open", docs=len(args.get("docs") or []))
        return {"opened": True}
    if op in {"index", "reopen"}:
        log(op)
        if op == "reopen":
            STATE["docs"] = json.loads(Path("fake_index.json").read_text(encoding="utf-8"))
        else:
            Path("fake_index.json").write_text(json.dumps(STATE["docs"]), encoding="utf-8")
            if KIND == "remem" and STATE.get("working_dir"):
                Path(STATE["working_dir"]).mkdir(parents=True, exist_ok=True)
                (Path(STATE["working_dir"]) / "graph.pkl").write_bytes(b"fake")
            if KIND == "tsm":
                group_file().write_text(json.dumps(STATE["docs"]), encoding="utf-8")
                Path(STATE["cache_folder"]).mkdir(parents=True, exist_ok=True)
                (Path(STATE["cache_folder"]) / "tree.json").write_text("{}", encoding="utf-8")
        return {"documents": len(STATE["docs"])}
    if op == "reset":
        log("reset", group=STATE["group_id"])
        group_file().unlink(missing_ok=True)
        return {"reset": STATE["group_id"]}
    if op == "query":
        log("query", question_date=args.get("question_date"))
        docs = STATE["docs"]
        if KIND == "remem":
            return {"answer": "Tea", "reasoning": [],
                    "passages": [{"node": f"verbatim-{i}", "type": "verbatim", "text": d["text"]}
                                 for i, d in enumerate(docs)],
                    "source_docs": [[d["doc_id"]] for d in docs]}
        texts = {str(i): d["text"] for i, d in enumerate(docs)}
        evidence = "--- Retrieved Chunks ---\n" + "\n\n".join(texts.values())
        evidence += "\n\n--- Snapshot Summaries ---\n[Snapshot s1] Alice likes Tea."
        reply = {"evidence": evidence, "leaf_positions": list(range(len(docs))), "leaf_texts": texts}
        if args.get("native_answer"):
            reply["answer"] = "Tea"
        return reply
    raise ValueError(f"unknown op {op}")


if __name__ == "__main__":
    if os.environ.get("FAKE_HANG_THREAD"):
        # Like native libraries that leave a non-daemon thread behind.
        import threading
        threading.Thread(target=threading.Event().wait).start()
    sys.exit(protocol.serve(handle))
