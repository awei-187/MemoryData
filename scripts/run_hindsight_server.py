"""Run the native Hindsight memory engine with remote model APIs.

Use D:/conda/envs/hindsight/python.exe; the benchmark stays in memorydata.
All database and log files owned by this launcher go under its state directory.
No model weights are downloaded. The database needs the existing pg0 runtime
unless HINDSIGHT_API_DATABASE_URL is explicitly configured.
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import httpx
    import yaml
    from dotenv import load_dotenv
    from methods.hindsight.api_config import server_environment

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_config", default=str(ROOT / "config/hybrid_hindsight.yaml"))
    parser.add_argument("--state_dir", default=str(ROOT / "results/hindsight_server"))
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    load_dotenv(ROOT / ".env", override=False)
    config = yaml.safe_load(Path(args.agent_config).read_text(encoding="utf-8"))
    env = server_environment(config)
    env["HINDSIGHT_API_PORT"] = str(args.port)
    os.environ.update(env)
    if args.worker:
        source = Path(config.get("hindsight_source_path") or "") / "hindsight-api-slim"
        if (source / "hindsight_api").is_dir():
            sys.path.insert(0, str(source))
        if env.get('OPENAI_API_CONNECT_TIMEOUT_SECONDS') or env.get('OPENAI_API_MAX_RETRIES'):
            from methods.hindsight.api_config import configure_native_embeddings_request_policy
            configure_native_embeddings_request_policy()
        sys.argv = [sys.argv[0]]
        from hindsight_api.main import main as serve
        serve()
        return

    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", args.port))
        except OSError:
            raise RuntimeError(f"Port {args.port} is occupied; the existing service will not be changed") from None
    state = Path(args.state_dir).resolve()
    state.mkdir(parents=True, exist_ok=True)
    stop = state / "stop.request"
    stop.unlink(missing_ok=True)
    status_path = state / "status.json"
    process = database = None

    def status(phase, **extra):
        value = {"phase": phase, "port": args.port, "model": env["HINDSIGHT_API_LLM_MODEL"],
                 "embedding_model": env["HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL"], **extra}
        if env.get('OPENAI_API_CONNECT_TIMEOUT_SECONDS'):
            value['embedding_request_policy'] = {
                'connect_timeout_seconds': float(env['OPENAI_API_CONNECT_TIMEOUT_SECONDS']),
                'sdk_max_retries': int(env.get('OPENAI_API_MAX_RETRIES', '3'))}
        if env.get('HINDSIGHT_API_LLM_CONNECT_TIMEOUT'):
            value['native_llm_connect_timeout_seconds'] = float(env['HINDSIGHT_API_LLM_CONNECT_TIMEOUT'])
        status_path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(value, ensure_ascii=False), flush=True)

    try:
        status("starting_database")
        if not env.get("HINDSIGHT_API_DATABASE_URL"):
            from pg0 import Pg0
            name = "memorydata-hindsight-" + hashlib.sha256(str(state).encode()).hexdigest()[:12]
            database = Pg0(name=name, data_dir=str(state / "database"))
            env["HINDSIGHT_API_DATABASE_URL"] = database.start().uri
        with (state / "server.stdout.log").open("a", encoding="utf-8") as stdout, \
             (state / "server.stderr.log").open("a", encoding="utf-8") as stderr:
            status("starting_hindsight")
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
                       "--agent_config", str(Path(args.agent_config).resolve()), "--port", str(args.port)]
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            headers = {"Authorization": "Bearer " + env["HINDSIGHT_API_KEY"]} if env.get("HINDSIGHT_API_KEY") else {}
            with httpx.Client(trust_env=False, timeout=3, headers=headers) as client:
                for attempt in range(120):
                    if process.poll() is not None:
                        raise RuntimeError("Hindsight exited during startup; inspect the server logs")
                    try:
                        if client.get(f"http://127.0.0.1:{args.port}/health/ready").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    if stop.exists():
                        return
                    if attempt % 10 == 0:
                        print("Waiting for native Hindsight API readiness...", flush=True)
                    time.sleep(1)
                else:
                    raise TimeoutError("Hindsight API did not become ready within startup deadline")
            status("ready", server_pid=process.pid)
            while not stop.exists():
                if process.poll() is not None:
                    raise RuntimeError("Hindsight server exited; inspect the server logs")
                time.sleep(1)
    except Exception:
        status("failed")
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if database is not None:
            database.stop()
        if status_path.exists() and json.loads(status_path.read_text(encoding="utf-8"))["phase"] != "failed":
            status("stopped", database_preserved=True)


if __name__ == "__main__":
    main()
