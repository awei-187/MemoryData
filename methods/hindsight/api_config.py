"""Resolve API-backed Hindsight settings without persisting credentials."""
import os
from utils.provider_utils import resolve_model_config, model_api_request_options


def configure_native_embeddings_request_policy():
    """Opt-in bootstrap for the native embedder's SDK clients, without editing upstream.

    This native version exposes no embedding connect-timeout setting. Its
    loop-local SDK accessor is preserved; only public SDK timeout/retry fields
    are set before a request. Model, text, batches and vectors are untouched.
    """
    options = model_api_request_options()
    if not options:
        return
    from hindsight_api.engine.embeddings import OpenAIEmbeddings
    if getattr(OpenAIEmbeddings, '_memorydata_request_policy_applied', False):
        return
    original = OpenAIEmbeddings._loop_client

    def configured_client(self):
        client = original(self)
        for name, value in options.items():
            setattr(client, name, value)
        return client

    OpenAIEmbeddings._loop_client = configured_client
    OpenAIEmbeddings._memorydata_request_policy_applied = True


def resolve_answer_config(config):
    resolved = resolve_model_config(config)
    if not str(resolved.get("model") or "").strip():
        raise ValueError("Hindsight answer model is empty")
    return resolved


def server_environment(config, environment=None):
    """Map the copied OPENAI_* settings to native Hindsight API variables.

    Only pass this result to the server process; it contains credentials and
    must never be serialized into benchmark configs, results, or logs.
    """
    env = dict(os.environ if environment is None else environment)
    config = resolve_model_config(config, env)
    url = env.get("HINDSIGHT_API_LLM_BASE_URL") or env.get("OPENAI_BASE_URL")
    key = env.get("HINDSIGHT_API_LLM_API_KEY") or env.get("OPENAI_API_KEY")
    if not url or not key:
        raise ValueError("Set OPENAI_BASE_URL and OPENAI_API_KEY in the project .env before starting Hindsight")
    provider = env.get("HINDSIGHT_API_LLM_PROVIDER", "openai")
    if provider != "openai":
        raise ValueError("This API profile requires HINDSIGHT_API_LLM_PROVIDER=openai")
    env.update({
        "HINDSIGHT_API_LLM_PROVIDER": "openai",
        "HINDSIGHT_API_LLM_MODEL": env.get("HINDSIGHT_API_LLM_MODEL") or config.get("hindsight_llm_model", "gpt-4o-mini"),
        "HINDSIGHT_API_LLM_BASE_URL": url.rstrip("/"),
        "HINDSIGHT_API_LLM_API_KEY": key,
        "HINDSIGHT_API_EMBEDDINGS_PROVIDER": "openai",
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL": env.get("HINDSIGHT_API_EMBEDDINGS_OPENAI_MODEL") or config.get("hindsight_embeddings_model", "text-embedding-3-small"),
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL": env.get("HINDSIGHT_API_EMBEDDINGS_OPENAI_BASE_URL") or url.rstrip("/"),
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY": env.get("HINDSIGHT_API_EMBEDDINGS_OPENAI_API_KEY") or key,
        "HINDSIGHT_API_EMBEDDINGS_OPENAI_DIMENSIONS": env.get("HINDSIGHT_API_EMBEDDINGS_OPENAI_DIMENSIONS") or str(config.get("hindsight_embeddings_dimensions", 1536)),
        # The existing E:/AML Hindsight run uses RRF instead of a local neural reranker.
        "HINDSIGHT_API_RERANKER_PROVIDER": config.get("hindsight_reranker_provider", "rrf"),
        "HINDSIGHT_API_HOST": "127.0.0.1",
        "HINDSIGHT_API_WORKERS": "1",
        "PYTHONIOENCODING": "utf-8",
    })
    # The native LLM provider owns its retry loop and has an official connect
    # timeout setting. Map only an explicitly requested transport override;
    # preserve native request deadlines/retries and an explicit native value.
    request_options = model_api_request_options(env)
    if 'timeout' in request_options:
        env.setdefault('HINDSIGHT_API_LLM_CONNECT_TIMEOUT', str(request_options['timeout'].connect))
    return env
