"""Shared provider utilities for multi-model LLM/embedding client creation.

Centralises provider normalisation, env-var resolution, LangChain client
factories, and the ApproximateTokenizer fallback used across method modules.
"""
import logging
import os
import math
from pathlib import Path


logger = logging.getLogger(__name__)
_WARNED_IGNORED_ENV = set()

# Method-specific endpoint variables that the unified profile fills from the
# shared endpoint; the adapters would otherwise have read them.
UNIFIED_BASE_URL_OVERRIDES = {"simplemem_base_url": "SIMPLEMEM_BASE_URL",
                              "lightmem_base_url": "LIGHTMEM_BASE_URL"}


def warn_ignored_env_overrides(names, reason, environment=None):
    """Warn once per variable that is set but has no effect under the unified model profile."""
    env = os.environ if environment is None else environment
    for name in names:
        if env.get(name) and name not in _WARNED_IGNORED_ENV:
            _WARNED_IGNORED_ENV.add(name)
            logger.warning("%s is set but ignored: %s", name, reason)


def model_api_request_options(environment=None):
    """Optional request reliability settings; absent variables preserve SDK defaults."""
    env = os.environ if environment is None else environment
    options = {}
    if env.get('OPENAI_API_CONNECT_TIMEOUT_SECONDS'):
        import httpx
        connect = float(env['OPENAI_API_CONNECT_TIMEOUT_SECONDS'])
        if not math.isfinite(connect) or connect <= 0:
            raise ValueError('OPENAI_API_CONNECT_TIMEOUT_SECONDS must be positive and finite')
        options['timeout'] = httpx.Timeout(600.0, connect=connect)
    if env.get('OPENAI_API_MAX_RETRIES'):
        retries = int(env['OPENAI_API_MAX_RETRIES'])
        if not 0 <= retries <= 10:
            raise ValueError('OPENAI_API_MAX_RETRIES must be between 0 and 10')
        options['max_retries'] = retries
    return options


def model_api_trust_env(value=None):
    """Allow remote model clients to use HTTP(S)_PROXY; keep legacy default."""
    if value is None:
        value = os.environ.get("OPENAI_API_TRUST_ENV", "false")
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized not in {"true", "false", "1", "0"}:
        raise ValueError("OPENAI_API_TRUST_ENV must be true or false")
    return normalized in {"true", "1"}


def resolve_model_config(config, environment=None):
    """Resolve opt-in model settings and native adapter aliases, never API keys.

    Legacy configs without *_env model fields retain their existing behavior.
    This runs before output naming and is also safe to call from AgentWrapper.
    """
    env = os.environ if environment is None else environment
    resolved = dict(config)
    model_variable = config.get("model_env")
    if model_variable:
        model = str(env.get(model_variable) or config.get("model") or "").strip()
        if not model:
            raise ValueError("Answer model is empty")
        resolved["model"] = model
        for alias in ("zep_local_small_model", "zep_local_answer_model",
                      "memtree_aggregate_model", "hindsight_llm_model",
                      "remem_llm_model", "tsm_llm_model"):
            if alias in config:
                resolved[alias] = model
    embedding_variable = config.get("embedding_model_env")
    if embedding_variable:
        embedding = str(env.get(embedding_variable) or config.get("embedding_model") or "").strip()
        if not embedding:
            raise ValueError("Embedding model is empty")
        resolved["embedding_model"] = embedding
        for alias in ("text_embedding", "mem0_embedder_model", "embedder_model",
                      "lightmem_embedding_model", "a_mem_embedding_model",
                      "memtree_embedding_model", "hindsight_embeddings_model",
                      "remem_embedding_model", "tsm_embedding_model"):
            if alias in config:
                resolved[alias] = embedding
    dimension_variable = config.get("embedding_dim_env")
    if dimension_variable:
        dimensions = int(env.get(dimension_variable) or config.get("embedding_dim", 1536))
        if dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive")
        resolved["embedding_dim"] = dimensions
        for alias in ("lightmem_embedding_dims", "hindsight_embeddings_dimensions",
                      "tsm_embedding_dimensions"):
            if alias in config:
                resolved[alias] = dimensions
    trust_variable = config.get("model_api_trust_env_env")
    if trust_variable and env.get(trust_variable):
        resolved["model_api_trust_env"] = model_api_trust_env(env[trust_variable])
    # Materialize endpoint aliases used by adapters that do not resolve *_env.
    if model_variable:
        url = config.get("base_url") or env.get(config.get("base_url_env") or "OPENAI_BASE_URL")
        for alias in ("a_mem_base_url", "simplemem_base_url", "lightmem_base_url",
                      "zep_local_answer_base_url"):
            if alias in config and not config[alias]:
                resolved[alias] = url
                if alias in UNIFIED_BASE_URL_OVERRIDES:
                    warn_ignored_env_overrides(
                        [UNIFIED_BASE_URL_OVERRIDES[alias]],
                        f"model_env is set, so {alias} uses the shared endpoint "
                        "(set the YAML field explicitly to use another one)", env)
        if "memagent" in str(config.get("agent_name", "")).lower():
            resolved["base_url"] = url
    return resolved


class ApproximateTokenizer:
    """Character-level fallback tokenizer for non-GPT providers.

    Used when tiktoken cannot load an encoding (e.g. non-OpenAI model names).
    Token counts are approximate (1 char ≈ 1 token) but sufficient for
    context-length guards that only need rough estimates.
    """

    def encode(self, text, disallowed_special=()):
        return list(text or "")

    def decode(self, tokens):
        return "".join(tokens)


class HuggingFaceTokenizerAdapter:
    """Thin adapter that makes a Hugging Face tokenizer look like tiktoken."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def encode(self, text, disallowed_special=(), add_special_tokens=False):
        del disallowed_special
        return self._tokenizer.encode(text or "", add_special_tokens=add_special_tokens)

    def decode(self, tokens):
        return self._tokenizer.decode(tokens, skip_special_tokens=True)


def load_local_hf_tokenizer(candidate):
    """Load a local Hugging Face tokenizer when a matching model path exists."""
    normalized_candidate = str(candidate or "").strip()
    if not normalized_candidate:
        return None

    resolved_candidates = []
    candidate_path = Path(normalized_candidate).expanduser()
    if candidate_path.exists():
        resolved_candidates.append(candidate_path)

    configured_models_root = os.getenv("LOCAL_MODELS_ROOT")
    candidate_roots = []
    if configured_models_root:
        candidate_roots.append(Path(configured_models_root).expanduser())

    repo_models_root = Path(__file__).resolve().parents[1] / "models"
    candidate_roots.append(repo_models_root)
    candidate_roots.append(Path("/data/models"))

    for candidate_root in candidate_roots:
        models_root_candidate = candidate_root / normalized_candidate
        if models_root_candidate.exists():
            resolved_candidates.append(models_root_candidate)

    tried_paths = set()
    for resolved_path in resolved_candidates:
        resolved_str = str(resolved_path)
        if resolved_str in tried_paths:
            continue
        tried_paths.add(resolved_str)
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                resolved_str,
                local_files_only=True,
                trust_remote_code=True,
            )
            return HuggingFaceTokenizerAdapter(tokenizer)
        except Exception:
            continue

    return None

_PROVIDER_ALIASES = {
    "azure": "azure_openai",
    "azure_openai": "azure_openai",
    "openai": "openai",
    "openai-compatible": "openai_compatible",
    "openai_compatible": "openai_compatible",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "google": "gemini",
    "google_genai": "gemini",
    "gemini": "gemini",
    "deepseek": "openai_compatible",
}


def normalize_provider(provider_name):
    """Normalize provider name aliases to a stable internal identifier."""
    normalized = (provider_name or "").strip().lower()
    return _PROVIDER_ALIASES.get(normalized, normalized)


def resolve_env_value(explicit_env_name=None, fallback_env_names=None):
    """Resolve a config value from an explicit or fallback environment variable."""
    if explicit_env_name:
        return os.environ.get(explicit_env_name)
    for env_name in fallback_env_names or []:
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def resolve_base_url(base_url=None, base_url_env=None):
    """Resolve an OpenAI-compatible base URL from config or environment."""
    if base_url:
        return base_url
    if base_url_env:
        return os.environ.get(base_url_env)
    return os.environ.get("OPENAI_BASE_URL")


def use_azure_openai(provider, azure_endpoint=None, base_url=None, base_url_env=None):
    """Return True when the resolved configuration should use Azure OpenAI."""
    return provider == "azure_openai" or (
        provider == "openai"
        and (azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"))
        and not base_url
        and not base_url_env
    )


def create_chat_llm(model, temperature=0.7, max_tokens=None, provider="openai",
                    base_url=None, base_url_env=None, api_key_env=None,
                    azure_endpoint=None, azure_api_version=None):
    """Create a LangChain chat LLM for the given provider.

    Supports: openai, azure_openai, openai_compatible, gemini.
    max_tokens=None leaves the parameter unset (uses provider default).
    """
    provider = normalize_provider(provider)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = {
            "model": model,
            "temperature": temperature,
            "google_api_key": resolve_env_value(api_key_env, ["GOOGLE_API_KEY"]),
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatGoogleGenerativeAI(**kwargs)

    if use_azure_openai(provider, azure_endpoint=azure_endpoint,
                        base_url=base_url, base_url_env=base_url_env):
        from langchain_openai import AzureChatOpenAI
        kwargs = {
            "azure_deployment": model,
            "api_version": azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION"),
            "azure_endpoint": azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"),
            "api_key": resolve_env_value(api_key_env, ["AZURE_OPENAI_API_KEY"]),
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return AzureChatOpenAI(**kwargs)

    resolved_base_url = resolve_base_url(base_url, base_url_env)
    api_key = resolve_env_value(api_key_env, ["OPENAI_API_KEY"])
    if provider == "openai_compatible" and not resolved_base_url:
        raise RuntimeError("OpenAI-compatible models require 'base_url' or 'base_url_env'.")

    from langchain_openai import ChatOpenAI
    kwargs = {"model": model, "temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    if api_key:
        kwargs["api_key"] = api_key
    return ChatOpenAI(**kwargs)


def create_embedding_model(model=None, provider="openai", base_url=None, base_url_env=None,
                           api_key_env=None, azure_endpoint=None, azure_api_version=None,
                           azure_deployment=None):
    """Create a LangChain embedding model for the given provider.

    Supports: openai, azure_openai, openai_compatible.
    azure_deployment defaults to the model name when using Azure.
    """
    provider = normalize_provider(provider)
    embedding_model_name = model or "text-embedding-3-small"

    if use_azure_openai(provider, azure_endpoint=azure_endpoint,
                        base_url=base_url, base_url_env=base_url_env):
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            model=embedding_model_name,
            azure_deployment=azure_deployment or embedding_model_name,
            api_version=azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_key=resolve_env_value(api_key_env, ["AZURE_OPENAI_API_KEY"]),
        )

    resolved_base_url = resolve_base_url(base_url, base_url_env)
    api_key = resolve_env_value(api_key_env, ["OPENAI_API_KEY"])
    if provider == "openai_compatible" and not resolved_base_url:
        raise RuntimeError("OpenAI-compatible embedding models require 'base_url' or 'base_url_env'.")

    from langchain_openai import OpenAIEmbeddings
    kwargs = {"model": embedding_model_name}
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    if api_key:
        kwargs["api_key"] = api_key
    return OpenAIEmbeddings(**kwargs)
