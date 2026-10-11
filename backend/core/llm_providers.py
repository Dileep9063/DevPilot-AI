"""
LLM provider selection and automatic fallback.

Providers are built lazily from environment variables, so a missing key or
package simply removes that provider instead of crashing the import.

Environment:
    LLM_PROVIDER_ORDER   comma list, default "openrouter,gemini,huggingface"
    OPENROUTER_API_KEY / OPENROUTER_MODEL (default openrouter/free)
    GEMINI_API_KEY     / GEMINI_MODEL     (default gemini-3.8-flash)
    HF_TOKEN           / HF_REPO_ID       (default openai/gpt-oss-120b)
    LLM_RETRIES          retries per provider on transient errors (default 1)
"""

import logging
import os
import random
import time

logger = logging.getLogger(__name__)
MAX_RETRIES = 3


TRANSIENT_MARKERS = (
    "429",
    "rate limit",
    "too many requests",
    "503",
    "service unavailable",
    "unavailable",
    "temporarily unavailable",
    "high demand",
    "timeout",
    "quota",
    "overloaded",
)


def is_transient_provider_error(exc: Exception) -> bool:
    """True when the provider looks temporarily unavailable or rate-limited."""
    message = str(exc).lower()
    return any(marker in message for marker in TRANSIENT_MARKERS)


# ------------------------------------------------------------
# Provider factories (imports are local so missing packages
# only disable that one provider)
# ------------------------------------------------------------

def _make_openrouter():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.environ.get("OPENROUTER_MODEL", "openrouter/free"),
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
    )


def _make_gemini():
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"),
        google_api_key=os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY"),
    )


def _make_huggingface():
    token = os.environ.get("HF_TOKEN")
    if not token:
        return None
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    return ChatHuggingFace(
        llm=HuggingFaceEndpoint(
            repo_id=os.environ.get("HF_REPO_ID", "openai/gpt-oss-120b"),
            task="text-generation",
            huggingfacehub_api_token=token,
            max_new_tokens=1000,
        )
    )


FACTORIES = {
    "openrouter": _make_openrouter,
    "gemini": _make_gemini,
    "huggingface": _make_huggingface,
}

_providers_cache = None


def get_providers(refresh: bool = False):
    """Return [(name, llm), ...] in configured priority order."""
    global _providers_cache

    if _providers_cache is not None and not refresh:
        return _providers_cache

    order = [
        name.strip().lower()
        for name in os.environ.get(
            "LLM_PROVIDER_ORDER", "openrouter,gemini,huggingface"
        ).split(",")
        if name.strip()
    ]

    providers = []
    for name in order:
        factory = FACTORIES.get(name)
        if factory is None:
            logger.warning("Unknown LLM provider ignored: %s", name)
            continue
        try:
            llm = factory()
        except Exception as exc:  # missing package, bad config, ...
            logger.warning("LLM provider unavailable during initialization: %s (%s)", name, type(exc).__name__)
            continue
        if llm is not None:
            providers.append((name, llm))

    _providers_cache = providers
    return providers


class AllProvidersFailed(RuntimeError):
    pass


def invoke_with_fallback(
    messages,
    tools=None,
    providers=None,
    retries=None,
    sleep=time.sleep,
):
    """
    Try each provider in order.

    - Transient errors (429/503/timeouts) are retried on the same provider
      `retries` times with a short backoff, then the next provider is tried.
    - Any other error moves straight to the next provider.
    - If every provider fails, AllProvidersFailed lists each error.
    """
    providers = providers if providers is not None else get_providers()

    if not providers:
        raise AllProvidersFailed(
            "No LLM provider is configured. Set OPENROUTER_API_KEY, "
            "GEMINI_API_KEY or HF_TOKEN."
        )

    if retries is None:
        try:
            retries = int(os.environ.get("LLM_RETRIES", "1"))
        except (TypeError, ValueError):
            retries = 1
    # Bound retries so a bad environment setting cannot create an unbounded delay.
    retries = max(0, min(int(retries), MAX_RETRIES))

    errors = []

    for name, llm in providers:
        runnable = llm.bind_tools(tools) if tools else llm

        for attempt in range(retries + 1):
            started = time.monotonic()
            try:
                result = runnable.invoke(messages)
                latency_ms = round((time.monotonic() - started) * 1000)
                usage = getattr(result, "usage_metadata", None) or {}
                response_meta = getattr(result, "response_metadata", None) or {}
                token_usage = response_meta.get("token_usage") or response_meta.get("usage") or {}
                input_tokens = usage.get("input_tokens", token_usage.get("prompt_tokens"))
                output_tokens = usage.get("output_tokens", token_usage.get("completion_tokens"))
                estimated_cost = None
                input_rate_setting = os.environ.get(f"LLM_COST_{name.upper()}_INPUT_PER_1M")
                output_rate_setting = os.environ.get(f"LLM_COST_{name.upper()}_OUTPUT_PER_1M")
                if (
                    isinstance(input_tokens, (int, float))
                    and isinstance(output_tokens, (int, float))
                    and input_rate_setting is not None
                    and output_rate_setting is not None
                ):
                    try:
                        input_rate = max(0.0, float(input_rate_setting))
                        output_rate = max(0.0, float(output_rate_setting))
                        estimated_cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
                    except (TypeError, ValueError):
                        estimated_cost = None
                logger.info(
                    "LLM usage: provider=%s model=%s attempt=%s latency_ms=%s input_tokens=%s output_tokens=%s estimated_cost_usd=%s",
                    name,
                    response_meta.get("model_name") or response_meta.get("model") or "unknown",
                    attempt + 1,
                    latency_ms,
                    input_tokens if isinstance(input_tokens, (int, float)) else "unknown",
                    output_tokens if isinstance(output_tokens, (int, float)) else "unknown",
                    round(estimated_cost, 8) if estimated_cost is not None else "unknown",
                )
                return result

            except Exception as exc:
                logger.warning(
                    "LLM provider failed: provider=%s attempt=%s error_type=%s transient=%s",
                    name, attempt + 1, type(exc).__name__, is_transient_provider_error(exc),
                )
                if is_transient_provider_error(exc) and attempt < retries:
                    # Exponential backoff with jitter, capped to avoid long waits.
                    delay = min(8.0, 1.5 * (2 ** attempt))
                    sleep(delay + random.uniform(0, delay * 0.25))
                    continue

                # Avoid propagating provider exception text, which can contain
                # request details or credentials, into API/error responses.
                errors.append(f"{name}: {type(exc).__name__}")
                break

    raise AllProvidersFailed(
        "All LLM providers failed — " + " | ".join(errors)
    )
