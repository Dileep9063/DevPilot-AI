"""Small chat helper backed by the shared, lazy LLM provider chain.

Provider credentials are checked only when a request is made. Importing Django
views therefore does not require every optional provider to be configured.
"""
from langchain_core.messages import HumanMessage

from core.llm_providers import invoke_with_fallback


def ask_gemini(message):
    """Return a chat response using configured providers and safe fallback."""
    response = invoke_with_fallback([HumanMessage(content=message)])
    content = getattr(response, "content", response)

    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", "")) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)
