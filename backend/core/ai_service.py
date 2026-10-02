import os

from django.conf import settings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI


gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)

hf_llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        task="text-generation",
        huggingfacehub_api_token=os.environ["HF_TOKEN"],
        max_new_tokens=1000,
    )
)


def _is_transient_provider_error(exc):
    message = str(exc).lower()

    markers = (
        "429",
        "rate limit",
        "too many requests",
        "503",
        "service unavailable",
        "unavailable",
        "temporarily unavailable",
        "high demand",
    )

    return any(marker in message for marker in markers)


def ask_gemini(message):
    """Use Gemini for normal chat and fall back to Hugging Face on transient errors."""

    try:
        return gemini_llm.invoke(message).content
    except Exception as exc:
        if not _is_transient_provider_error(exc):
            raise

        return hf_llm.invoke(message).content
