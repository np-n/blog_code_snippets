"""
LLM Setup — Switchable between OpenAI, OpenRouter, and Groq
=============================================================
Control via .env:

    LLM_PROVIDER=openai          # openai | openrouter | groq  (default: openai)
    LLM_MODEL=gpt-4.1-mini      # optional model override

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OpenAI models (OPENAI_API_KEY required)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
gpt-4.1-mini          ← default  (fast, cheap, great tool-calling)
gpt-4.1
gpt-4o
gpt-4o-mini

Get a key → https://platform.openai.com/api-keys

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Free tool-calling models on OpenRouter (OPENROUTER_API_KEY required)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
meta-llama/llama-3.3-70b-instruct:free
openai/gpt-oss-20b:free
nvidia/nemotron-3-super-120b-a12b:free

Browse current free models → https://openrouter.ai/models?max_price=0
Get a free key → https://openrouter.ai

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Free models on Groq (GROQ_API_KEY required)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
llama-3.3-70b-versatile         ← Groq default
llama-3.1-8b-instant            fastest

Get a free key → https://console.groq.com
"""

import os

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv()

_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
_MODEL_OVERRIDE = os.getenv("LLM_MODEL")

_DEFAULTS: dict[str, str] = {
    "openai":      "gpt-4.1-mini",
    "openrouter":  "openai/gpt-oss-20b:free",
    "groq":        "llama-3.3-70b-versatile",
}


def get_llm(
    model: str | None = None,
    temperature: float = 0,
) -> BaseChatModel:
    """
    Return a configured LangChain chat model.

    Provider  → LLM_PROVIDER env var  (openai | openrouter | groq, default: openai)
    Model     → model arg > LLM_MODEL env > provider default

    Args:
        model: Optional model ID override.
        temperature: Sampling temperature (0 = deterministic).

    Returns:
        BaseChatModel ready for use with LangChain / LangGraph agents.
    """
    provider = _PROVIDER
    chosen_model = model or _MODEL_OVERRIDE or _DEFAULTS.get(provider, "gpt-4.1-mini")

    if provider == "openai":
        return _openai(chosen_model, temperature)
    if provider == "openrouter":
        return _openrouter(chosen_model, temperature)
    if provider == "groq":
        return _groq(chosen_model, temperature)

    raise ValueError(
        f"Unknown LLM_PROVIDER='{provider}'. Use 'openai', 'openrouter', or 'groq'."
    )


# ── Builders ──────────────────────────────────────────────────────────────────

def _openai(model: str, temperature: float) -> BaseChatModel:
    """ChatOpenAI pointed directly at the OpenAI API."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError(
            "OPENAI_API_KEY is not set.\n"
            "Get a key at https://platform.openai.com/api-keys and add it to your .env"
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )


def _openrouter(model: str, temperature: float) -> BaseChatModel:
    """ChatOpenAI pointed at OpenRouter's OpenAI-compatible endpoint."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key or api_key == "your_openrouter_api_key_here":
        raise ValueError(
            "OPENROUTER_API_KEY is not set.\n"
            "Get a free key at https://openrouter.ai and add it to your .env"
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://github.com/mcp-research-assistant",
            "X-Title": "MCP Research Assistant",
        },
    )


def _groq(model: str, temperature: float) -> BaseChatModel:
    """ChatGroq with the specified model."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY is not set.\n"
            "Get a free key at https://console.groq.com and add it to your .env"
        )
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=model,
        temperature=temperature,
        groq_api_key=api_key,
    )
