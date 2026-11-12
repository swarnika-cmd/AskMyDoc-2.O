import os
import re
import threading
from functools import lru_cache
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

DEFAULT_OLLAMA_MODEL = "llama3.1:8b-instruct-q4_K_M"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# Which provider/model served the most recent request (updated by TrackedLLM.invoke).
_served_lock = threading.Lock()
_last_served = {"provider": None, "model": None}


def last_served() -> dict:
    with _served_lock:
        return dict(_last_served)


def _record_served(provider: str, model: str):
    with _served_lock:
        _last_served["provider"] = provider
        _last_served["model"] = model


def groq_model_name() -> str:
    return os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)


def message_text(response) -> str:
    """
    Returns the final answer text of a chat response. Reasoning models (e.g. openai/gpt-oss-*)
    keep their reasoning outside `content`; some models inline it in <think>...</think> blocks,
    and content can also be a list of parts. Only the final text is returned.
    """
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "".join(
            part if isinstance(part, str) else part.get("text", "")
            for part in content
            if isinstance(part, str) or (isinstance(part, dict) and part.get("type", "text") == "text")
        )
    return _THINK_RE.sub("", content or "").strip()


class TrackedLLM:
    """
    Thin wrapper over a primary chat model with an optional fallback. If the primary raises at call
    time (server unreachable, model missing, ...) the request is re-run on the fallback. Records which
    provider actually served the request.
    """

    def __init__(self, primary, primary_label, fallback=None, fallback_label=None):
        self.primary, self.primary_label = primary, primary_label
        self.fallback, self.fallback_label = fallback, fallback_label

    def invoke(self, messages, *args, **kwargs):
        try:
            result = self.primary.invoke(messages, *args, **kwargs)
            _record_served(*self.primary_label)
            return result
        except Exception as e:
            if self.fallback is None:
                raise
            print(f"{self.primary_label[0]} request failed ({type(e).__name__}: {e}); falling back to {self.fallback_label[0]}.")
            result = self.fallback.invoke(messages, *args, **kwargs)
            _record_served(*self.fallback_label)
            return result


def _groq(groq_model: str, groq_key: str):
    from langchain_groq import ChatGroq
    return ChatGroq(api_key=groq_key, model_name=groq_model, temperature=0)


def _env_float(name: str, default: Optional[float]) -> Optional[float]:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


@lru_cache(maxsize=8)
def _build_llm(provider: str, ollama_model: str, base_url: str, groq_model: str, groq_key: str,
               connect_timeout: float, read_timeout: Optional[float]):
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        print(f"Initializing ChatOllama model '{ollama_model}' at {base_url}")
        ollama = ChatOllama(
            base_url=base_url,
            model=ollama_model,
            temperature=0,
            # Short connect timeout so an unreachable Ollama fails over quickly; no read timeout by default.
            client_kwargs={"timeout": httpx.Timeout(read_timeout, connect=connect_timeout)},
        )
        if groq_key:
            return TrackedLLM(ollama, ("ollama", ollama_model), _groq(groq_model, groq_key), ("groq", groq_model))
        return TrackedLLM(ollama, ("ollama", ollama_model))
    if provider == "groq":
        if not groq_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        print(f"Initializing ChatGroq model '{groq_model}'")
        return TrackedLLM(_groq(groq_model, groq_key), ("groq", groq_model))
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def get_llm():
    """
    Returns the configured chat model (Ollama, or Groq). With Ollama and a Groq key set, a request-time
    Ollama failure is retried on Groq. Cached per provider/model/endpoint/key/timeouts, so the client
    is built once, not per request.
    """
    return _build_llm(
        os.getenv("LLM_PROVIDER", "ollama").lower(),
        os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        os.getenv("LLM_BASE_URL", "http://localhost:11434"),
        groq_model_name(),
        os.getenv("GROQ_API_KEY", ""),
        _env_float("OLLAMA_CONNECT_TIMEOUT", 3.0),
        _env_float("OLLAMA_READ_TIMEOUT", None),
    )
