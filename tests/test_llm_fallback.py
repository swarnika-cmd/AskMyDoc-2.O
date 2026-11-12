"""Offline unit tests for the LLM client: caching and the Ollama -> Groq fallback.
Run: pytest tests/test_llm_fallback.py  (no server, no network)."""
import socket
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import app.llm as llm


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _env(monkeypatch, **kw):
    for k in ("LLM_PROVIDER", "LLM_BASE_URL", "GROQ_API_KEY", "GROQ_MODEL"):
        monkeypatch.delenv(k, raising=False)
    for k, v in kw.items():
        monkeypatch.setenv(k, v)
    llm._build_llm.cache_clear()


def test_groq_used_when_ollama_unreachable(monkeypatch):
    _env(monkeypatch, LLM_PROVIDER="ollama", LLM_BASE_URL=f"http://127.0.0.1:{_closed_port()}",
         GROQ_API_KEY="test-key", GROQ_MODEL="test/model")
    fake_groq = FakeListChatModel(responses=["from-groq"])
    with patch.object(llm, "_groq", return_value=fake_groq) as groq_ctor:
        model = llm.get_llm()
        out = model.invoke([HumanMessage(content="hi")])
    groq_ctor.assert_called_once_with("test/model", "test-key")
    assert llm.message_text(out) == "from-groq"
    assert llm.last_served() == {"provider": "groq", "model": "test/model"}


def test_no_fallback_without_groq_key(monkeypatch):
    _env(monkeypatch, LLM_PROVIDER="ollama", LLM_BASE_URL=f"http://127.0.0.1:{_closed_port()}")
    model = llm.get_llm()
    try:
        model.invoke([HumanMessage(content="hi")])
        raised = False
    except Exception:
        raised = True
    assert raised


def test_client_is_cached(monkeypatch):
    _env(monkeypatch, LLM_PROVIDER="ollama")
    assert llm.get_llm() is llm.get_llm()


def test_message_text_handles_parts_and_plain():
    assert llm.message_text(AIMessage(content="  ok ")) == "ok"
    assert llm.message_text(AIMessage(content=[{"type": "text", "text": "a"}, {"type": "reasoning", "text": "x"}, "b"])) == "ab"


def test_think_blocks_are_stripped():
    assert llm.message_text(AIMessage(content="<think>hmm</think>\nfinal")) == "final"


def test_timeouts_are_part_of_cache_key(monkeypatch):
    _env(monkeypatch, LLM_PROVIDER="ollama")
    a = llm.get_llm()
    monkeypatch.setenv("OLLAMA_READ_TIMEOUT", "5")
    assert llm.get_llm() is not a
    monkeypatch.delenv("OLLAMA_READ_TIMEOUT")
