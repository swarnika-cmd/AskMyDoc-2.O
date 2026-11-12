"""Offline tests: upload filename handling, embeddings backend choice, faithfulness scoring, metrics."""
import json
import os

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import app.llm as llm
import app.main as main
import app.evaluate as evaluate
import app.ingest as ingest
from app.metrics import MetricsTracker


# ---- upload filename (path traversal) ----
@pytest.mark.parametrize("raw,expected", [
    ("../../etc/passwd.txt", "passwd.txt"),
    ("..\\..\\evil.txt", "evil.txt"),
    ("/abs/path/doc.pdf", "doc.pdf"),
    ("plain.txt", "plain.txt"),
])
def test_safe_upload_name(raw, expected):
    assert main.safe_upload_name(raw) == expected


@pytest.mark.parametrize("raw", ["", ".", "..", "../..", "...", "dir/"])
def test_safe_upload_name_rejects(raw):
    with pytest.raises(main.HTTPException):
        main.safe_upload_name(raw)


def test_upload_writes_inside_uploads_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen = {}

    def fake_ingest(path, collection, name):
        seen["path"], seen["name"] = path, name
        return {"success": True, "chunksCount": 1}

    monkeypatch.setattr(main, "ingest_document", fake_ingest)
    client = TestClient(main.app)
    r = client.post("/api/upload", files={"document": ("../../escape.txt", b"hello", "text/plain")})
    assert r.status_code == 200
    assert os.path.dirname(os.path.abspath(seen["path"])) == str(tmp_path / "uploads")
    assert seen["name"] == "escape.txt"
    assert not (tmp_path.parent / "escape.txt").exists()


# ---- embeddings backend ----
def test_hf_token_alone_does_not_switch_to_api(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_abc")
    monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)
    sentinel = object()
    monkeypatch.setattr(ingest, "get_local_embeddings", lambda name: sentinel)
    assert ingest.get_embeddings() is sentinel


def test_hf_api_backend_is_explicit(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "hf_api")
    monkeypatch.setenv("HF_TOKEN", "hf_abc")
    sentinel = object()
    monkeypatch.setattr(ingest, "_get_api_embeddings", lambda name, tok: sentinel)
    assert ingest.get_embeddings() is sentinel
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(ValueError):
        ingest.get_embeddings()


# ---- faithfulness ----
class _FakeJudge:
    def __init__(self, text=None, error=None):
        self.text, self.error = text, error

    def invoke(self, messages):
        if self.error:
            raise self.error
        return AIMessage(content=self.text)


def test_faithfulness_computed_from_claims(monkeypatch):
    payload = {"claims": [{"supported": True}, {"supported": True}, {"supported": False}, {"supported": True}],
               "score": 0.99, "explanation": "x"}
    monkeypatch.setattr(evaluate, "get_llm", lambda: _FakeJudge(json.dumps(payload)))
    out = evaluate.evaluate_faithfulness("q", "ctx", "an answer that is long enough to not be a greeting at all")
    assert out["score"] == 0.75


def test_faithfulness_null_on_error_and_empty(monkeypatch):
    monkeypatch.setattr(evaluate, "get_llm", lambda: _FakeJudge(error=RuntimeError("boom")))
    assert evaluate.evaluate_faithfulness("q", "c", "some longer answer text here for the judge to look at")["score"] is None
    monkeypatch.setattr(evaluate, "get_llm", lambda: _FakeJudge('{"claims": [], "score": 1.0}'))
    assert evaluate.evaluate_faithfulness("q", "c", "some longer answer text here for the judge to look at")["score"] is None


# ---- metrics ----
def _metrics(monkeypatch, provider, served, ollama_model="llama3.1:8b-instruct-q4_K_M"):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    monkeypatch.setenv("OLLAMA_MODEL", ollama_model)
    monkeypatch.setattr("app.metrics.last_served", lambda: served)
    return MetricsTracker().get_metrics()


def test_metrics_quantization_not_invented(monkeypatch):
    none = {"provider": None, "model": None}
    assert _metrics(monkeypatch, "ollama", none)["quantization"] == "Q4_K_M"
    assert _metrics(monkeypatch, "ollama", none, "llama3.1:8b")["quantization"] == "unknown"
    assert _metrics(monkeypatch, "groq", none)["quantization"] == "unknown"


def test_metrics_reports_provider_that_served(monkeypatch):
    m = _metrics(monkeypatch, "ollama", {"provider": "groq", "model": "openai/gpt-oss-20b"})
    assert m["provider"] == "groq" and m["configured_provider"] == "ollama"
    assert m["quantization"] == "unknown"
