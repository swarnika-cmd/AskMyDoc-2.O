# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] - 2026-11-13

### Fixed
- Groq fallback now applies at request time: if Ollama is unreachable or errors and `GROQ_API_KEY` is set, the call is retried on Groq (it previously only covered client construction). Ollama timeouts are configurable: `OLLAMA_CONNECT_TIMEOUT` (default 3 s) and `OLLAMA_READ_TIMEOUT` (default none, as before).
- Embeddings are local unless `EMBEDDING_BACKEND=hf_api` is set explicitly; a merely present `HF_TOKEN` no longer switches to the Inference API (which had no fallback once the per-request probe was removed).
- `/api/metrics` no longer invents a quantization (`unknown` unless the Ollama tag states one), reports the provider that actually served the last request, and reports app-process memory (CUDA allocation or RSS).
- Faithfulness is computed in code from the judge's claims; a judge failure returns `null` and the UI shows "unavailable" instead of 50%.
- Upload file names are reduced to their base name; empty or dot-only names are rejected (path traversal fix).
- `message_text` strips inline `<think>` blocks.
- Groq fallback: `llama-3.1-8b-instant` no longer exists on Groq. The model is now `GROQ_MODEL` (default `openai/gpt-oss-20b`). Reasoning output of `gpt-oss` models is kept out of the answer text.
- Removed the extra `embed_query("test")` call that ran on every chat request and on every upload.
- The LLM client and the embeddings object are created once per process instead of on every request.
- `/api/metrics` reports the configured Groq model instead of a hard-coded one.

### Changed
- README, TECHNICAL_DOCS and CONTRIBUTING describe the v2 stack as it is: FastAPI, LangChain, ChromaDB, all-MiniLM-L6-v2 embeddings, Ollama locally and Groq as the hosted option. Pinecone is noted as v1 only.
- CONTRIBUTING now describes the Python workflow (it described the old Node.js one).
- Added `EMBEDDING_MODEL`, `EMBEDDING_BACKEND`, `GROQ_MODEL` and the Ollama timeout settings to `.env.example`; the Groq key line is commented out so a copied file does not carry a placeholder key.
- Raised the `langchain-ollama` floor to 1.1.0 (the version tested here; `client_kwargs` is used) and `langchain-groq` to 1.1.0.
- Chroma store wrappers are cached per (path, namespace).
- `public/index.html` replaced with a short factual page.

### Removed
- Gemini mentions in the UI (`public/app.html`) and the unused `langchain-google-genai` dependency.
- Marketing claims the code does not support: user counts, testimonials, customer logos, pricing plans, encryption at rest, "no hallucinations", page-level citations, team collaboration, SSO, API plans, DOCX/CSV/XLSX/EPUB/HTML support, "production-grade", "zero cloud API dependency".
- The unpublished benchmark table from the README (the figures had no script output behind them).
- The preloaded "RBI Guidelines" demo button and `demo-compliance` handling (no such collection ships with the repo).
- `vercel.json`, which pointed at a non-existent `server.js`, and the `VERCEL` branch in `app/main.py`. The app deploys with Docker.
