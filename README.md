<div align="center">
  <h1 align="center">AskMyDoc 2.0: RAG over your PDF and text files</h1>
  <p align="center">
    <strong>A FastAPI app that answers questions from uploaded documents using LangChain, ChromaDB, all-MiniLM-L6-v2 embeddings, and Ollama (local) or Groq (hosted) as the LLM.</strong>
  </p>

  <br />
  <img src="public/demo_screenshot.png" alt="AskMyDoc UI Screenshot" width="100%" style="border-radius: 12px; border: 1px solid #e2ddd4; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
</div>

---

## Overview
AskMyDoc 2.0 parses PDF/TXT files into parent and child chunks, embeds the child chunks with Hugging Face `sentence-transformers/all-MiniLM-L6-v2`, and stores them in ChromaDB (one collection per document). A question retrieves the top matching chunks and passes their parent chunks to an LLM that is told to answer only from that context.

- **LLM:** Ollama (default `llama3.1:8b-instruct-q4_K_M`, set with `OLLAMA_MODEL`). Groq is used when `LLM_PROVIDER=groq`, and as the fallback when Ollama fails and `GROQ_API_KEY` is set. The Groq model is `GROQ_MODEL` (default `openai/gpt-oss-20b`). When a Groq key is set, any Ollama error at request time (server unreachable, model missing) re-runs the request on Groq. Timeouts: `OLLAMA_CONNECT_TIMEOUT` (seconds, default 3) and `OLLAMA_READ_TIMEOUT` (seconds, default none).
- **Embeddings:** all-MiniLM-L6-v2 on the local machine by default. Set `EMBEDDING_BACKEND=hf_api` (and `HF_TOKEN`) to use the Hugging Face Inference API instead; it is never selected implicitly, and there is no automatic fallback between the two. `EMBEDDING_MODEL` overrides the model id or path.
- **Parent/child chunking:** 3000-char parents (300 overlap), 400-char children (100 overlap) (`app/ingest.py`).
- **Faithfulness check:** `POST /api/evaluate` asks the LLM to list the claims in an answer and mark each as supported or not by the context. The score is computed in code as supported claims / total claims from the judge's per-claim verdicts. It is a model judgement, not a guarantee. If the judge fails or returns no usable verdicts, the score is `null` and the UI shows "unavailable".
- **`/api/metrics`:** request counts, average LLM and retrieval latency, queue depth, an approximate tokens/sec (word count x 1.3 over LLM time), and memory of the app process only: `torch.cuda.memory_allocated` if CUDA is available, otherwise process RSS (not Ollama's memory). `provider` and `model_name` show what served the last request (Ollama or the Groq fallback); `quantization` is `unknown` unless the Ollama tag ends in a quant suffix such as `q4_K_M`.
- **Scanned PDFs:** if PyPDF (30 characters or fewer) and then pdfplumber extract no text, the file is sent to the third-party OCR.space API.
- **Docker Compose:** `docker-compose.yml` runs Ollama and the FastAPI app. It reserves an NVIDIA GPU for Ollama, so it fails to start without the NVIDIA container runtime (see Quick Start for alternatives).

Pinecone was used in v1 (the Node.js version). v2 has no Pinecone code.

---

## Stack
- Python 3.11, FastAPI, Uvicorn
- LangChain, ChromaDB (local persistent)
- Hugging Face `sentence-transformers/all-MiniLM-L6-v2` embeddings
- Ollama (local LLM), Groq (hosted LLM / fallback)
- Docker and Docker Compose

---

## API Endpoints Reference

| Endpoint | Method | Description |
|---|---|---|
| `GET /api/health` | `GET` | Returns `status` plus the configured LLM provider and vector store type. Config only; it does not contact Ollama or Groq |
| `GET /api/metrics` | `GET` | Live infrastructure metrics (approximate token throughput, app-process memory (CUDA allocation or process RSS, not Ollama's VRAM), latencies, queue depth) |
| `POST /api/upload` | `POST` | Ingest PDF/TXT document, apply parent-child chunking, and persist in ChromaDB |
| `POST /api/chat` | `POST` | Execute grounded vector search and return LLM response with confidence badge |
| `POST /api/evaluate` | `POST` | LLM-as-a-judge faithfulness groundedness score evaluation |
| `POST /api/delete` | `POST` | Delete vector collection/namespace from ChromaDB |

---

## Quick Start with Docker Compose (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/swarnika-cmd/AskMyDoc-2.O.git
   cd AskMyDoc-2.O
   ```
2. Start the stack, then pull the Ollama model into the Ollama container:
   ```bash
   docker compose up -d --build
   docker compose exec ollama ollama pull llama3.1:8b-instruct-q4_K_M
   ```
   The compose file reserves an NVIDIA GPU for Ollama (`deploy.resources` block), so it fails without the NVIDIA container runtime. To run without a GPU, remove that block from the `ollama` service. Alternatively run Ollama on the host and set `LLM_BASE_URL` for the app service (for example `http://host.docker.internal:11434`) and drop the `ollama` service.
4. Access the web interface at `http://localhost:8000` or inspect live metrics at `http://localhost:8000/api/metrics`.

---

## Manual Local Setup

### 1. Installation & Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
```

### 2. Configure Environment (`.env`)
```env
LLM_PROVIDER=ollama   # or groq
LLM_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M
# GROQ_API_KEY=...        # needed for LLM_PROVIDER=groq or the Groq fallback
# GROQ_MODEL=openai/gpt-oss-20b
EMBEDDING_BATCH_SIZE=32
VECTOR_STORE_TYPE=chromadb
CHROMA_DB_PATH=./chroma_db
```

### 3. Run FastAPI Backend
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## Deployment
The app deploys with Docker (`Dockerfile`, `docker-compose.yml`). There is no Vercel config: the dependencies (sentence-transformers with torch, ChromaDB on local disk) do not fit Vercel's Python runtime limits.

`scripts/benchmark.py` measures latency against a running server. No benchmark results are published here.

---

## Testing
To run the automated backend test suite:
```bash
python tests/test_backend.py
```
`tests/test_backend.py` and `tests/test_fastapi_client.py` (in-process `TestClient`, no server needed; `test_backend.py` calls a server on port 8000), and both need an LLM backend for chat; `tests/quick_unit_test.py` runs ingestion only.
