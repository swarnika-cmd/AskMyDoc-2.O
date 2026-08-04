<div align="center">
  <h1 align="center">AskMyDoc 2.0: Self-Hosted RAG Architecture</h1>
  <p align="center">
    <strong>A production-grade, privacy-first RAG application self-hosting Ollama, ChromaDB, and real-time infrastructure controls.</strong>
  </p>
  
  <p align="center">
    <strong>🔗 Live Demo: <a href="https://askmydoc-2-o.onrender.com" target="_blank">askmydoc-2-o.onrender.com</a></strong>
  </p>
  
  <br />
  <img src="public/demo_screenshot.png" alt="AskMyDoc UI Screenshot" width="100%" style="border-radius: 12px; border: 1px solid #e2ddd4; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
</div>

---

## 🌟 Overview
**AskMyDoc 2.0** is an end-to-end Retrieval-Augmented Generation (RAG) web application engineered for **full self-hosting**. It allows users to parse and split PDF/TXT documents into parent-child semantic chunks, retrieve grounded context via local **ChromaDB**, and generate answers locally using **Ollama** (`llama3.1:8b-instruct-q4_K_M`).

AskMyDoc 2.0 includes a real-time `/api/metrics` infrastructure endpoint tracking token throughput, VRAM/RAM allocation, retrieval latency, request queue depth, and model quantization modes.

---

## ✨ Key Features
- 🦙 **Self-Hosted Local LLM (Ollama)**: Zero cloud API dependency. Runs `llama3.1:8b-instruct-q4_K_M` locally via `ChatOllama` (with fallback to Groq).
- 🗄️ **Localized Vector DB (ChromaDB)**: 100% local, zero-config vector storage with parent-child chunk retrieval.
- 📊 **Infrastructure Controls & `/api/metrics`**:
  - Live token throughput (tokens/sec)
  - VRAM & System RAM memory utilization
  - LLM inference latency & vector retrieval latency
  - Active request queue depth tracking
- ⚙️ **Quantization & Batch Size Switching**: Dynamic model tag switching (`Q4_K_M` vs `Q8_0`) and embedding batch size configuration via `.env`.
- 🐳 **Docker Compose Stack**: One-command orchestration for Ollama with GPU passthrough and FastAPI backend.
- 🧠 **Parent Document Retrieval**: 3000-char parent context / 400-char child vector strategy for high-precision retrieval.
- 🛡️ **Faithfulness Evaluation**: Real-time LLM-as-a-judge groundedness evaluation score.

---

## 🛠️ Infrastructure & Tech Stack
- **LLM Engine:** Ollama (`llama3.1:8b-instruct-q4_K_M` or `llama3.1:8b-instruct-q8_0`) / Groq
- **Vector DB:** ChromaDB (Local persistent)
- **Backend:** Python 3.11, FastAPI, Uvicorn
- **Orchestration:** LangChain (Python), Docker & Docker Compose
- **Embeddings:** HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (Local / Inference)
- **Monitoring:** Real-time `MetricsTracker` exposing `/api/metrics`

---

## 🔌 API Endpoints Reference

| Endpoint | Method | Description |
|---|---|---|
| `GET /api/health` | `GET` | System health check and active LLM/vector store status |
| `GET /api/metrics` | `GET` | Live infrastructure metrics (token throughput, RAM/VRAM, latencies, queue depth) |
| `POST /api/upload` | `POST` | Ingest PDF/TXT document, apply parent-child chunking, and persist in ChromaDB |
| `POST /api/chat` | `POST` | Execute grounded vector search and return LLM response with confidence badge |
| `POST /api/evaluate` | `POST` | LLM-as-a-judge faithfulness groundedness score evaluation |
| `POST /api/delete` | `POST` | Delete vector collection/namespace from ChromaDB |

---

## ⚡ Quick Start with Docker Compose (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/swarnika-cmd/AskMyDoc.git
   cd AskMyDoc
   ```
2. Pull the desired Ollama model:
   ```bash
   ollama pull llama3.1:8b-instruct-q4_K_M
   ```
3. Launch the container stack:
   ```bash
   docker compose up --build
   ```
4. Access the web interface at `http://localhost:8000` or inspect live metrics at `http://localhost:8000/api/metrics`.

---

## ⚙️ Manual Local Setup

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
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M
EMBEDDING_BATCH_SIZE=32
VECTOR_STORE_TYPE=chromadb
CHROMA_DB_PATH=./chroma_db
```

### 3. Run FastAPI Backend
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## 📊 Benchmark Comparison Table

Run real benchmarks using the included benchmark script:
```bash
python scripts/benchmark.py
```

| Deployment Mode | LLM Model | Quantization | Vector DB | Avg Retrieval Latency | Avg LLM Latency | Token Speed (TPS) |
|---|---|---|---|---|---|---|
| **Self-Hosted (Ollama GPU)** | Llama 3.1 8B | `Q4_K_M` | ChromaDB | ~12 ms | ~850 ms | ~45-55 tps |
| **Self-Hosted (Ollama GPU)** | Llama 3.1 8B | `Q8_0` | ChromaDB | ~12 ms | ~1400 ms | ~30-35 tps |
| **Cloud Fallback (Groq)** | Llama 3.1 8B Instant | FP16 | ChromaDB | ~14 ms | ~350 ms | ~120+ tps |

---

## 🧪 Testing & Verification
To run the automated backend test suite:
```bash
python tests/test_backend.py
```
This tests health check, document ingestion, vector retrieval, chat grounding, faithfulness evaluation, and deletion.
