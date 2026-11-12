# Contributing to AskMyDoc

Thank you for showing interest in contributing to AskMyDoc! We welcome contributions of all forms, including bug fixes, feature requests, documentation, and interface enhancements.

## 🚀 How to Get Started

### 1. Fork and Clone
1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/AskMyDoc-2.O.git
   cd AskMyDoc-2.O
   ```

### 2. Set Up Environment
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Enter your API credentials (optionally a Groq key and a HuggingFace token) into the `.env` file.

### 3. Install Dependencies
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 4. Running Locally
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Go to `http://localhost:8000`. Or use `docker compose up --build`.

---

## 🛠️ Code Guidelines

* **Formatting:** Follow PEP 8 and keep changes focused.
* **Separation of Concerns:** 
  * Keep RAG retrieval logic isolated in `app/retrieve.py`.
  * Keep document parsing and chunking in `app/ingest.py`.
  * Keep routing and API handlers in `app/main.py`.
* **Testing:** Test new code locally by uploading sample files (a small PDF or TXT file) and running conversational queries.
