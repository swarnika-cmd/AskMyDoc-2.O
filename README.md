<div align="center">
  <h1 align="center">AskMyDoc: NotebookLM RAG (Python Rebuild)</h1>
  <p align="center">
    <strong>A production-ready RAG application that lets you upload, chat, and evaluate your documents.</strong>
  </p>
  
  <br />
  
  <p><em>Turn any static PDF into an intelligent conversation in seconds. Now powered by Python FastAPI.</em></p>

  <br />
  <img src="public/demo_screenshot.png" alt="AskMyDoc UI Screenshot" width="100%" style="border-radius: 12px; border: 1px solid #e2ddd4; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
</div>


---

## 🌟 Overview
**AskMyDoc** is an end-to-end Retrieval-Augmented Generation (RAG) web application inspired by Google NotebookLM. It allows users to upload PDF and TXT documents, parses and splits them into intelligent semantic chunks, and enables natural language conversations grounded strictly in the document's content.

This Python rebuild introduces a FastAPI backend, uses Python LangChain, integrates a dual vector store layout (ChromaDB or Pinecone), and features a real-time LLM-as-a-judge faithfulness (groundedness) evaluation.

## ✨ Key Features
- 🚀 **FastAPI Backend**: Built using Python FastAPI for asynchronous performance and typed validation.
- ⚡ **Lightning Fast LLM**: Powered by **Groq** (`llama-3.1-8b-instant`) for instantaneous, deterministic, and grounded responses.
- 🧠 **Flexible Embeddings**: Utilizes HuggingFace embeddings (`sentence-transformers/all-MiniLM-L6-v2`) via HF Serverless API or local sentence-transformers fallback.
- 🗄️ **Dual Vector Storage**: Supports **ChromaDB** (local persistent storage under `./chroma_db`) and **Pinecone** (cloud), selectable via `.env` configuration.
- 🏗️ **Parent Document Retrieval**: Implements a dual-chunking strategy (3000-char parent context / 400-char child vectors) to maximize retrieval precision while preserving deep context.
- 💬 **Conversational Memory**: Features active chat history routing. The AI intelligently reformulates follow-up questions into standalone queries before searching the vector database.
- 🛡️ **Confidence and Faithfulness**:
  - Displays grounding confidence badges (**HIGH CONFIDENCE**, **LOW CONFIDENCE**, or **NOT FOUND IN DOCUMENT**).
  - Automatically evaluates the answer's **Faithfulness (Groundedness)** in real-time, displaying a percentage score of how grounded the AI response is in the source text.
- 📁 **Preloaded Demo Mode**: Instantly load and test-query the preloaded "RBI Integrated Ombudsman Scheme 2021" policy document.
- 🎨 **Premium User Interface**: Features a highly aesthetic, responsive, and minimalist frontend inspired by Google NotebookLM.

## 🛠️ Technology Stack
- **Backend:** Python 3, FastAPI, Uvicorn
- **Frontend:** Vanilla HTML, CSS, JavaScript (NotebookLM aesthetic)
- **Orchestration:** LangChain (Python)
- **Vector DBs:** ChromaDB (Local persistent) & Pinecone (Cloud)
- **Document Parsing:** `PyPDF` / `pdfplumber` with OCR Space API fallback

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- A Groq API Key (Free)
- A HuggingFace Token (Optional, for HF Inference embeddings)

### 2. Installation
Clone the repository and set up a Python virtual environment:
```bash
# Clone the repository
git clone https://github.com/swarnika-cmd/AskMyDoc.git
cd AskMyDoc

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Setup
Rename `.env.example` to `.env` and fill in your keys:
```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
VECTOR_STORE_TYPE=chromadb
```

### 4. Running the App
Start the FastAPI server:
```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Navigate to `http://127.0.0.1:8000/app.html` in your browser. Upload documents, query them, and watch the confidence and faithfulness evaluation process in real-time!

### 5. Running Tests
To run the automated verification script:
```bash
python tests/test_backend.py
```
This script uploads a temporary document, runs grounded and out-of-bounds chat queries, tests the faithfulness evaluator, and deletes the vectors.
