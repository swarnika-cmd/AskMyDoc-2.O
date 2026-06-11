# Technical Documentation: AskMyDoc RAG System (Python Rebuild)

This document provides an architectural overview of the AskMyDoc Retrieval-Augmented Generation (RAG) system, outlining the design decisions, data flow, and specific technical implementations that power the application.

## 1. System Architecture

AskMyDoc follows a standard RAG paradigm decoupled into three distinct layers:
1. **Frontend (Client)**: A vanilla HTML/CSS/JS single-page application handling file uploads, user prompts, and UI state management (inspired by Google NotebookLM).
2. **Backend API (Server)**: A Python FastAPI application that exposes endpoints for document upload (`/api/upload`), LLM chat generation (`/api/chat`), document deletion (`/api/delete`), and faithfulness evaluation (`/api/evaluate`).
3. **RAG Engine (LangChain)**: The core Python logic orchestrating document parsing, text splitting (Parent-Child chunking), embedding generation, vector storage abstraction, and context-aware LLM retrieval.

---

## 2. Document Ingestion Pipeline (`app/ingest.py`)

When a user uploads a document, the system processes it through a strict pipeline to ensure maximum semantic retention.

### 2.1 File Parsing & Validation
- **Native PDF Parsing**: The system first attempts to extract raw text layers from PDFs using `pypdf` (via LangChain's `PyPDFLoader`).
- **OCR Fallback**: If the PDF is purely image-based or scanned (detected by extracting less than 50 characters of meaningful text), the system automatically routes the file to the external OCR API (`ocr.space`) to visually extract the text using python `requests`. This prevents "silent failures" where scanned documents result in empty vector spaces.

### 2.2 Parent-Child Semantic Chunking
- **Strategy**: Dual-Chunking Parent-Child strategy using `RecursiveCharacterTextSplitter`.
- **Configuration**:
  - **Parent Chunks**: `chunkSize: 3000`, `chunkOverlap: 300`
  - **Child Chunks**: `chunkSize: 400`, `chunkOverlap: 100`
- **Rationale**: Searching small child chunks (400 chars) increases the mathematical precision of the similarity search. However, small chunks don't contain enough surrounding context for the LLM to generate a complete answer. When a child chunk matches the query, the system retrieves its associated **parent chunk** (3000 chars) to feed as context to the LLM. This provides high-precision retrieval with comprehensive, context-rich grounding.

### 2.3 Embeddings & Vector Database Abstraction (`app/database.py`)
- **Embeddings**: Text chunks are embedded using the HuggingFace `sentence-transformers/all-MiniLM-L6-v2` model. The system tries the HuggingFace Serverless Inference API first, and automatically falls back to local execution if the API key is not configured.
- **Database Abstraction**: Supports both **ChromaDB** (local persistent database) and **Pinecone** (cloud database), selectable via `.env` configuration.
- **Namespace Isolation**: Every uploaded document is assigned a unique namespace (e.g., `doc_report_pdf_1778449428219`). This allows the system to easily isolate vectors by document, enabling features like per-document deletion and targeted multi-document queries.

---

## 3. Retrieval & Generation Pipeline (`app/retrieve.py`)

When a user submits a query, the backend reconstructs the context dynamically before querying the LLM.

### 3.1 Conversational Query Reformulation
If a chat session has history, the user's latest query might contain pronoun references (e.g., "What are its key terms?"). The system reformulates the query into a standalone question using Groq's `llama-3.1-8b-instant` LLM based on the conversation history before searching the database.

### 3.2 Multi-Document Cross-Retrieval
The application supports querying across *multiple* active documents simultaneously.
1. The frontend identifies all checked documents in the sidebar and sends their namespaces to the backend.
2. The backend searches **every** provided namespace for the top 5 most relevant chunks.
3. Chunks are aggregated across namespaces, sorted descending by similarity, and sliced to the top 5 child chunks across the entire workspace.

### 3.3 Distance Metrics & Score Normalization
- **ChromaDB** returns L2 squared distance (smaller is better). We normalize this to a similarity score: `similarity = 1 - distance / 2`.
- **Pinecone** returns cosine similarity directly.
The system sorts all retrieved chunks uniformly using these normalized scores, preventing database bias from affecting multi-document retrieval.

### 3.4 Grounding & Prompting
Deduplicated parent contexts of the top child chunks are combined into a system prompt. The LLM is forced to return structured JSON containing:
- `isConversational` (boolean)
- `noEvidence` (boolean)
- `answer` (string)

---

## 4. Faithfulness Evaluation (`app/evaluate.py`)

AskMyDoc implements an automated **LLM-as-a-Judge** faithfulness (groundedness) evaluation after every retrieval.
- **Claims Extraction**: The evaluator prompts the LLM to identify all independent factual claims made in the generated answer.
- **Support Checking**: For each claim, the LLM determines whether it is directly supported by the retrieved context chunks (returning true/false and a detailed explanation).
- **Faithfulness Score**: Evaluated as `(number of supported claims) / (total claims)`.
- **UI Presentation**: The frontend displays a real-time progress indicator which updates to show the groundedness percentage (e.g., "Faithfulness: 100% (Grounded)") with a tooltip explaining the reasoning.
