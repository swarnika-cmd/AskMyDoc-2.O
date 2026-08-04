import os
import time
import random
import string
import base64
import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.database import save_documents_to_store

load_dotenv()

def ingest_document(file_path: str, collection_name: str, original_file_name: str = "") -> dict:
    """
    Ingests a document (TXT or PDF), applies Parent-Child chunking,
    generates HuggingFace embeddings, and saves them to the vector store.
    If the PDF contains no extractable text, falls back to the OCR.space API.
    """
    print(f"Starting ingestion for: {file_path} ({original_file_name})")
    
    docs = []
    is_text = original_file_name.lower().endswith('.txt') or file_path.lower().endswith('.txt')
    
    if is_text:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            docs = [Document(page_content=text, metadata={"source": original_file_name or file_path})]
            print("Loaded 1 text document.")
        except Exception as e:
            raise ValueError(f"Failed to read text file: {e}")
    else:
        # Load PDF using PyPDFLoader
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Check if PDF contains extractable text
            total_text_length = sum(len(doc.page_content.strip()) for doc in docs)
            has_text = total_text_length > 50
            
            if not docs or not has_text:
                print("PyPDFLoader found no extractable text. Attempting OCR fallback...")
                docs = run_ocr_fallback(file_path, original_file_name)
            else:
                # Update source metadata for loaded docs
                for doc in docs:
                    doc.metadata["source"] = original_file_name or file_path
                print(f"Loaded {len(docs)} pages from PDF.")
        except Exception as pdf_error:
            print(f"PyPDFLoader error: {pdf_error}. Trying OCR fallback...")
            try:
                docs = run_ocr_fallback(file_path, original_file_name)
            except Exception as ocr_error:
                raise ValueError(f"Failed to parse PDF and OCR fallback failed: {ocr_error}")

    if not docs:
        print("Ingestion yielded 0 documents.")
        return {"success": False, "chunksCount": 0}

    # 2. Chunking: Parent Document Strategy
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=300
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    )

    parent_docs = parent_splitter.split_documents(docs)
    child_chunks = []

    for parent in parent_docs:
        # Generate parent ID matching JS style (parent_timestamp_random)
        random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=9))
        parent_id = f"parent_{int(time.time() * 1000)}_{random_str}"
        parent.metadata["parentId"] = parent_id

        children = child_splitter.split_documents([parent])
        for child in children:
            child.metadata["parentId"] = parent_id
            child.metadata["parentContent"] = parent.page_content
            # Keep original source in child metadata
            child.metadata["source"] = parent.metadata.get("source", original_file_name or file_path)
            if "ocr" in parent.metadata:
                child.metadata["ocr"] = True
            child_chunks.append(child)

    print(f"Split document into {len(parent_docs)} parent chunks and {len(child_chunks)} child chunks.")

    if not child_chunks:
        return {"success": True, "chunksCount": 0}

    # 3. Embeddings: Initialize HuggingFace embeddings (Inference or Local)
    hf_token = os.getenv("HF_TOKEN")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    if hf_token:
        print("Initializing HuggingFace Serverless embeddings...")
        try:
            from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
            embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key=hf_token,
                model_name=model_name
            )
            # Test connection
            embeddings.embed_query("test")
            print("Serverless Embeddings API ready")
        except Exception as e:
            print(f"Serverless Embeddings API failed to initialize: {e}. Falling back to local Embeddings.")
            embeddings = get_local_embeddings(model_name)
    else:
        print("HF_TOKEN not set. Initializing local HuggingFace embeddings...")
        embeddings = get_local_embeddings(model_name)

    # 4. Storage
    try:
        save_documents_to_store(child_chunks, embeddings, collection_name)
        print(f"Successfully ingested {len(child_chunks)} chunks into vector store, namespace: {collection_name}")
        return {"success": True, "chunksCount": len(child_chunks)}
    except Exception as e:
        print(f"Vector store saving error: {e}")
        raise RuntimeError(f"Vector store error: {e}")

def get_local_embeddings(model_name: str):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    print(f"Initializing local HuggingFace embeddings ({model_name}) with batch size: {batch_size}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={'normalize_embeddings': True, 'batch_size': batch_size}
    )

def run_ocr_fallback(file_path: str, original_file_name: str) -> list:
    """
    Sends the PDF file to OCR.space API and returns a list containing one Document.
    """
    api_key = os.getenv("OCR_API_KEY", "helloworld")
    try:
        with open(file_path, "rb") as f:
            pdf_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        payload = {
            'base64Image': f"data:application/pdf;base64,{pdf_base64}",
            'language': 'eng',
            'isOverlayRequired': 'false',
            'OCREngine': '2'
        }
        headers = {
            'apikey': api_key
        }
        
        print("Sending request to OCR.space...")
        response = requests.post(
            'https://api.ocr.space/parse/image',
            data=payload,
            headers=headers,
            timeout=180
        )
        
        if response.status_code != 200:
            raise ValueError(f"OCR Service returned status code {response.status_code}")
            
        data = response.json()
        if data.get("IsErroredOnProcessing"):
            raise ValueError(f"OCR API Error: {data.get('ErrorMessage') or 'Unknown error'}")
            
        parsed_results = data.get("ParsedResults", [])
        extracted_text = "\n".join(r.get("ParsedText", "") for r in parsed_results)
        
        if extracted_text.strip():
            print("Loaded 1 document via OCR fallback.")
            return [Document(
                page_content=extracted_text, 
                metadata={"source": original_file_name or file_path, "ocr": True}
            )]
        else:
            print("OCR fallback yielded no text.")
            return []
    except Exception as e:
        print(f"OCR fallback error: {e}")
        raise
