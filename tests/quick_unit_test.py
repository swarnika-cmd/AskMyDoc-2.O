import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Add project root to sys.path
sys.path.insert(0, os.getcwd())

from app.metrics import metrics
from app.database import get_vector_store, save_documents_to_store, delete_namespace
from app.ingest import get_local_embeddings
from langchain_core.documents import Document

def test_unit():
    print("--- 1. Testing Metrics Tracker ---")
    m = metrics.get_metrics()
    print("Initial metrics:", m)
    assert "model_name" in m
    assert "memory" in m
    print("[PASS] Metrics test passed.")

    print("\n--- 2. Testing Local ChromaDB Vector Store ---")
    embeddings = get_local_embeddings("sentence-transformers/all-MiniLM-L6-v2")
    doc = Document(page_content="AskMyDoc 2.0 supports local ChromaDB and Ollama.", metadata={"source": "test.txt", "parentId": "parent_1"})
    ns = "quick_test_ns"
    save_documents_to_store([doc], embeddings, ns)
    
    vs = get_vector_store(embeddings, ns)
    results = vs.similarity_search_with_score("AskMyDoc", k=1)
    print(f"Retrieved {len(results)} chunks. Content: '{results[0][0].page_content}'")
    assert len(results) > 0
    assert "AskMyDoc" in results[0][0].page_content
    print("[PASS] Vector Store test passed.")

    print("\n--- 3. Testing Vector Store Deletion ---")
    deleted = delete_namespace(ns)
    print("Deleted namespace:", deleted)
    assert deleted is True
    print("[PASS] Deletion test passed.")

    print("\n--- ALL UNIT TESTS COMPLETED SUCCESSFULLY ---")

if __name__ == "__main__":
    test_unit()
