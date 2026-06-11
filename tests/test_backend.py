import os
import sys
import time
import requests

# Base URL for the FastAPI backend (local)
BASE_URL = "http://127.0.0.1:8000"

def test_health():
    print("Testing /api/health...")
    try:
        res = requests.get(f"{BASE_URL}/api/health")
        print("Health response:", res.json())
        assert res.status_code == 200
        assert res.json().get("status") == "ok"
        print("[OK] Health test passed")
        return True
    except Exception as e:
        print("[FAIL] Health test failed:", e)
        return False

def test_upload_and_chat():
    print("\nTesting /api/upload and /api/chat...")
    test_filename = "temp_test_doc.txt"
    test_filepath = os.path.join(os.getcwd(), test_filename)
    collection_name = f"test_collection_{int(time.time())}"
    
    # Write a temporary test file with specific content
    test_content = (
        "AskMyDoc is an advanced document assistant built on Python, FastAPI, and LangChain. "
        "It supports parent-child chunking for high accuracy retrieval. "
        "It allows users to toggle between local ChromaDB and cloud Pinecone databases. "
        "The project was created by developer Swarnika."
    )
    with open(test_filepath, "w", encoding="utf-8") as f:
        f.write(test_content)
        
    try:
        # 1. Ingest
        with open(test_filepath, "rb") as f:
            files = {"document": (test_filename, f, "text/plain")}
            data = {"collectionName": collection_name}
            res = requests.post(f"{BASE_URL}/api/upload", files=files, data=data)
            
        print("Upload status code:", res.status_code)
        upload_json = res.json()
        print("Upload response:", upload_json)
        assert res.status_code == 200
        assert upload_json.get("result", {}).get("success") is True
        print("[OK] Ingestion test passed")
        
        # Give DB a second to persist
        time.sleep(1.5)

        # 2. Chat Query (Grounded)
        print("\nSending grounded query: 'Who created AskMyDoc?'")
        chat_data = {
            "query": "Who created AskMyDoc?",
            "namespaces": [collection_name],
            "history": []
        }
        res_chat = requests.post(f"{BASE_URL}/api/chat", json=chat_data)
        print("Chat status code:", res_chat.status_code)
        chat_json = res_chat.json()
        print("Chat response:", chat_json)
        assert res_chat.status_code == 200
        assert "Swarnika" in chat_json.get("answer", "")
        assert chat_json.get("confidence") in ["HIGH", "LOW"]
        print("[OK] Grounded query test passed")

        # 3. Evaluate Faithfulness
        print("\nSending evaluation request for the answer...")
        context_text = f"[Source 1 - {test_filename}]:\n{test_content}"
        eval_data = {
            "query": "Who created AskMyDoc?",
            "context": context_text,
            "answer": chat_json.get("answer")
        }
        res_eval = requests.post(f"{BASE_URL}/api/evaluate", json=eval_data)
        print("Eval status code:", res_eval.status_code)
        eval_json = res_eval.json()
        print("Eval response:", eval_json)
        assert res_eval.status_code == 200
        assert eval_json.get("score", 0.0) >= 0.8
        print("[OK] Faithfulness evaluation test passed")

        # 4. Chat Query (Out of Bounds)
        print("\nSending out of bounds query: 'What is the capital of Mars?'")
        chat_data_oob = {
            "query": "What is the capital of Mars?",
            "namespaces": [collection_name],
            "history": []
        }
        res_oob = requests.post(f"{BASE_URL}/api/chat", json=chat_data_oob)
        oob_json = res_oob.json()
        print("OOB response:", oob_json)
        assert oob_json.get("confidence") == "NOT_FOUND"
        assert "evidence" in oob_json.get("answer", "").lower()
        print("[OK] Out of bounds query test passed")

        # 5. Delete namespace
        print("\nDeleting namespace...")
        delete_data = {"namespace": collection_name}
        res_del = requests.post(f"{BASE_URL}/api/delete", json=delete_data)
        print("Delete response:", res_del.json())
        assert res_del.status_code == 200
        print("[OK] Delete namespace test passed")
        
        return True
    except Exception as e:
        print("[FAIL] Ingestion/Chat test failed:", e)
        return False
    finally:
        # Clean up temporary local test file
        if os.path.exists(test_filepath):
            os.remove(test_filepath)

if __name__ == "__main__":
    print("Running integration tests...")
    h_ok = test_health()
    if not h_ok:
        print("Server is not running or unreachable at", BASE_URL)
        sys.exit(1)
        
    chat_ok = test_upload_and_chat()
    if not chat_ok:
        sys.exit(1)
        
    print("\nALL TESTS PASSED SUCCESSFULLY!")
