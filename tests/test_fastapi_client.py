import os
import time
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_api():
    print("Testing /api/health...")
    res = client.get("/api/health")
    print("Health response:", res.json())
    assert res.status_code == 200
    assert res.json().get("status") == "ok"
    print("[OK] Health test passed")

    print("\nTesting /api/metrics...")
    res_m = client.get("/api/metrics")
    print("Metrics response:", res_m.json())
    assert res_m.status_code == 200
    assert "active_requests_queue_depth" in res_m.json()
    print("[OK] Metrics test passed")

    print("\nTesting /api/upload and /api/chat...")
    test_filename = "temp_test_doc.txt"
    test_filepath = os.path.join(os.getcwd(), test_filename)
    collection_name = f"test_collection_{int(time.time())}"
    
    test_content = (
        "AskMyDoc is an advanced self-hosted document assistant built on Python, FastAPI, and LangChain. "
        "It supports parent-child chunking for high accuracy retrieval with local ChromaDB. "
        "The project was created by developer Swarnika."
    )
    with open(test_filepath, "w", encoding="utf-8") as f:
        f.write(test_content)

    try:
        with open(test_filepath, "rb") as f:
            response = client.post(
                "/api/upload",
                files={"document": (test_filename, f, "text/plain")},
                data={"collectionName": collection_name}
            )
        print("Upload status code:", response.status_code)
        print("Upload response:", response.json())
        assert response.status_code == 200
        assert response.json().get("result", {}).get("success") is True
        print("[OK] Ingestion test passed")

        # Chat test
        chat_data = {
            "query": "Who created AskMyDoc?",
            "namespaces": [collection_name],
            "history": []
        }
        res_chat = client.post("/api/chat", json=chat_data)
        print("Chat status code:", res_chat.status_code)
        chat_json = res_chat.json()
        print("Chat response:", chat_json)
        assert res_chat.status_code == 200
        assert "Swarnika" in chat_json.get("answer", "")
        print("[OK] Chat query test passed")

        # Evaluation test
        eval_data = {
            "query": "Who created AskMyDoc?",
            "context": f"[Source 1 - {test_filename}]:\n{test_content}",
            "answer": chat_json.get("answer")
        }
        res_eval = client.post("/api/evaluate", json=eval_data)
        print("Eval status code:", res_eval.status_code)
        print("Eval response:", res_eval.json())
        assert res_eval.status_code == 200
        assert res_eval.json().get("score", 0.0) >= 0.8
        print("[OK] Evaluation test passed")

        # Delete test
        res_del = client.post("/api/delete", json={"namespace": collection_name})
        print("Delete response:", res_del.json())
        assert res_del.status_code == 200
        print("[OK] Delete test passed")

    finally:
        if os.path.exists(test_filepath):
            os.remove(test_filepath)

if __name__ == "__main__":
    test_api()
