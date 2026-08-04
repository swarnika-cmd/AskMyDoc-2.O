import os
import sys
import time
import requests
import json

BASE_URL = os.getenv("BENCHMARK_BASE_URL", "http://127.0.0.1:8000")

def run_benchmark():
    print("=" * 60)
    print("      AskMyDoc Infrastructure & Performance Benchmark      ")
    print("=" * 60)
    
    # 1. Health & Config
    try:
        res = requests.get(f"{BASE_URL}/api/health")
        health = res.json()
        print(f"Status: {health.get('status')}")
        print(f"LLM Provider: {health.get('provider')}")
        print(f"Vector Store: {health.get('vector_store')}")
    except Exception as e:
        print(f"Error connecting to backend at {BASE_URL}: {e}")
        sys.exit(1)

    # 2. Ingest benchmark doc
    test_filename = "benchmark_sample.txt"
    test_filepath = os.path.join(os.getcwd(), test_filename)
    collection_name = f"benchmark_col_{int(time.time())}"
    
    sample_text = (
        "AskMyDoc Infrastructure Evaluation Report.\n"
        "Self-hosting LLMs like Llama 3.1 8B using Ollama allows full control over data privacy, zero API costs, and lower latency.\n"
        "ChromaDB provides fast local vector search with parent-child chunking.\n"
        "Metrics tracked include token throughput (TPS), VRAM allocation, retrieval latency, and request queue depth."
    )
    
    with open(test_filepath, "w", encoding="utf-8") as f:
        f.write(sample_text)
        
    try:
        print("\n--- 1. Ingestion Benchmark ---")
        t0 = time.time()
        with open(test_filepath, "rb") as f:
            files = {"document": (test_filename, f, "text/plain")}
            data = {"collectionName": collection_name}
            res = requests.post(f"{BASE_URL}/api/upload", files=files, data=data)
        ingest_time_ms = (time.time() - t0) * 1000
        print(f"Ingestion Status Code: {res.status_code}")
        print(f"Ingestion Latency: {ingest_time_ms:.2f} ms")
        
        # 3. Query Benchmark
        queries = [
            "What metrics are tracked in AskMyDoc?",
            "Why is Ollama used for self-hosting?",
            "What vector store is used for fast local search?"
        ]
        
        print("\n--- 2. End-to-End Query & Retrieval Benchmark ---")
        latencies = []
        for i, q in enumerate(queries, 1):
            t_start = time.time()
            chat_res = requests.post(f"{BASE_URL}/api/chat", json={
                "query": q,
                "namespaces": [collection_name]
            })
            t_elapsed = (time.time() - t_start) * 1000
            latencies.append(t_elapsed)
            print(f"Query {i}: '{q}' -> {t_elapsed:.2f} ms (Confidence: {chat_res.json().get('confidence')})")

        avg_latency = sum(latencies) / len(latencies)
        print(f"\nAverage End-to-End Query Latency: {avg_latency:.2f} ms")

        # 4. Fetch Metrics Endpoint
        print("\n--- 3. Live Metrics Endpoint (/api/metrics) ---")
        metrics_res = requests.get(f"{BASE_URL}/api/metrics")
        if metrics_res.status_code == 200:
            m = metrics_res.json()
            print(json.dumps(m, indent=2))
            
            print("\n" + "=" * 60)
            print("                       SUMMARY RESULT                      ")
            print("=" * 60)
            print(f"Model:                    {m.get('model_name')}")
            print(f"Quantization:             {m.get('quantization')}")
            print(f"Avg Retrieval Latency:    {m.get('avg_retrieval_latency_ms')} ms")
            print(f"Avg LLM Latency:          {m.get('avg_llm_latency_ms')} ms")
            print(f"Token Throughput:         {m.get('approx_token_throughput_tps')} tokens/sec")
            print(f"Memory Utilization:       {m.get('memory', {}).get('utilization_pct')}% ({m.get('memory', {}).get('type')})")
            print("=" * 60)
        else:
            print("Failed to fetch metrics")

        # Cleanup collection
        requests.post(f"{BASE_URL}/api/delete", json={"namespace": collection_name})
    finally:
        if os.path.exists(test_filepath):
            os.remove(test_filepath)

if __name__ == "__main__":
    run_benchmark()
