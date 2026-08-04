import os
import time
import threading
from typing import Dict, Any, List

class MetricsTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self.active_requests = 0
        self.total_requests = 0
        self.llm_latencies: List[float] = []
        self.retrieval_latencies: List[float] = []
        self.tokens_generated = 0
        self.total_llm_time = 0.0

    def start_request(self):
        with self._lock:
            self.active_requests += 1
            self.total_requests += 1

    def end_request(self):
        with self._lock:
            self.active_requests = max(0, self.active_requests - 1)

    def record_retrieval_latency(self, latency_ms: float):
        with self._lock:
            self.retrieval_latencies.append(latency_ms)
            if len(self.retrieval_latencies) > 100:
                self.retrieval_latencies.pop(0)

    def record_llm_inference(self, latency_ms: float, approx_tokens: int = 0):
        with self._lock:
            self.llm_latencies.append(latency_ms)
            if len(self.llm_latencies) > 100:
                self.llm_latencies.pop(0)
            self.tokens_generated += approx_tokens
            self.total_llm_time += (latency_ms / 1000.0)

    def get_memory_info() -> Dict[str, Any]:
        """Attempt to fetch GPU memory; fallback to system RAM via psutil."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
                total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                return {
                    "type": "GPU",
                    "device": torch.cuda.get_device_name(0),
                    "used_mb": round(allocated, 2),
                    "total_mb": round(total, 2),
                    "utilization_pct": round((allocated / total) * 100, 2)
                }
        except ImportError:
            pass

        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "type": "CPU_RAM",
                "used_mb": round(mem.used / (1024 * 1024), 2),
                "total_mb": round(mem.total / (1024 * 1024), 2),
                "utilization_pct": mem.percent
            }
        except Exception:
            return {"type": "Unknown", "used_mb": 0, "total_mb": 0, "utilization_pct": 0}

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            avg_llm_latency = (
                round(sum(self.llm_latencies) / len(self.llm_latencies), 2)
                if self.llm_latencies else 0.0
            )
            avg_retrieval_latency = (
                round(sum(self.retrieval_latencies) / len(self.retrieval_latencies), 2)
                if self.retrieval_latencies else 0.0
            )
            tps = (
                round(self.tokens_generated / self.total_llm_time, 2)
                if self.total_llm_time > 0 else 0.0
            )

            provider = os.getenv("LLM_PROVIDER", "ollama").lower()
            model = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M") if provider == "ollama" else "llama-3.1-8b-instant (Groq)"
            
            # Infer quantization from model tag if present
            quantization = "Q4_K_M"
            if "q8" in model.lower() or "q8_0" in model.lower():
                quantization = "Q8_0"
            elif "q4" in model.lower():
                quantization = "Q4_K_M"
            elif "f16" in model.lower() or "fp16" in model.lower():
                quantization = "FP16"
            elif provider == "groq":
                quantization = "Cloud API (FP16)"

            return {
                "provider": provider,
                "model_name": model,
                "quantization": quantization,
                "active_requests_queue_depth": self.active_requests,
                "total_requests_handled": self.total_requests,
                "avg_llm_latency_ms": avg_llm_latency,
                "avg_retrieval_latency_ms": avg_retrieval_latency,
                "approx_token_throughput_tps": tps,
                "total_tokens_generated": self.tokens_generated,
                "memory": MetricsTracker.get_memory_info()
            }

# Singleton metrics instance
metrics = MetricsTracker()
