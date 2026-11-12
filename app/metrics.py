import os
import re
import time
import threading
from typing import Dict, Any, List
from app.llm import last_served, groq_model_name, DEFAULT_OLLAMA_MODEL

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
        """Memory of this app process: torch GPU allocation if CUDA is available, else process RSS. Not Ollama's memory."""
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
            proc = psutil.Process()
            rss = proc.memory_info().rss
            total = psutil.virtual_memory().total
            return {
                "type": "PROCESS_RSS",
                "used_mb": round(rss / (1024 * 1024), 2),
                "total_mb": round(total / (1024 * 1024), 2),
                "utilization_pct": round(rss / total * 100, 2)
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

            configured_provider = os.getenv("LLM_PROVIDER", "ollama").lower()
            served = last_served()
            # Report what actually served the last request; before any request, the configured provider.
            provider = served["provider"] or configured_provider
            if served["model"]:
                model = served["model"]
            elif configured_provider == "ollama":
                model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
            else:
                model = groq_model_name()
            model_label = f"{model} (Groq)" if provider == "groq" else model

            # Quantization is only known if an Ollama tag states it (e.g. ...-q4_K_M, ...-q8_0, ...-fp16).
            quantization = "unknown"
            if provider == "ollama":
                m = re.search(r"(q\d(?:_[a-z0-9]+)*|fp16|f16|bf16)$", model.lower())
                if m:
                    quantization = m.group(1).upper()

            return {
                "provider": provider,
                "configured_provider": configured_provider,
                "model_name": model_label,
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
