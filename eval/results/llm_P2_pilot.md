# LLM-judged results, pipeline P2 (PILOT (n=20), n=20 of 20 selected)

Generator openai/gpt-oss-20b, judge openai/gpt-oss-120b, reasoning_effort low, temperature 0. This is a PILOT on 20 items; it measures tokens and plumbing and is not a result.

| Metric | Value | n |
|---|---|---|
| Correctness: correct (answerable) | 93.8% (15 correct, 0 partial, 1 incorrect) | 16 |
| Faithfulness, mean | 1.000 | 15 |
| Refusal accuracy on unanswerable | 100.0% | 4 |
| False-refusal rate on answerable | 6.2% | 16 |
| Retrieval hit rate on these answerable items | 93.8% | 16 |
| Judge vs generator refusal-flag agreement | 100.0% | 20 |

| Tokens per call | prompt mean | prompt max | completion mean (incl. reasoning) | completion max | calls |
|---|---|---|---|---|---|
| generator (openai/gpt-oss-20b) | 2412.8 | 5530 | 101.1 | 172 | 20 |
| judge (openai/gpt-oss-120b) | 2743.9 | 5841 | 145.5 | 213 | 20 |

| Latency | p50 | p95 |
|---|---|---|
| retrieval (ms) | 5.9 | 6.5 |
| generator call, Groq network time (s) | 0.79 | 4.04 |
| end to end: retrieval + generator call (ms) | 796 | 4046 |

Retrieval latency here is from the LLM run's process, on a machine shared with other jobs: provisional. Generator latency is the duration of the successful HTTP call as recorded when the response was first cached.

| Cost (INR per query) | mean |
|---|---|
| generation (what a production query pays) | 0.01987 |
| judge (evaluation only) | 0.04692 |
| generation + judge | 0.06679 |

Prices: Groq list prices read 2026-11-13, USD to INR 94.04 (frankfurter, 2026-11-12).

Rate-limit headers seen: {"generator": {"limit_requests": ["1000"], "limit_tokens": ["8000"], "min_remaining_tokens_seen": 81}, "judge": {"limit_requests": ["1000"], "limit_tokens": ["8000"], "min_remaining_tokens_seen": 44}}
Retry events recorded on cached calls: {"judge:429": 2, "generator:429": 1}

Run history (this pipeline): 2026-11-13T00:06:14Z wall 544s calls {'openai/gpt-oss-20b': 20, 'openai/gpt-oss-120b': 20} 429s {'openai/gpt-oss-120b': 2, 'openai/gpt-oss-20b': 1} stop completed; 2026-11-13T00:15:57Z wall 41s calls {} 429s {} stop completed; 2026-11-13T00:21:12Z wall 471s calls {'openai/gpt-oss-120b': 20} 429s {'openai/gpt-oss-120b': 2} stop completed
