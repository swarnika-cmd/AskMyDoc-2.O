# LLM-judged results, pipeline P2 (PARTIAL, 114 of 150 selected items processed)

Generator openai/gpt-oss-20b, judge openai/gpt-oss-120b, reasoning_effort low, temperature 0. PARTIAL run: 114 of 150 selected items processed.

Item counts: selected 150; judged ok 113; no_answer 1; judge_parse_error 0; error 0; pending 36.
Non-stop finish reasons: generator 1, judge 0 (q083=length).

| Metric | Value | n |
|---|---|---|
| Correct, share of answerable (judge; no_answer counted incorrect) | 78.4% (76 correct, 1 partial, 20 incorrect) | 97 |
| Faithfulness, mean (judged non-refusals with claims) | 0.972 | 83 |
| Refusal accuracy on unanswerable (generator no-evidence flag) | 100.0% | 17 |
| False-refusal rate on answerable (generator no-evidence flag) | 13.4% | 97 |
| Judge refusal cross-check: agreement with the flag | 100.0% | 113 |
| Retrieval hit rate on these answerable items | 81.4% | 97 |

| Tokens per call | prompt mean | prompt max | completion mean (incl. reasoning) | completion max | calls |
|---|---|---|---|---|---|
| generator (openai/gpt-oss-20b) | 2397.3 | 5968 | 108.0 | 1024 | 114 |
| judge (openai/gpt-oss-120b) | 2718.0 | 6279 | 166.8 | 503 | 113 |

| Latency (n=114) | p50 | p95 |
|---|---|---|
| retrieval (ms) | 8.8 | 10.7 |
| generator call, Groq network time (s) | 0.80 | 1.99 |
| end to end: retrieval + generator call (ms) | 812 | 1994 |

Retrieval latency here is from the LLM run's process, on a machine shared with other jobs: provisional. Generator latency is the duration of the successful HTTP call as recorded when the response was first cached.

| Cost (INR per query) | mean | n |
|---|---|---|
| generation (what a production query pays) | 0.01996 | 114 |
| judge (evaluation only) | 0.04775 | 113 |
| generation + judge | 0.06771 | |

Prices: Groq list prices read 2026-11-13, USD to INR 94.04 (frankfurter, 2026-11-12).

Rate-limit headers seen: {"generator": {"limit_requests": ["1000"], "limit_tokens": ["8000"], "min_remaining_tokens_seen": 23}, "judge": {"limit_requests": ["1000"], "limit_tokens": ["8000"], "min_remaining_tokens_seen": 44}}
Retry events recorded on cached calls: {"judge:429": 3, "generator:429": 9}
The served model name is recorded per call but is not part of the cache key.

Run history (this pipeline): 2026-11-13T00:37:59Z wall 1131s calls {'openai/gpt-oss-20b': 48, 'openai/gpt-oss-120b': 40} 429s {'openai/gpt-oss-20b': 2, 'openai/gpt-oss-120b': 1} stop daily_limit:openai/gpt-oss-120b:tokens per day; 2026-11-13T17:37:40Z wall 1464s calls {'openai/gpt-oss-20b': 67, 'openai/gpt-oss-120b': 55} 429s {'openai/gpt-oss-20b': 7, 'openai/gpt-oss-120b': 2} stop daily_limit:openai/gpt-oss-120b:tokens per day
