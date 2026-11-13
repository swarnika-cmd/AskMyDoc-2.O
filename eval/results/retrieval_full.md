# Retrieval results (full)

> PROVISIONAL LATENCY: measured on a machine shared with other running jobs (container limited to 4 CPUs; an LLM run was active at the same time). Recall, MRR, context size and the other quality numbers are deterministic; the latency column will be replaced by an idle-machine re-run.

n = 130 answerable items. Hit = the supporting quote (whitespace-normalised) appears in one of the top-5 returned units. For P2 to P4 the unit is the parent text sent to the generator; for P1 it is the 800-character chunk. Mean context chars is the total size of the returned units per query.

| Pipeline | recall@5 | MRR@5 | mean context chars | latency p50 (ms) | latency p95 (ms) |
|---|---|---|---|---|---|
| P1 fixed 800/100 chunks + dense | 0.6308 | 0.4710 | 3220.0 | 5.8 | 7.6 |
| P2 parent-child + dense (current app design) | 0.8000 | 0.6163 | 7176.8 | 5.8 | 7.2 |
| P3 parent-child + BM25 + dense, RRF k=60 | 0.8692 | 0.7044 | 6895.1 | 12.4 | 18.4 |
| P4 P3 + cross-encoder rerank (bge-reranker-base, top-20) | 0.8692 | 0.7150 | 6679.7 | 1297.0 | 1771.4 |

Hits whose matching unit is not on the gold doc/page (the same text appears elsewhere): P1 0, P2 1, P3 1, P4 1.

Multi-hop items are scored on the primary quote only in recall@5. Secondary retrieval-only metric, all refs retrieved@5 (the primary quote and every extra_refs quote each appear in some top-5 unit):

| Pipeline | all refs retrieved@5 | n multi-hop items |
|---|---|---|
| P1 | 0.3667 | 30 |
| P2 | 0.5333 | 30 |
| P3 | 0.4667 | 30 |
| P4 | 0.5333 | 30 |

Recall ceiling (the quote is inside at least one unit of the gold page; items below the ceiling have a quote that straddles a unit boundary):

| Pipeline | ceiling | items not inside any unit |
|---|---|---|
| P1 | 0.9846 | q040, q132 |
| P2 | 1.0000 | none |
| P3 | 1.0000 | none |
| P4 | 1.0000 | none |

recall@5 by question type:

| Pipeline | factual (n=70) | multi-hop (n=30) | table (n=30) |
|---|---|---|---|
| P1 | 0.7000 | 0.5000 | 0.6000 |
| P2 | 0.8000 | 0.7333 | 0.8667 |
| P3 | 0.9000 | 0.7333 | 0.9333 |
| P4 | 0.9571 | 0.6667 | 0.8667 |

Limits: recall@5 by quote containment misses paraphrased support and favours larger units (see the context-size column). Vector search uses HNSW settings chosen for reproducible results; eval/results/checks.md shows they returned the same top-5 children as exact search and as Chroma's defaults for P2.

Latency method: time.perf_counter around pipeline.retrieve() only (query embedding, search, fusion, rerank, parent mapping); index build excluded. Warm-up: the first 3 questions are run once, untimed and discarded; then each question is run once and timed. Percentiles use linear interpolation. Single process, pipelines run one after another.

Environment: Python 3.11.18, torch 2.15.0+cpu, sentence-transformers 6.3.0, transformers 5.22.2, chromadb 1.5.9, rank-bm25 0.2.2, langchain-text-splitters 1.1.3, langchain-core 1.8.0, numpy 2.4.6, httpx 0.28.1; date 2026-11-13T00:59:51Z; none: run from a build copy (not a git checkout). CPU: Apple M4 Pro. Docker CPUs: visible 12, limit 4.0, torch threads 4.
Models: sentence-transformers/all-MiniLM-L6-v2 rev 1110a243fdf4706b3f48f1d95db1a4f5529b4d41; BAAI/bge-reranker-base rev 2cfc18c9415c912f9d8155881c133215df768a70.
