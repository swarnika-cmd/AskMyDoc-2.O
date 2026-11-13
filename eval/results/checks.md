# Checks behind README statements

## HNSW settings vs exact search

P2 child search, top-5 children, 130 answerable questions. Evaluation configuration: {"num_threads": 1, "max_neighbors": 32, "ef_construction": 400, "ef_search": 400}.

| Comparison | identical top-5 sets | identical ordered lists |
|---|---|---|
| eval vs exact | 130 of 130 | 130 of 130 |
| default vs exact | 130 of 130 | 130 of 130 |
| default vs eval | 130 of 130 | 130 of 130 |

## App refusal gate (best child similarity)

Across all 150 questions (including the unanswerable ones), the best child cosine similarity has minimum 0.5111 and mean 0.6774. Questions below 0.2 (the app's NOT_FOUND threshold): 0. Below 0.4: 0. Lowest five: q017 0.5111 (unanswerable), q030 0.5160 (unanswerable), q077 0.5203 (factual), q014 0.5213 (unanswerable), q045 0.5289 (multi-hop).

Environment: none: run from a build copy (not a git checkout).
