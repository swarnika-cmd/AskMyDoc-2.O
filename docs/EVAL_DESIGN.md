# AskMyDoc V2 evaluation design

Status: approved 2026-11-13.

## Goal

Measure how well AskMyDoc retrieves and answers over messy public PDFs, and compare four retrieval pipelines on the same corpus and the same golden set. Every number in the README must come from a committed script and its raw output.

## Corpus

- 8 to 10 public PDFs, for example RBI and SEBI circulars or a public university handbook. At least 2 are scanned or table-heavy.
- `eval/corpus.json` records, per document: id, title, source URL, licence or public status, SHA-256 of the PDF, page count and extraction method.
- PDFs are not committed. `eval/fetch_corpus.py` downloads them and checks each SHA-256.
- `eval/extract.py` produces `eval/corpus/pages.jsonl` (one row per doc and page: `doc`, `page`, `text`, `method`). Text PDFs use pypdf, the same library the app's `PyPDFLoader` uses. Scanned pages use Tesseract OCR in Docker. The extracted text is committed, because OCR output varies between Tesseract versions and CI must not need OCR.
- Every pipeline indexes `pages.jsonl`, and the golden set validates against it. So retrieval hits and quote checks use the same text.

## Golden set

- 150 items in `eval/golden.jsonl`. Fields: `id`, `question`, `answer`, `doc`, `page`, `supporting_quote`, `type`.
- Types: `factual`, `multi-hop`, `table`, `unanswerable`. 20 items are `unanswerable`. They have an empty `supporting_quote` and `doc`/`page` set to null.
- A quote is 30 to 300 characters from a single page. `eval/validate_golden.py` checks that every quote appears verbatim in the text of `doc` at `page`, after whitespace normalisation (runs of whitespace collapse to one space, then strip). Any failure exits non-zero, and CI runs this script.
- Multi-hop items cite their primary page in `page`. They may add `extra_refs: [{doc, page, supporting_quote}]`, which are validated the same way.
- Build method, stated in the README: questions were written from the source documents, and every answer is backed by a verbatim quote that `validate_golden.py` checks.

## Pipelines

All pipelines use `sentence-transformers/all-MiniLM-L6-v2` embeddings (the app's model) and ChromaDB in memory.

| # | Name | Retrieval unit returned to the generator |
|---|---|---|
| P1 | fixed-size chunks + dense | 800-character chunks, 100 overlap |
| P2 | parent-child + dense (current app design) | search 400/100 children, return 3,000/300 parents, deduplicated |
| P3 | parent-child + BM25 + dense, fused with RRF (k=60) | as P2 |
| P4 | P3 + cross-encoder rerank (`BAAI/bge-reranker-base`, top-20 candidates) | as P2 |

The P2 to P4 chunk sizes are imported from the app code, so the eval measures what ships.

## Metrics

- **recall@5:** the fraction of answerable items where one of the top 5 retrieved units contains the supporting quote (whitespace-normalised). For P2 to P4 the unit is the parent text sent to the generator. Parents are about 4 times larger than P1 chunks, which favours them, so the results table also reports the mean context characters per query.
- **MRR:** the mean of 1/rank of the first unit containing the quote, or 0 if none of the top 5 does.
- **Answer correctness:** the judge compares the model's answer with the reference answer: correct, partial or incorrect. The README reports the share rated correct.
- **Faithfulness:** the judge checks each claim in the answer against the retrieved context. Score = supported claims / total claims.
- **Refusal accuracy:** on the 20 unanswerable items, the share where the model declines. Also reported: the false-refusal rate on answerable items.
- **Latency:** p50 and p95 of retrieval time (all pipelines, measured locally) and end-to-end time including the generator call (pipelines that generate). Generator latency is Groq network time and is labelled as such.
- **₹ per query:** (input tokens × input price + output tokens × output price), using token counts returned by the API. Prices are Groq's published list prices per 1M tokens on 2026-11-13: gpt-oss-20b $0.075 input / $0.30 output, gpt-oss-120b $0.15 input / $0.60 output, from https://console.groq.com/docs/models. The USD to INR rate and its date are recorded in `eval/prices.json`. Generation and judge cost are reported separately; a production query pays only for generation.

## Models

- Generator: `openai/gpt-oss-20b`. Judge: `openai/gpt-oss-120b`. These are different models, and the judge is the larger one. Both use `reasoning_effort: low` and temperature 0. The judge prompt is committed at `eval/prompts/judge.txt`.
- These are the app's Groq fallback model (now `openai/gpt-oss-20b`) and a stronger judge. The local Ollama path is not evaluated.

## Scope under the rate limits

Groq free tier, per model: 30 requests/minute, 1,000 requests/day, 8,000 tokens/minute, 200,000 tokens/day.

- Retrieval metrics for all four pipelines need no LLM calls and run fully.
- Generation and judging run for two pipelines only: P2 (the current design) and whichever of P1, P3 or P4 has the best recall@5. That is 150 × 2 = 300 generator calls and 300 judge calls.
- A P2 to P4 generator call carries up to 5 parents of about 3,000 characters, so roughly 4,000 tokens. At 200K tokens/day this is about 50 calls per model per day, so the full LLM-judged run takes several days. The 20-question pilot measures real tokens per call. The plan is fixed after the pilot, and the decision is logged.
- Every response is cached on disk keyed by sha256(model + prompt), so reruns and resumes are free. The client paces calls against a token-per-minute budget, backs off exponentially on 429 responses, and stops cleanly on a daily-limit 429. It writes `eval/results/llm_progress.json` with the resume point.
- The golden set is never reduced, and models are never swapped to fit quota. A partial run is reported as partial, with n stated.

## CI

- On every PR: `validate_golden.py` and the retrieval eval on a fixed 30-item subset (`eval/ci_subset.txt`), for all four pipelines. The job fails if any pipeline's recall@5 drops more than 3 points below `eval/results/baseline_ci.json`. No API key is needed. Hugging Face models are cached with actions/cache, keyed on the model revision.
- LLM-judged eval: runs only when a `GROQ_API_KEY` repo secret exists, and is skipped otherwise.

## Runtime judge fix

The app's per-answer judge currently scores against context sent by the client. The fix: the server returns chunk ids with each answer. The judge endpoint accepts those ids, fetches the chunk text from the store, and ignores any client-supplied context.

## Known limitations (README)

- 150 items over 8 documents, single-pass construction (not double-annotated): confidence intervals are wide. The README reports n and does not claim small differences.
- An LLM judge has its own bias. The 20 spot-checked item ids are listed for re-checking by hand.
- recall@5 by quote containment misses paraphrased support and favours larger units. The context-size column makes the second point visible.
