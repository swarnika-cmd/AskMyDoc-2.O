#!/usr/bin/env python3
"""Generate answers (openai/gpt-oss-20b) and judge them (openai/gpt-oss-120b) for one retrieval pipeline.

  GROQ_API_KEY=... python eval/run_llm.py --pipeline P2 --ids pilot [--max-calls N]

--ids: "pilot" (first 20 ids of ci_subset.txt, in file order), "ci" (all 30), "all" (150) or a file with one id per line.
Resumable: every API response is cached under eval/cache/, so a rerun repeats nothing. --max-calls caps the number of
uncached API calls per model in one invocation. On a daily-limit 429 the run stops cleanly and eval/results/llm_progress.json
records the completed ids and the resume command.

Generator and judge calls run concurrently on two threads with separate token-per-minute pacers.
"""
import argparse
import datetime
import json
import os
import queue
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import llm_eval as le  # noqa: E402
import llm_client as lc  # noqa: E402
import pipelines as pl  # noqa: E402
import metrics  # noqa: E402
from validate_golden import load_ci, load_golden  # noqa: E402

RESULTS = os.path.join(HERE, "results")
CACHE = os.path.join(HERE, "cache")
PILOT_SIZE = 20


def select_ids(spec, golden):
    ids_all = [g["id"] for g in golden]
    if spec in ("pilot", "ci"):
        ci = load_ci(os.path.join(HERE, "ci_subset.txt"))
        return ci[:PILOT_SIZE] if spec == "pilot" else ci
    if spec == "all":
        return ids_all
    with open(spec, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


def load_rows(path):
    rows = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[r["id"]] = r
    return rows


def write_rows(path, rows, order):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for i in order:
            if i in rows:
                f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def call_summary(rec):
    return {"usage": rec.get("usage"), "latency_s": rec.get("latency_s"), "finish_reason": rec.get("finish_reason"),
            "cache_key": rec["key"], "cached": rec["cached"], "retries": rec.get("retries"),
            "pacer_wait_s": rec.get("pacer_wait_s"), "rate_headers": rec.get("rate_headers"),
            "served_model": rec.get("served_model")}


def update_progress(path, pipeline, spec, selected, rows, stopped_reason, run_info, max_calls):
    prog = {"pipelines": {}, "runs": []}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            prog = json.load(f)
    done = [i for i in selected if rows.get(i, {}).get("completed")]
    pending = [i for i in selected if i not in set(done)]
    cmd = "make eval-llm PIPELINE=%s IDS=%s" % (pipeline, spec)
    if max_calls is not None:
        cmd += " MAX_CALLS=%d" % max_calls
    prog["pipelines"][pipeline] = {
        "ids_spec": spec, "n_selected": len(selected), "completed_ids": done, "pending_ids": pending,
        "stopped_reason": stopped_reason, "resume_command": cmd,
        "equivalent": "GROQ_API_KEY=... python eval/run_llm.py --pipeline %s --ids %s" % (pipeline, spec),
    }
    prog["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    prog["runs"] = (prog.get("runs", []) + [run_info])[-30:]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(prog, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True, choices=("P1", "P2", "P3", "P4"))
    ap.add_argument("--ids", default="pilot")
    ap.add_argument("--max-calls", type=int, default=None, help="max uncached API calls per model in this run")
    ap.add_argument("--tpm", type=int, default=lc.DEFAULT_TPM_BUDGET)
    ap.add_argument("--note", default=None, help="free text stored with this run in llm_progress.json")
    ap.add_argument("--cache-dir", default=CACHE)
    ap.add_argument("--results-dir", default=RESULTS)
    args = ap.parse_args(argv)

    api_key = os.environ.get("GROQ_API_KEY", "")
    golden = load_golden(os.path.join(HERE, "golden.jsonl"))
    by_id = {g["id"]: g for g in golden}
    order = [g["id"] for g in golden]
    selected = select_ids(args.ids, golden)
    items = [by_id[i] for i in selected]
    os.makedirs(os.path.join(args.results_dir, "raw"), exist_ok=True)
    raw_path = os.path.join(args.results_dir, "raw", "llm_%s.jsonl" % args.pipeline)
    prog_path = os.path.join(args.results_dir, "llm_progress.json")
    with open(os.path.join(HERE, "prices.json"), encoding="utf-8") as f:
        json.load(f)  # fail early if the price file is missing or invalid

    started = time.time()
    started_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print("pipeline %s, %d items (%s)" % (args.pipeline, len(items), args.ids), flush=True)

    # Retrieval for every selected item, timed per query (warm-up policy in pipelines.run_queries).
    pages = pl.load_pages(os.path.join(HERE, "corpus", "pages.jsonl"))
    embedder = pl.Embedder()
    reranker = pl.Reranker() if args.pipeline == "P4" else None
    pipe = pl.build_pipeline(args.pipeline, pages, embedder, reranker)
    retrieved = pl.run_queries(pipe, [g["question"] for g in items])

    rows = load_rows(raw_path)
    lock = threading.Lock()
    stop = threading.Event()
    stop_reason = []
    q = queue.Queue()
    client = lc.GroqClient(api_key, args.cache_dir, tpm_budget=args.tpm, max_calls=args.max_calls,
                           log=lambda m: print("  [client] " + m, flush=True))

    def cached_or_call(model, messages, max_tokens, response_format=None):
        if not api_key and client.cached(model, messages, {
                "temperature": 0, "reasoning_effort": "low", "max_completion_tokens": max_tokens,
                **({"response_format": response_format} if response_format else {})}) is None:
            raise lc.LLMError("GROQ_API_KEY is not set and the response is not cached")
        return client.chat(model, messages, max_tokens, response_format=response_format)

    def halt(reason):
        with lock:
            if not stop_reason:
                stop_reason.append(reason)
        stop.set()

    def save_row(row):
        """Record a row unless an earlier run already completed this id (an error must not undo a result)."""
        with lock:
            if rows.get(row["id"], {}).get("completed") and not row.get("completed"):
                return
            rows[row["id"]] = row
            write_rows(raw_path, rows, order)

    def error_row(row, stage, exc):
        row["status"] = "error"
        row["completed"] = False
        row["error"] = {"stage": stage, "class": type(exc).__name__, "message": str(exc)[:300]}
        print("  %s %s error (%s): %s" % (row["id"], stage, type(exc).__name__, str(exc)[:200]), flush=True)
        save_row(row)

    def generator_worker():
        try:
            for item, (units, ms) in zip(items, retrieved):
                if stop.is_set():
                    break
                hit = None if item["type"] == "unanswerable" else metrics.hit_rank([u["text"] for u in units], item["supporting_quote"])
                row = {"id": item["id"], "type": item["type"], "pipeline": args.pipeline, "question": item["question"],
                       "reference_answer": item["answer"],
                       "retrieval": {"ranked_unit_ids": [u["id"] for u in units], "hit_rank": hit,
                                     "context_chars": sum(len(u["text"]) for u in units), "retrieval_ms": round(ms, 3)}}
                try:
                    context = le.render_context(units)
                    msgs = le.generator_messages(item["question"], context)
                    rec = cached_or_call(le.GENERATOR_MODEL, msgs, le.GENERATOR_MAX_TOKENS)
                except lc.DailyLimitReached as e:
                    halt("daily_limit:%s:%s" % (e.model, e.kind))
                    break
                except lc.CallBudgetExhausted:
                    halt("max_calls:%s" % le.GENERATOR_MODEL)
                    break
                except Exception as e:  # noqa: BLE001  (recorded below, never dropped)
                    error_row(row, "generator", e)
                    continue
                parsed = le.parse_generator(rec["content"])
                answer = parsed["answer"]
                gen = dict(call_summary(rec), answer=answer, no_evidence=parsed["no_evidence"], parse_ok=parsed["parse_ok"])
                if parsed["no_evidence"]:
                    # As app/retrieve.py does: a no-evidence reply is shown to the user as the fixed refusal text.
                    gen["raw_answer"] = answer
                    answer = gen["answer"] = le.REFUSAL_TEXT
                row["generator"] = gen
                print("  %s answered: prompt=%s completion=%s latency=%.1fs cached=%s" % (
                    item["id"], (rec.get("usage") or {}).get("prompt_tokens"), (rec.get("usage") or {}).get("completion_tokens"),
                    rec.get("latency_s") or 0, rec["cached"]), flush=True)
                if not answer.strip():
                    # Empty reply that is not a refusal: counted as incorrect, not as a refusal; no judge call.
                    row["status"], row["completed"] = "no_answer", True
                    save_row(row)
                    continue
                q.put((item, context, answer, row))
        except Exception as e:  # noqa: BLE001
            print("  generator worker failed: %s: %s" % (type(e).__name__, e), flush=True)
            halt("worker_error:generator:%s" % type(e).__name__)
        finally:
            q.put(None)

    def judge_worker():
        while True:
            job = q.get()
            if job is None:
                return
            item, context, answer, row = job
            try:
                if stop.is_set() and stop_reason and stop_reason[0].startswith("daily_limit:" + le.JUDGE_MODEL):
                    continue
                msgs = le.judge_messages(item, answer, context)
                try:
                    rec = cached_or_call(le.JUDGE_MODEL, msgs, le.JUDGE_MAX_TOKENS, {"type": "json_object"})
                except lc.DailyLimitReached as e:
                    halt("daily_limit:%s:%s" % (e.model, e.kind))
                    continue
                except lc.CallBudgetExhausted:
                    halt("max_calls:%s" % le.JUDGE_MODEL)
                    continue
                except Exception as e:  # noqa: BLE001
                    error_row(row, "judge", e)
                    continue
                parsed = le.parse_judge(rec["content"])
                judge = dict(call_summary(rec), parse_ok=parsed["parse_ok"])
                if parsed["parse_ok"]:
                    judge.update(correctness=parsed["correctness"], is_refusal=parsed["is_refusal"], reason=parsed["reason"],
                                 claims=parsed["claims"], faithfulness=le.faithfulness(parsed["claims"]))
                    row["status"] = "ok"
                else:
                    judge["raw_reply"] = (rec["content"] or "")[:500]
                    row["status"] = "judge_parse_error"
                row["judge"] = judge
                row["completed"] = True  # a parse error is deterministic (the reply is cached), so a rerun would not change it
                save_row(row)
                print("  %s judged: %s refusal=%s prompt=%s completion=%s latency=%.1fs cached=%s" % (
                    item["id"], judge.get("correctness", "PARSE_ERROR"), judge.get("is_refusal"),
                    (rec.get("usage") or {}).get("prompt_tokens"), (rec.get("usage") or {}).get("completion_tokens"),
                    rec.get("latency_s") or 0, rec["cached"]), flush=True)
            except Exception as e:  # noqa: BLE001
                error_row(row, "judge", e)

    threads = [threading.Thread(target=generator_worker, name="generator"), threading.Thread(target=judge_worker, name="judge")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    wall = time.time() - started
    reason = stop_reason[0] if stop_reason else None
    done = [i for i in selected if rows.get(i, {}).get("completed")]
    if reason is None:
        reason = "completed" if len(done) == len(selected) else "errors_or_unparsed_items"
    errored = [i for i in selected if rows.get(i, {}).get("status") == "error"]
    if errored:
        print("items with recorded errors: %s" % ", ".join(errored), flush=True)
    run_info = {"pipeline": args.pipeline, "ids_spec": args.ids, "started_utc": started_utc, "wall_seconds": round(wall, 1),
                "api_calls": dict(client.api_calls), "http_429": dict(client.n_429), "completed_this_invocation_total": len(done),
                "stopped_reason": reason, "max_calls": args.max_calls, "tpm_budget": args.tpm, "note": args.note}
    update_progress(prog_path, args.pipeline, args.ids, selected, rows, reason, run_info, args.max_calls)
    print("done: %d/%d complete, stop=%s, wall=%.1fs, api_calls=%s, 429s=%s" % (
        len(done), len(selected), reason, wall, dict(client.api_calls), dict(client.n_429)), flush=True)
    return 0 if reason == "completed" or reason.startswith(("daily_limit", "max_calls")) else 1


if __name__ == "__main__":
    sys.exit(main())
