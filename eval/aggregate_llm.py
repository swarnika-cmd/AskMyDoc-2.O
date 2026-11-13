#!/usr/bin/env python3
"""Aggregate eval/results/raw/llm_<pipeline>.jsonl into eval/results/llm_<pipeline>[_pilot].json and .md.

  python eval/aggregate_llm.py --pipeline P2 --ids pilot     (-> llm_P2_pilot.*, labelled PILOT)
  python eval/aggregate_llm.py --pipeline P2 --ids all       (-> llm_P2.*; labelled PARTIAL when fewer than 150 items finished)

Refusal is the generator's no-evidence flag (the app's own behaviour); the judge's is_refusal is a cross-check. Faithfulness
is supported claims / total claims, averaged over judged non-refusals with at least one claim. Correctness share is over
answerable items that were judged plus no_answer items (counted incorrect); judge_parse_error and error items are excluded
from those denominators and reported separately. Every metric states its n.
"""
import argparse
import collections
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import llm_eval as le  # noqa: E402
import metrics  # noqa: E402
from run_llm import load_rows, select_ids  # noqa: E402
from validate_golden import load_golden  # noqa: E402

RESULTS = os.path.join(HERE, "results")


def row_status(r):
    """Rows written before statuses existed are 'ok' when completed."""
    return r.get("status") or ("ok" if r.get("completed") else "error")


def token_stats(rows, section):
    p = [r[section]["usage"]["prompt_tokens"] for r in rows if (r.get(section, {}).get("usage") or {}).get("prompt_tokens") is not None]
    c = [r[section]["usage"]["completion_tokens"] for r in rows if (r.get(section, {}).get("usage") or {}).get("completion_tokens") is not None]
    return {"n_calls": len(p), "prompt_tokens": {"mean": metrics.mean(p), "max": max(p) if p else 0, "total": sum(p)},
            "completion_tokens": {"mean": metrics.mean(c), "max": max(c) if c else 0, "total": sum(c)}}


def aggregate(rows, selected, prices, pipeline, label):
    by_status = collections.Counter(row_status(r) for r in rows)
    got = {r["id"] for r in rows}
    pending = [i for i in selected if i not in got]
    gen_rows = [r for r in rows if r.get("generator")]            # the generator answered (any status but a generator error)
    ok = [r for r in rows if row_status(r) == "ok"]               # judged and parsed
    ans_gen = [r for r in gen_rows if r["type"] != "unanswerable"]
    una_gen = [r for r in gen_rows if r["type"] == "unanswerable"]
    refused = lambda r: bool(r["generator"]["no_evidence"])       # noqa: E731  primary refusal signal: the app's own flag

    # correctness: answerable items that were judged, plus no_answer items (counted incorrect)
    corr_rows = [r for r in ans_gen if row_status(r) in ("ok", "no_answer")]
    corr = collections.Counter("incorrect" if row_status(r) == "no_answer" else r["judge"]["correctness"] for r in corr_rows)
    n_corr = len(corr_rows)
    faith_vals = [r["judge"]["faithfulness"] for r in ok if not refused(r) and r["judge"].get("faithfulness") is not None]
    agree = [r["judge"]["is_refusal"] == refused(r) for r in ok]

    gen_lat = [r["generator"]["latency_s"] for r in gen_rows if r["generator"].get("latency_s") is not None]
    ret_ms = [r["retrieval"]["retrieval_ms"] for r in gen_rows]
    e2e_ms = [r["retrieval"]["retrieval_ms"] + r["generator"]["latency_s"] * 1000.0 for r in gen_rows
              if r["generator"].get("latency_s") is not None]
    gen_cost = [le.cost_inr(prices, le.GENERATOR_MODEL, r["generator"]["usage"]) for r in gen_rows if r["generator"].get("usage")]
    judge_cost = [le.cost_inr(prices, le.JUDGE_MODEL, r["judge"]["usage"]) for r in rows if (r.get("judge") or {}).get("usage")]

    heads = []
    for r in rows:
        for sec in ("generator", "judge"):
            h = (r.get(sec) or {}).get("rate_headers") or {}
            if h:
                heads.append((sec, h))

    def seen(name, sec):
        return sorted({h[name] for s, h in heads if s == sec and name in h})

    def min_remaining(sec):
        vals = []
        for s, h in heads:
            if s == sec and "x-ratelimit-remaining-tokens" in h:
                try:
                    vals.append(int(h["x-ratelimit-remaining-tokens"]))
                except ValueError:
                    pass
        return min(vals) if vals else None

    events = collections.Counter()
    for r in rows:
        for sec in ("generator", "judge"):
            for e in (r.get(sec) or {}).get("retries") or []:
                events["%s:%s" % (sec, e["status"])] += 1
    non_stop = {sec: sorted("%s=%s" % (r["id"], r[sec].get("finish_reason")) for r in rows
                            if r.get(sec) and r[sec].get("finish_reason") != "stop") for sec in ("generator", "judge")}
    errors = [{"id": r["id"], **r.get("error", {})} for r in rows if row_status(r) == "error"]

    def share(num, den):
        return num / float(den) if den else None

    return {
        "label": label,
        "pipeline": pipeline,
        "generator_model": le.GENERATOR_MODEL,
        "judge_model": le.JUDGE_MODEL,
        "counts": {
            "selected": len(selected), "ok": by_status["ok"], "no_answer": by_status["no_answer"],
            "judge_parse_error": by_status["judge_parse_error"], "error": by_status["error"], "pending": len(pending),
            "answerable_answered": len(ans_gen), "unanswerable_answered": len(una_gen),
        },
        "n_selected": len(selected),
        "n_completed": by_status["ok"] + by_status["no_answer"] + by_status["judge_parse_error"],
        "errors": errors,
        "non_stop_finishes": {"generator": non_stop["generator"], "judge": non_stop["judge"],
                              "counts": {k: len(v) for k, v in non_stop.items()}},
        "correctness": {
            "correct": corr["correct"], "partial": corr["partial"], "incorrect": corr["incorrect"],
            "share_correct": share(corr["correct"], n_corr), "share_correct_or_partial": share(corr["correct"] + corr["partial"], n_corr),
            "n": n_corr,
            "note": "answerable items that were judged, plus no_answer items (counted incorrect); judge_parse_error and error items are excluded",
        },
        "faithfulness": {"mean": metrics.mean(faith_vals) if faith_vals else None, "n": len(faith_vals),
                         "note": "supported claims / total claims, over judged answers that are not refusals and have at least one claim"},
        "refusal": {
            "signal": "generator noEvidence flag (the app's own refusal behaviour)",
            "accuracy_on_unanswerable": share(sum(1 for r in una_gen if refused(r)), len(una_gen)),
            "n_unanswerable": len(una_gen),
            "false_refusal_rate_on_answerable": share(sum(1 for r in ans_gen if refused(r)), len(ans_gen)),
            "n_answerable": len(ans_gen),
            "judge_crosscheck": {"agreement": share(sum(agree), len(agree)), "n": len(agree),
                                 "disagreements": [r["id"] for r in ok if r["judge"]["is_refusal"] != refused(r)]},
        },
        "retrieval_hit_rate_on_answerable": {"value": share(sum(1 for r in ans_gen if r["retrieval"]["hit_rank"] is not None), len(ans_gen)),
                                             "n": len(ans_gen)},
        "tokens": {"generator": token_stats(gen_rows, "generator"), "judge": token_stats(rows, "judge")},
        "latency": {
            "retrieval_ms": {"p50": metrics.percentile(ret_ms, 50), "p95": metrics.percentile(ret_ms, 95)},
            "generator_groq_network_s": {"p50": metrics.percentile(gen_lat, 50), "p95": metrics.percentile(gen_lat, 95)},
            "end_to_end_ms": {"p50": metrics.percentile(e2e_ms, 50), "p95": metrics.percentile(e2e_ms, 95),
                              "definition": "retrieval + generator call; the generator part is Groq network time"},
            "n": len(e2e_ms),
        },
        "cost_inr_per_query": {
            "generation_mean": metrics.mean(gen_cost), "generation_n": len(gen_cost),
            "judge_mean": metrics.mean(judge_cost), "judge_n": len(judge_cost),
            "evaluation_total_mean": metrics.mean(gen_cost) + metrics.mean(judge_cost),
            "note": "a production query pays for generation only",
            "usd_to_inr": prices["usd_to_inr"], "prices_read_on": prices["prices_read_on"],
        },
        "rate_limit_headers_seen": {
            sec: {"limit_requests": seen("x-ratelimit-limit-requests", sec), "limit_tokens": seen("x-ratelimit-limit-tokens", sec),
                  "min_remaining_tokens_seen": min_remaining(sec)} for sec in ("generator", "judge")},
        "retry_events": dict(events),
        "served_model_note": "served_model is recorded per call but is not part of the cache key",
        "ids_completed": [r["id"] for r in rows if row_status(r) in ("ok", "no_answer", "judge_parse_error")],
        "ids_pending": pending,
    }


def write_md(path, a, runs):
    tag = a["label"]
    c, f, rf, t, lat, cost, k = (a["correctness"], a["faithfulness"], a["refusal"], a["tokens"], a["latency"],
                                 a["cost_inr_per_query"], a["counts"])
    pct = lambda x: "n/a" if x is None else "%.1f%%" % (100 * x)  # noqa: E731
    hit = a["retrieval_hit_rate_on_answerable"]
    xc = rf["judge_crosscheck"]
    lines = [
        "# LLM-judged results, pipeline %s (%s, %d of %d selected items processed)" % (a["pipeline"], tag, a["n_completed"], a["n_selected"]), "",
        "Generator %s, judge %s, reasoning_effort low, temperature 0. %s" % (
            a["generator_model"], a["judge_model"],
            "This is a PILOT on %d items; it measures tokens and plumbing and is not a result." % a["n_selected"]
            if tag.startswith("PILOT") else ("PARTIAL run: %d of %d selected items processed." % (a["n_completed"], a["n_selected"])
                                             if a["n_completed"] < a["n_selected"] else "")), "",
        "Item counts: selected %d; judged ok %d; no_answer %d; judge_parse_error %d; error %d; pending %d." % (
            k["selected"], k["ok"], k["no_answer"], k["judge_parse_error"], k["error"], k["pending"]),
        "Non-stop finish reasons: generator %d, judge %d%s." % (
            a["non_stop_finishes"]["counts"]["generator"], a["non_stop_finishes"]["counts"]["judge"],
            (" (%s)" % ", ".join(a["non_stop_finishes"]["generator"] + a["non_stop_finishes"]["judge"])) if (a["non_stop_finishes"]["generator"] or a["non_stop_finishes"]["judge"]) else ""),
        "", "| Metric | Value | n |", "|---|---|---|",
        "| Correct, share of answerable (judge; no_answer counted incorrect) | %s (%d correct, %d partial, %d incorrect) | %d |" % (
            pct(c["share_correct"]), c["correct"], c["partial"], c["incorrect"], c["n"]),
        "| Faithfulness, mean (judged non-refusals with claims) | %s | %d |" % ("n/a" if f["mean"] is None else "%.3f" % f["mean"], f["n"]),
        "| Refusal accuracy on unanswerable (generator no-evidence flag) | %s | %d |" % (pct(rf["accuracy_on_unanswerable"]), rf["n_unanswerable"]),
        "| False-refusal rate on answerable (generator no-evidence flag) | %s | %d |" % (pct(rf["false_refusal_rate_on_answerable"]), rf["n_answerable"]),
        "| Judge refusal cross-check: agreement with the flag | %s | %d |" % (pct(xc["agreement"]), xc["n"]),
        "| Retrieval hit rate on these answerable items | %s | %d |" % (pct(hit["value"]), hit["n"]),
        "", "| Tokens per call | prompt mean | prompt max | completion mean (incl. reasoning) | completion max | calls |", "|---|---|---|---|---|---|",
    ]
    for sec, name in (("generator", a["generator_model"]), ("judge", a["judge_model"])):
        s = t[sec]
        lines.append("| %s (%s) | %.1f | %d | %.1f | %d | %d |" % (sec, name, s["prompt_tokens"]["mean"], s["prompt_tokens"]["max"],
                                                                   s["completion_tokens"]["mean"], s["completion_tokens"]["max"], s["n_calls"]))
    lines += [
        "", "| Latency (n=%d) | p50 | p95 |" % lat["n"], "|---|---|---|",
        "| retrieval (ms) | %.1f | %.1f |" % (lat["retrieval_ms"]["p50"], lat["retrieval_ms"]["p95"]),
        "| generator call, Groq network time (s) | %.2f | %.2f |" % (lat["generator_groq_network_s"]["p50"], lat["generator_groq_network_s"]["p95"]),
        "| end to end: retrieval + generator call (ms) | %.0f | %.0f |" % (lat["end_to_end_ms"]["p50"], lat["end_to_end_ms"]["p95"]),
        "", "Retrieval latency here is from the LLM run's process, on a machine shared with other jobs: provisional. "
        "Generator latency is the duration of the successful HTTP call as recorded when the response was first cached.", "",
        "| Cost (INR per query) | mean | n |", "|---|---|---|",
        "| generation (what a production query pays) | %.5f | %d |" % (cost["generation_mean"], cost["generation_n"]),
        "| judge (evaluation only) | %.5f | %d |" % (cost["judge_mean"], cost["judge_n"]),
        "| generation + judge | %.5f | |" % cost["evaluation_total_mean"],
        "", "Prices: Groq list prices read %s, USD to INR %.2f (%s, %s)." % (
            cost["prices_read_on"], cost["usd_to_inr"]["rate"], cost["usd_to_inr"]["source"], cost["usd_to_inr"]["rate_date"]), "",
        "Rate-limit headers seen: %s" % json.dumps(a["rate_limit_headers_seen"]),
        "Retry events recorded on cached calls: %s" % (json.dumps(a["retry_events"]) or "{}"),
        "The served model name is recorded per call but is not part of the cache key.",
        "",
    ]
    if a["errors"]:
        lines.append("Errors: " + "; ".join("%s %s %s" % (e["id"], e.get("stage"), e.get("class")) for e in a["errors"]))
        lines.append("")
    if runs:
        lines.append("Run history (this pipeline): " + "; ".join(
            "%s wall %.0fs calls %s 429s %s stop %s" % (r["started_utc"], r["wall_seconds"], r["api_calls"], r["http_429"], r["stopped_reason"])
            for r in runs))
        lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--ids", default="all")
    ap.add_argument("--results-dir", default=RESULTS)
    args = ap.parse_args(argv)

    golden = load_golden(os.path.join(HERE, "golden.jsonl"))
    selected = select_ids(args.ids, golden)
    rows_by_id = load_rows(os.path.join(args.results_dir, "raw", "llm_%s.jsonl" % args.pipeline))
    rows = [rows_by_id[i] for i in selected if i in rows_by_id]
    with open(os.path.join(HERE, "prices.json"), encoding="utf-8") as f:
        prices = json.load(f)
    if args.ids == "pilot":
        label, stem = "PILOT (n=%d)" % len(selected), "llm_%s_pilot" % args.pipeline
    elif args.ids == "all":
        n_done = sum(1 for r in rows if r.get("completed"))
        label, stem = ("FULL" if n_done == len(selected) else "PARTIAL"), "llm_%s" % args.pipeline
    else:
        label, stem = args.ids.upper() if args.ids in ("ci",) else "SUBSET", "llm_%s_%s" % (args.pipeline, os.path.splitext(os.path.basename(args.ids))[0])

    a = aggregate(rows, selected, prices, args.pipeline, label)
    runs = []
    prog = os.path.join(args.results_dir, "llm_progress.json")
    if os.path.exists(prog):
        with open(prog, encoding="utf-8") as f:
            runs = [r for r in json.load(f).get("runs", []) if r["pipeline"] == args.pipeline and r["ids_spec"] == args.ids]
    a["runs"] = runs
    a["created_utc"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(os.path.join(args.results_dir, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(a, f, indent=2, ensure_ascii=False)
        f.write("\n")
    write_md(os.path.join(args.results_dir, stem + ".md"), a, runs)
    print("wrote %s.json/.md (%s, %d of %d complete)" % (stem, label, a["n_completed"], a["n_selected"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
