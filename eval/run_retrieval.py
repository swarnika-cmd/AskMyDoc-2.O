#!/usr/bin/env python3
"""Retrieval evaluation for P1 to P4 over the answerable golden items.

  python eval/run_retrieval.py --subset full   -> eval/results/retrieval_full.json (+ .md), raw/retrieval_<P>.jsonl
  python eval/run_retrieval.py --subset ci     -> eval/results/retrieval_ci.json (+ .md), raw/retrieval_ci_<P>.jsonl

No LLM calls. Latency is measured per query with time.perf_counter around retrieval only (see pipelines.run_queries).
"""
import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import metrics  # noqa: E402
import pipelines as pl  # noqa: E402
import envinfo  # noqa: E402
from validate_golden import load_ci, load_golden, norm  # noqa: E402

RESULTS = os.path.join(HERE, "results")


def select_items(subset):
    golden = load_golden(os.path.join(HERE, "golden.jsonl"))
    if subset == "ci":
        ids = load_ci(os.path.join(HERE, "ci_subset.txt"))
        by_id = {g["id"]: g for g in golden}
        golden = [by_id[i] for i in ids]
    return [g for g in golden if g["type"] != "unanswerable"]


def unit_pool(pipe):
    """All retrieval units of a pipeline: P1 chunks, or parents for P2 to P4."""
    return list(pipe.units.values()) if hasattr(pipe, "units") else list(pipe.parents.values())


def ceiling(pool, items):
    """Recall ceiling: the share of items whose quote is inside at least one unit of the gold page.

    Items that fall short have a quote that straddles a unit boundary, so no pipeline could return it whole."""
    by_page = {}
    for u in pool:
        by_page.setdefault((u["doc"], u["page"]), []).append(norm(u["text"]))
    missing = [g["id"] for g in items
               if not any(norm(g["supporting_quote"]) in t for t in by_page.get((g["doc"], g["page"]), []))]
    return {"recall_ceiling": 1.0 - len(missing) / float(len(items)), "n": len(items), "items_not_in_any_unit": missing}


def evaluate_pipeline(name, pages, items, embedder, reranker):
    t0 = time.perf_counter()
    pipe = pl.build_pipeline(name, pages, embedder, reranker)
    build_s = time.perf_counter() - t0
    results = pl.run_queries(pipe, [g["question"] for g in items])

    ranks, chars, lats, raw = [], [], [], []
    by_type = collections.defaultdict(lambda: ([], [], []))
    off_gold = 0
    multi_all = []  # multi-hop items: True when every cited quote (primary and extra refs) is in the top-5 units
    for g, (units, ms) in zip(items, results):
        texts = [u["text"] for u in units]
        rank = metrics.hit_rank(texts, g["supporting_quote"])
        on_gold = None if rank is None else (units[rank - 1]["doc"], units[rank - 1]["page"]) == (g["doc"], g["page"])
        off_gold += 1 if on_gold is False else 0
        c = sum(len(t) for t in texts)
        ranks.append(rank)
        chars.append(c)
        lats.append(ms)
        for acc, v in zip(by_type[g["type"]], (rank, c, ms)):
            acc.append(v)
        row = {"id": g["id"], "type": g["type"], "ranked_unit_ids": [u["id"] for u in units],
               "hit_rank": rank, "hit_on_gold_page": on_gold, "context_chars": c, "retrieval_ms": round(ms, 3)}
        if g["type"] == "multi-hop" and g.get("extra_refs"):
            ref_ranks = [rank] + [metrics.hit_rank(texts, r["supporting_quote"]) for r in g["extra_refs"]]
            row["ref_hit_ranks"] = ref_ranks
            multi_all.append(all(r is not None for r in ref_ranks))
        raw.append(row)
    out = {
        "name": pl.PIPELINE_NAMES[name],
        "metrics": metrics.summarise(ranks, chars, lats),
        "multi_hop_all_refs_recall_at_5": {"value": (sum(multi_all) / float(len(multi_all))) if multi_all else None, "n": len(multi_all)},
        "ceiling": ceiling(unit_pool(pipe), items),
        "hits_off_gold_page": off_gold,
        "by_type": {t: metrics.summarise(*acc) for t, acc in sorted(by_type.items())},
        "index": dict(pipe.stats, build_seconds=round(build_s, 2)),
    }
    return out, raw


def write_md(path, data):
    m = data["meta"]
    lines = ["# Retrieval results (%s)" % m["subset"], ""]
    if m.get("note"):
        lines += ["> " + m["note"], ""]
    lines += [
        "n = %d answerable items. Hit = the supporting quote (whitespace-normalised) appears in one of the top-5 "
        "returned units. For P2 to P4 the unit is the parent text sent to the generator; for P1 it is the 800-character "
        "chunk. Mean context chars is the total size of the returned units per query." % data["n_items"],
        "",
        "| Pipeline | recall@5 | MRR@5 | mean context chars | latency p50 (ms) | latency p95 (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for key, r in data["pipelines"].items():
        x = r["metrics"]
        lines.append("| %s %s | %.4f | %.4f | %.1f | %.1f | %.1f |" % (
            key, r["name"], x["recall_at_5"], x["mrr_at_5"], x["mean_context_chars"],
            x["latency_ms"]["p50"], x["latency_ms"]["p95"]))
    lines += ["", "Hits whose matching unit is not on the gold doc/page (the same text appears elsewhere): " + ", ".join(
        "%s %d" % (k, r["hits_off_gold_page"]) for k, r in data["pipelines"].items()) + "."]
    lines += ["", "Multi-hop items are scored on the primary quote only in recall@5. Secondary retrieval-only metric, all refs "
              "retrieved@5 (the primary quote and every extra_refs quote each appear in some top-5 unit):", "",
              "| Pipeline | all refs retrieved@5 | n multi-hop items |", "|---|---|---|"]
    for key, r in data["pipelines"].items():
        v = r["multi_hop_all_refs_recall_at_5"]
        lines.append("| %s | %s | %d |" % (key, "n/a" if v["value"] is None else "%.4f" % v["value"], v["n"]))
    lines += ["", "Recall ceiling (the quote is inside at least one unit of the gold page; items below the ceiling have a quote "
              "that straddles a unit boundary):", "", "| Pipeline | ceiling | items not inside any unit |", "|---|---|---|"]
    for key, r in data["pipelines"].items():
        cl = r["ceiling"]
        lines.append("| %s | %.4f | %s |" % (key, cl["recall_ceiling"], ", ".join(cl["items_not_in_any_unit"]) or "none"))
    lines += ["", "recall@5 by question type:", "", "| Pipeline | " + " | ".join(
        "%s (n=%d)" % (t, next(iter(data["pipelines"].values()))["by_type"][t]["n"])
        for t in next(iter(data["pipelines"].values()))["by_type"]) + " |"]
    first = next(iter(data["pipelines"].values()))
    lines.append("|---|" + "---|" * len(first["by_type"]))
    for key, r in data["pipelines"].items():
        lines.append("| %s | " % key + " | ".join("%.4f" % r["by_type"][t]["recall_at_5"] for t in first["by_type"]) + " |")
    lines += [
        "",
        "Limits: recall@5 by quote containment misses paraphrased support and favours larger units (see the context-size "
        "column). Vector search uses HNSW settings chosen for reproducible results; eval/results/checks.md shows they "
        "returned the same top-5 children as exact search and as Chroma's defaults for P2.",
        "",
        "Latency method: %s" % m["latency_method"],
        "",
        "Environment: Python %s, %s; date %s; %s. CPU: %s. Docker CPUs: visible %s, limit %s, torch threads %s." % (
            m["environment"]["python"], ", ".join("%s %s" % kv for kv in m["environment"]["libraries"].items() if kv[1]),
            m["environment"]["date_utc"], m["environment"]["vcs"],
            m["environment"]["cpu"].get("host_cpu") or m["environment"]["cpu"].get("container_cpuinfo_model"),
            m["environment"]["docker"]["visible_cpus"], m["environment"]["docker"]["cgroup_cpu_limit"],
            m["environment"]["docker"]["torch_threads"]),
        "Models: %s." % "; ".join("%s rev %s" % (k, v["revision"]) for k, v in m["environment"]["models"].items()),
        "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=("full", "ci"), default="full")
    ap.add_argument("--pipelines", default="P1,P2,P3,P4")
    ap.add_argument("--out", default=None, help="results json path (default eval/results/retrieval_<subset>.json)")
    ap.add_argument("--note", default=None, help="free text stored in the results and the .md")
    args = ap.parse_args(argv)
    if args.note is None:
        args.note = ("Latency is provisional: it is measured on a shared machine as a by-product of this run and is "
                     "not a benchmark. Recall, MRR, context size and the other quality numbers are deterministic.")

    names = [s.strip() for s in args.pipelines.split(",") if s.strip()]
    out_json = args.out or os.path.join(RESULTS, "retrieval_%s.json" % args.subset)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    raw_dir = os.path.join(RESULTS, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    items = select_items(args.subset)
    pages = pl.load_pages(os.path.join(HERE, "corpus", "pages.jsonl"))
    embedder = pl.Embedder()
    reranker = pl.Reranker() if "P4" in names else None

    data = {"meta": {}, "n_items": len(items), "pipelines": {}}
    for name in names:
        print("running %s (%s) on %d items" % (name, pl.PIPELINE_NAMES[name], len(items)), flush=True)
        res, raw = evaluate_pipeline(name, pages, items, embedder, reranker)
        data["pipelines"][name] = res
        raw_name = "retrieval_%s%s.jsonl" % ("ci_" if args.subset == "ci" else "", name)
        with open(os.path.join(raw_dir, raw_name), "w", encoding="utf-8") as f:
            for row in raw:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        x = res["metrics"]
        print("  recall@5=%.4f mrr@5=%.4f ctx=%.0f p50=%.1fms p95=%.1fms" % (
            x["recall_at_5"], x["mrr_at_5"], x["mean_context_chars"], x["latency_ms"]["p50"], x["latency_ms"]["p95"]),
            flush=True)

    models = {
        pl.EMBED_MODEL_ID: {"revision": pl.EMBED_MODEL_REV, "role": "embeddings (all pipelines)"},
        pl.RERANK_MODEL_ID: {"revision": pl.RERANK_MODEL_REV, "role": "cross-encoder rerank (P4)"},
    }
    data["meta"] = {
        "subset": args.subset,
        "ids": [g["id"] for g in items],
        "note": args.note,
        "latency_method": (
            "time.perf_counter around pipeline.retrieve() only (query embedding, search, fusion, rerank, parent "
            "mapping); index build excluded. Warm-up: the first %d questions are run once, untimed and discarded; "
            "then each question is run once and timed. Percentiles use linear interpolation. Single process, "
            "pipelines run one after another." % pl.WARMUP_QUERIES),
        "parameters": {
            "top_k": pl.TOP_K, "candidates_per_retriever": pl.CANDIDATES, "rrf_k": pl.RRF_K,
            "p1_chunk": [pl.P1_CHUNK_SIZE, pl.P1_CHUNK_OVERLAP],
            "parent": [pl.PARENT_CHUNK_SIZE, pl.PARENT_CHUNK_OVERLAP],
            "child": [pl.CHILD_CHUNK_SIZE, pl.CHILD_CHUNK_OVERLAP],
            "bm25": "rank_bm25.BM25Okapi over children, lowercase \\w+ tokens, zero-score children excluded",
            "vector_store": "chromadb EphemeralClient, default L2 space, normalised embeddings, HNSW %s" % json.dumps(pl.HNSW_CONFIG),
        },
        "environment": envinfo.environment(models),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    write_md(os.path.splitext(out_json)[0] + ".md", data)
    print("wrote %s" % out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
