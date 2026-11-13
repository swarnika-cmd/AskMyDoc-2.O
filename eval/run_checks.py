#!/usr/bin/env python3
"""Two measurements that back statements in the README. Writes eval/results/checks.json and checks.md.

1. HNSW: for P2 on the answerable questions, compare the top-5 child chunks from exact search (numpy cosine over the
   same embeddings), from the evaluation's Chroma configuration, and from Chroma's default configuration.
2. Refusal gate: the app marks an answer NOT_FOUND on its own when the best child similarity is below 0.2 (and below
   0.4 for summary-style questions it still answers). The app's score is 1 - distance/2 with Chroma's L2 distance on
   unit vectors, which is the cosine similarity. This counts how many of the 150 questions are below those thresholds.
   The gate is not modelled in the LLM run.

  python eval/run_checks.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import envinfo  # noqa: E402
import pipelines as pl  # noqa: E402
from validate_golden import load_golden  # noqa: E402

RESULTS = os.path.join(HERE, "results")


def main():
    golden = load_golden(os.path.join(HERE, "golden.jsonl"))
    pages = pl.load_pages(os.path.join(HERE, "corpus", "pages.jsonl"))
    emb = pl.Embedder()
    pc = pl._ParentChild(pages, emb)               # evaluation configuration (HNSW_CONFIG)
    default = pl._DenseIndex(emb, "pcdef", pc.children, hnsw=None)  # Chroma default settings

    ids = [c["id"] for c in pc.children]
    mat = np.array(emb.embed_documents([c["text"] for c in pc.children]), dtype=np.float64)

    def exact_top(qv, k=5):
        sims = mat @ qv
        order = sorted(range(len(sims)), key=lambda i: (-sims[i], i))
        return [ids[i] for i in order[:k]], float(sims[order[0]])

    answerable = [g for g in golden if g["type"] != "unanswerable"]
    same = {"eval_vs_exact": [0, 0], "default_vs_exact": [0, 0], "default_vs_eval": [0, 0]}  # [identical sets, identical ordered lists]
    for g in answerable:
        ex, _ = exact_top(np.array(emb.embed_query(g["question"]), dtype=np.float64))
        ev = pc.dense.query(g["question"], 5)
        df = default.query(g["question"], 5)
        for name, (a, b) in {"eval_vs_exact": (ev, ex), "default_vs_exact": (df, ex), "default_vs_eval": (df, ev)}.items():
            same[name][0] += set(a) == set(b)
            same[name][1] += a == b

    best = []
    for g in golden:
        _, top = exact_top(np.array(emb.embed_query(g["question"]), dtype=np.float64))
        best.append((g["id"], top, g["type"]))
    sims = [b[1] for b in best]
    out = {
        "environment": envinfo.environment({pl.EMBED_MODEL_ID: {"revision": pl.EMBED_MODEL_REV, "role": "embeddings"}}),
        "hnsw": {
            "pipeline": "P2 child search, top-5", "n_questions": len(answerable), "eval_config": pl.HNSW_CONFIG,
            "identical_top5_sets": {k: v[0] for k, v in same.items()},
            "identical_top5_ordered_lists": {k: v[1] for k, v in same.items()},
        },
        "refusal_gate": {
            "n_questions": len(golden), "min_best_cosine": min(sims), "mean_best_cosine": sum(sims) / len(sims),
            "below_0.2": sum(1 for s in sims if s < 0.2), "below_0.4": sum(1 for s in sims if s < 0.4),
            "lowest_5": [{"id": i, "type": t, "best_cosine": round(s, 4)} for i, s, t in sorted(best, key=lambda b: b[1])[:5]],
        },
    }
    with open(os.path.join(RESULTS, "checks.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    h, r = out["hnsw"], out["refusal_gate"]
    lines = [
        "# Checks behind README statements", "",
        "## HNSW settings vs exact search", "",
        "P2 child search, top-5 children, %d answerable questions. Evaluation configuration: %s." % (h["n_questions"], json.dumps(h["eval_config"])), "",
        "| Comparison | identical top-5 sets | identical ordered lists |", "|---|---|---|",
    ]
    for k in h["identical_top5_sets"]:
        lines.append("| %s | %d of %d | %d of %d |" % (k.replace("_", " "), h["identical_top5_sets"][k], h["n_questions"],
                                                      h["identical_top5_ordered_lists"][k], h["n_questions"]))
    lines += [
        "", "## App refusal gate (best child similarity)", "",
        "Across all %d questions (including the unanswerable ones), the best child cosine similarity has minimum %.4f and mean %.4f. "
        "Questions below 0.2 (the app's NOT_FOUND threshold): %d. Below 0.4: %d. Lowest five: %s." % (
            r["n_questions"], r["min_best_cosine"], r["mean_best_cosine"], r["below_0.2"], r["below_0.4"],
            ", ".join("%s %.4f (%s)" % (x["id"], x["best_cosine"], x["type"]) for x in r["lowest_5"])),
        "", "Environment: %s." % out["environment"]["vcs"], ""]
    with open(os.path.join(RESULTS, "checks.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
