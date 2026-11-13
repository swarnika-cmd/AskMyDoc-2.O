#!/usr/bin/env python3
"""Fail when a pipeline's recall@5 on the CI subset drops more than --max-drop points below the baseline.

  python eval/check_regression.py [--baseline eval/results/baseline_ci.json] [--current eval/results/retrieval_ci.json]

Points are percentage points of recall@5 (0.03 = 3 points). Standard library only.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def compare(baseline, current, max_drop_points=3.0):
    """Return (rows, failures). Each row: (pipeline, baseline, current, drop_points, ok)."""
    rows, fails = [], []
    if baseline.get("meta", {}).get("ids") != current.get("meta", {}).get("ids"):
        fails.append("item ids differ between baseline and current run (the subset changed): regenerate the baseline")
    for name, b in baseline["pipelines"].items():
        if name not in current["pipelines"]:
            fails.append("%s missing from the current run" % name)
            continue
        base = b["metrics"]["recall_at_5"]
        cur = current["pipelines"][name]["metrics"]["recall_at_5"]
        drop = (base - cur) * 100.0
        ok = drop <= max_drop_points + 1e-9
        rows.append((name, base, cur, drop, ok))
        if not ok:
            fails.append("%s recall@5 fell %.1f points (baseline %.4f, now %.4f; limit %.1f)" % (name, drop, base, cur, max_drop_points))
    return rows, fails


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=os.path.join(HERE, "results", "baseline_ci.json"))
    ap.add_argument("--current", default=os.path.join(HERE, "results", "retrieval_ci.json"))
    ap.add_argument("--max-drop", type=float, default=3.0, help="allowed recall@5 drop in points")
    args = ap.parse_args(argv)
    with open(args.baseline, encoding="utf-8") as f:
        baseline = json.load(f)
    with open(args.current, encoding="utf-8") as f:
        current = json.load(f)
    rows, fails = compare(baseline, current, args.max_drop)
    print("%-4s %9s %9s %12s" % ("", "baseline", "current", "drop (pts)"))
    for name, base, cur, drop, ok in rows:
        print("%-4s %9.4f %9.4f %12.1f  %s" % (name, base, cur, drop, "ok" if ok else "FAIL"))
    if fails:
        print("REGRESSION:")
        for f in fails:
            print("  - " + f)
        return 1
    print("OK: no pipeline dropped more than %.1f points" % args.max_drop)
    return 0


if __name__ == "__main__":
    sys.exit(main())
