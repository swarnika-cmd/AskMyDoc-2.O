"""Retrieval metrics. Standard library only.

A unit "hits" when the whitespace-normalised supporting quote is a substring of the whitespace-normalised unit
text. The normalisation is the one validate_golden.py uses (runs of whitespace collapse to one space, then strip).
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from validate_golden import norm  # noqa: E402

K = 5


def hit_rank(unit_texts, quote, k=K):
    """1-based rank of the first of the top-k units containing the quote, or None."""
    q = norm(quote)
    if not q:
        return None
    for rank, text in enumerate(unit_texts[:k], 1):
        if q in norm(text):
            return rank
    return None


def recall_at_k(ranks, k=K):
    """Share of items whose hit rank is within k. ranks: list of int or None."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / float(len(ranks))


def mrr_at_k(ranks, k=K):
    """Mean of 1/rank for hits within k; 0 for a miss."""
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r is not None and r <= k) / float(len(ranks))


def mean(values):
    values = list(values)
    return sum(values) / float(len(values)) if values else 0.0


def percentile(values, p):
    """Linear-interpolation percentile (the numpy default): p in [0, 100]."""
    xs = sorted(values)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * p / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarise(ranks, context_chars, latencies_ms):
    return {
        "n": len(ranks),
        "recall_at_5": recall_at_k(ranks),
        "mrr_at_5": mrr_at_k(ranks),
        "mean_context_chars": mean(context_chars),
        "latency_ms": {
            "p50": percentile(latencies_ms, 50),
            "p95": percentile(latencies_ms, 95),
            "mean": mean(latencies_ms),
            "n": len(latencies_ms),
        },
    }
