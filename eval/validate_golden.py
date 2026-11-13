#!/usr/bin/env python3
"""Validate eval/golden.jsonl against eval/corpus/pages.jsonl.

Usage: python3 eval/validate_golden.py [--golden PATH] [--pages PATH] [--ci PATH]

Checks:
  - ids are unique; required fields are present; type is one of the allowed types
  - exactly 150 items and exactly 20 unanswerable items
  - every supporting_quote (and every extra_refs quote) appears verbatim in the text
    of that doc at that page after whitespace normalisation
    (runs of whitespace collapse to one space, then strip)
  - quote length is 30 to 300 characters (after normalisation)
  - unanswerable items have doc = null, page = null and an empty quote
  - no document supplies more than 25% of the items
  - every id in ci_subset.txt exists (and the subset has 30 ids, 4 of them unanswerable)

Prints a per-type summary and every failure. Exits 1 on any failure.
Standard library only; runs on Python 3.9+.
"""
import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

TYPES = ("factual", "multi-hop", "table", "unanswerable")
REQUIRED = ("id", "question", "answer", "doc", "page", "supporting_quote", "type")
QUOTE_MIN, QUOTE_MAX = 30, 300
EXPECT_TOTAL = 150
EXPECT_UNANSWERABLE = 20
MAX_DOC_SHARE = 0.25
CI_SIZE = 30
CI_UNANSWERABLE = 4


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def load_pages(path):
    pages = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            pages[(r["doc"], r["page"])] = norm(r["text"])
    return pages


def load_golden(path):
    items = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except ValueError as e:
                items.append({"_parse_error": "line %d: %s" % (n, e)})
    return items


def load_ci(path):
    ids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                ids.append(s)
    return ids


def check_quote(label, doc, page, quote, pages, fails):
    if not isinstance(quote, str):
        fails.append("%s: supporting_quote is not a string" % label)
        return
    q = norm(quote)
    if not (QUOTE_MIN <= len(q) <= QUOTE_MAX):
        fails.append("%s: quote length %d outside %d-%d" % (label, len(q), QUOTE_MIN, QUOTE_MAX))
    if not isinstance(page, int) or isinstance(page, bool):
        fails.append("%s: page %r is not an integer" % (label, page))
        return
    text = pages.get((doc, page))
    if text is None:
        fails.append("%s: no text for doc=%r page=%r" % (label, doc, page))
    elif q not in text:
        fails.append("%s: quote not found verbatim in %s p%s: %r" % (label, doc, page, q[:80]))


def validate(items, pages, ci_ids=None, expect_total=EXPECT_TOTAL, expect_unanswerable=EXPECT_UNANSWERABLE,
             max_doc_share=MAX_DOC_SHARE, ci_size=CI_SIZE, ci_unanswerable=CI_UNANSWERABLE):
    """Return (failures, summary). summary has per-type and per-doc counts."""
    fails = []
    pages = {k: norm(v) for k, v in pages.items()}
    ids = collections.Counter()
    by_type = collections.Counter()
    by_doc = collections.Counter()
    by_id = {}

    for idx, it in enumerate(items):
        if "_parse_error" in it:
            fails.append("item #%d: unparseable JSON (%s)" % (idx + 1, it["_parse_error"]))
            continue
        label = "item %s" % it.get("id", "#%d" % (idx + 1))
        missing = [k for k in REQUIRED if k not in it]
        if missing:
            fails.append("%s: missing fields %s" % (label, ", ".join(missing)))
            continue
        ids[it["id"]] += 1
        by_id[it["id"]] = it
        t = it["type"]
        if t not in TYPES:
            fails.append("%s: type %r not in %s" % (label, t, TYPES))
            continue
        by_type[t] += 1
        for k in ("question", "answer"):
            if not isinstance(it[k], str) or not it[k].strip():
                fails.append("%s: %s is empty" % (label, k))

        if t == "unanswerable":
            if it["doc"] is not None or it["page"] is not None:
                fails.append("%s: unanswerable item must have doc=null and page=null" % label)
            if it["supporting_quote"] != "":
                fails.append("%s: unanswerable item must have an empty supporting_quote" % label)
            if it.get("extra_refs"):
                fails.append("%s: unanswerable item must not have extra_refs" % label)
            continue

        if not it["doc"]:
            fails.append("%s: answerable item needs a doc" % label)
            continue
        by_doc[it["doc"]] += 1
        check_quote(label, it["doc"], it["page"], it["supporting_quote"], pages, fails)
        for j, ref in enumerate(it.get("extra_refs") or []):
            rl = "%s extra_refs[%d]" % (label, j)
            if not isinstance(ref, dict) or not all(k in ref for k in ("doc", "page", "supporting_quote")):
                fails.append("%s: needs doc, page, supporting_quote" % rl)
                continue
            check_quote(rl, ref["doc"], ref["page"], ref["supporting_quote"], pages, fails)

    for i, c in ids.items():
        if c > 1:
            fails.append("duplicate id %s (%d times)" % (i, c))
    if len(items) != expect_total:
        fails.append("expected %d items, found %d" % (expect_total, len(items)))
    if by_type["unanswerable"] != expect_unanswerable:
        fails.append("expected %d unanswerable items, found %d" % (expect_unanswerable, by_type["unanswerable"]))
    if items:
        for d, c in by_doc.items():
            if c / float(len(items)) > max_doc_share:
                fails.append("document %s has %d of %d items (over %d%%)" % (d, c, len(items), int(max_doc_share * 100)))

    if ci_ids is not None:
        seen = set()
        for i in ci_ids:
            if i in seen:
                fails.append("ci_subset: duplicate id %s" % i)
            seen.add(i)
            if i not in by_id:
                fails.append("ci_subset: id %s not in golden set" % i)
        if len(ci_ids) != ci_size:
            fails.append("ci_subset: expected %d ids, found %d" % (ci_size, len(ci_ids)))
        n_un = sum(1 for i in ci_ids if i in by_id and by_id[i].get("type") == "unanswerable")
        if n_un != ci_unanswerable:
            fails.append("ci_subset: expected %d unanswerable ids, found %d" % (ci_unanswerable, n_un))

    return fails, {"by_type": dict(by_type), "by_doc": dict(by_doc), "total": len(items)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default=os.path.join(HERE, "golden.jsonl"))
    ap.add_argument("--pages", default=os.path.join(HERE, "corpus", "pages.jsonl"))
    ap.add_argument("--ci", default=os.path.join(HERE, "ci_subset.txt"))
    args = ap.parse_args(argv)

    pages = load_pages(args.pages)
    items = load_golden(args.golden)
    ci_ids = load_ci(args.ci)
    fails, summary = validate(items, pages, ci_ids)

    print("items: %d" % summary["total"])
    print("per type:")
    for t in TYPES:
        print("  %-13s %d" % (t, summary["by_type"].get(t, 0)))
    print("per document (answerable items, by primary doc):")
    for d in sorted(summary["by_doc"]):
        print("  %-30s %d" % (d, summary["by_doc"][d]))
    print("ci_subset ids: %d" % len(ci_ids))
    if fails:
        print("FAILURES (%d):" % len(fails))
        for f in fails:
            print("  - " + f)
        return 1
    print("OK: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
