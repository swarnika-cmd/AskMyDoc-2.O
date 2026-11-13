#!/usr/bin/env python3
"""Download the evaluation corpus PDFs and verify their SHA-256 against corpus.json.

Usage: python3 -I eval/fetch_corpus.py [--dest eval/corpus/pdf]
Needs only the Python standard library; uses curl when it is installed.

Files are saved as <dest>/<id>.pdf. A file that already exists with the right
hash is kept. Any download failure or hash mismatch exits non-zero.
The PDFs are untrusted input: this script only downloads and hashes them.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fetch_once(url, dest):
    """One download attempt. Uses curl when installed (some publisher sites reject
    Python's default client), otherwise urllib."""
    part = dest + ".part"
    if shutil.which("curl"):
        subprocess.run(["curl", "-sSL", "--fail", "--max-time", "300", "-o", part, url], check=True)
    else:
        with urllib.request.urlopen(url, timeout=300) as r, open(part, "wb") as out:
            shutil.copyfileobj(r, out)
    os.replace(part, dest)


def download(url, dest, attempts=4):
    last = None
    for i in range(attempts):
        try:
            _fetch_once(url, dest)
            return
        except Exception as e:  # TLS resets and timeouts are common here; retry
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError("download failed: %s (%s)" % (url, last))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(HERE, "corpus.json"))
    ap.add_argument("--dest", default=os.path.join(HERE, "corpus", "pdf"))
    args = ap.parse_args()

    with open(args.corpus, encoding="utf-8") as f:
        docs = json.load(f)
    os.makedirs(args.dest, exist_ok=True)

    bad = 0
    for d in docs:
        path = os.path.join(args.dest, d["id"] + ".pdf")
        if not (os.path.exists(path) and sha256_file(path) == d["sha256"]):
            try:
                download(d["source_url"], path)
            except RuntimeError as e:
                print("FAIL %s: %s" % (d["id"], e))
                bad += 1
                continue
        got = sha256_file(path)
        if got != d["sha256"]:
            print("MISMATCH %s: expected %s got %s" % (d["id"], d["sha256"], got))
            bad += 1
        else:
            print("ok %s %s" % (d["id"], got))
    if bad:
        print("%d document(s) failed" % bad)
        sys.exit(1)


if __name__ == "__main__":
    main()
