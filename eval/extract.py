#!/usr/bin/env python3
"""Extract per-page text from the corpus PDFs into eval/corpus/pages.jsonl.

Text pages use pypdf. A page with fewer than 30 characters of extractable text
is rendered with pdftoppm (poppler-utils) and OCR'd with Tesseract (eng).

Requirements: pypdf, tesseract-ocr, poppler-utils (run inside Docker, see README_DATA.md).
Usage: python3 -I eval/extract.py [--pdf-dir eval/corpus/pdf] [--out eval/corpus/pages.jsonl]

Output order is deterministic: documents in corpus.json order, pages ascending.
Row: {"doc", "page" (1-based), "text", "method" ("pypdf" | "tesseract")}.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from pypdf import PdfReader

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_CHARS = 30
DPI = "250"


def ocr_page(pdf_path, page_no):
    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "p")
        subprocess.run(
            ["pdftoppm", "-r", DPI, "-gray", "-png", "-f", str(page_no), "-l", str(page_no), pdf_path, prefix],
            check=True,
        )
        imgs = sorted(f for f in os.listdir(tmp) if f.endswith(".png"))
        if len(imgs) != 1:
            raise RuntimeError("expected one image for %s p%d, got %r" % (pdf_path, page_no, imgs))
        env = dict(os.environ, OMP_THREAD_LIMIT="1")
        out = subprocess.run(
            ["tesseract", os.path.join(tmp, imgs[0]), "-", "-l", "eng", "--psm", "3"],
            check=True, capture_output=True, env=env,
        )
        return out.stdout.decode("utf-8", errors="replace").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(HERE, "corpus.json"))
    ap.add_argument("--pdf-dir", default=os.path.join(HERE, "corpus", "pdf"))
    ap.add_argument("--out", default=os.path.join(HERE, "corpus", "pages.jsonl"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    with open(args.corpus, encoding="utf-8") as f:
        docs = json.load(f)

    rows = []
    problems = 0
    for d in docs:
        path = os.path.join(args.pdf_dir, d["id"] + ".pdf")
        reader = PdfReader(path)
        n = len(reader.pages)
        if d.get("pages") and d["pages"] != n:
            print("PAGE COUNT MISMATCH %s: corpus.json=%s pdf=%d" % (d["id"], d["pages"], n))
            problems += 1
        texts = []
        for i in range(n):
            try:
                t = reader.pages[i].extract_text() or ""
            except Exception as e:
                print("pypdf failed on %s p%d: %s" % (d["id"], i + 1, e))
                t = ""
            texts.append(t)
        need_ocr = [i for i, t in enumerate(texts) if len(t.strip()) < MIN_CHARS]
        ocr = {}
        if need_ocr:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for i, txt in zip(need_ocr, ex.map(lambda i: ocr_page(path, i + 1), need_ocr)):
                    ocr[i] = txt
        for i in range(n):
            if i in ocr:
                rows.append({"doc": d["id"], "page": i + 1, "text": ocr[i], "method": "tesseract"})
            else:
                rows.append({"doc": d["id"], "page": i + 1, "text": texts[i].strip(), "method": "pypdf"})
        print("%s: %d pages, %d OCR" % (d["id"], n, len(need_ocr)), flush=True)

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote %d rows to %s" % (len(rows), args.out))
    if problems:
        sys.exit(1)


if __name__ == "__main__":
    main()
