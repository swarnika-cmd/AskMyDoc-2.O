# Evaluation data: corpus and golden set

This folder holds the data that every AskMyDoc retrieval pipeline is evaluated on.

| File | What it is |
|---|---|
| `corpus.json` | One entry per source PDF: id, title, source URL, publisher, licence or public status, SHA-256, page count, kind, extraction method, retrieval date |
| `fetch_corpus.py` | Downloads the PDFs into `corpus/pdf/` (gitignored) and checks each SHA-256. Exits non-zero on a download failure or a hash mismatch |
| `extract.py` | Turns the PDFs into `corpus/pages.jsonl` |
| `corpus/pages.jsonl` | Extracted text, one row per document page: `doc`, `page` (1-based), `text`, `method`. This is the canonical text for everything downstream |
| `golden.jsonl` | 150 questions with reference answers and verbatim supporting quotes |
| `validate_golden.py` | Checks the golden set against `pages.jsonl`. CI runs it |
| `test_validate_golden.py` | Unit tests for the validator, including a deliberately broken item |
| `ci_subset.txt` | 30 golden ids used by the CI retrieval run |

## Corpus

Eight public PDFs, 413 pages in total. All were retrieved on 2026-11-13 from the publishers' own sites. The PDFs are not committed.

| id | Pages | Kind | Extraction | Source |
|---|---|---|---|---|
| `es1980-81` | 80 | scanned | Tesseract (all 80 pages) | indiabudget.gov.in, Economic Survey 1980-81 chapter 9 and statistical appendix |
| `pay-allowances-2023-24` | 47 | table-heavy | pypdf (17 pages: 1-18 except 4), Tesseract (30 pages: 4 and 19-47) | doe.gov.in, Annual Report on Pay and Allowances 2023-24 |
| `cusb-ug-ordinance` | 61 | text | pypdf | cusb.ac.in, Ordinance Governing Five-Year Integrated UG-PG Programme (Science, Arts and Commerce), Central University of South Bihar |
| `nitrkl-btech-regs` | 31 | text | pypdf | nitrkl.ac.in, B.Tech. regulations (summarised version) |
| `dea-frbm-h2-2023-24` | 13 | table-heavy | pypdf (11 pages), Tesseract (2 blank cover pages) | dea.gov.in, FRBM half-yearly review, FY 2023-24 |
| `sebi-cra-circular-2016` | 14 | text | pypdf | sebi.gov.in, circular SEBI/HO/MIRSD/MIRSD4/CIR/P/2016/119 on enhanced standards for credit rating agencies |
| `sebi-cra-consultation-2017` | 15 | text | pypdf | sebi.gov.in, consultation paper on the CRA regulatory framework |
| `sebi-mf-master-2011` | 152 | text | pypdf | sebi.gov.in, Master Circular for Mutual Funds (January 2011) |

Full URLs, publishers and hashes are in `corpus.json`.

Licence and redistribution: no explicit licence was found for any of these documents. They are published by government bodies and universities for public access. This repository stores only the extracted text (`corpus/pages.jsonl`) for evaluation; the PDFs are not redistributed. Source URLs and SHA-256 hashes are recorded in `corpus.json`. The cover of the pay and allowances report carries a "For official use only" marking, although the report is published on doe.gov.in.

Extraction rules: pypdf reads every page. A page with fewer than 30 characters of extractable text is rendered at 250 dpi with `pdftoppm` and read with Tesseract (`eng`, page segmentation mode 3). Documents appear in `corpus.json` order and pages ascend, so the output order is fixed.

The text is committed because OCR output varies between Tesseract versions, and downstream jobs must not need OCR. `pages.jsonl` was produced with Python 3.11, pypdf 7.0.1, Tesseract 5.5.0 and poppler 25.03.0.

Known text-quality limits, kept as they are because the evaluation is meant to run on messy input:

- OCR text contains recognition errors, hyphenated line breaks and broken table alignment (for example `escala- tion`, `cdible`).
- Nine landscape pages of `pay-allowances-2023-24` (pages 31 to 34, 38, 43, 45, 46, 47) were scanned rotated, and Tesseract returns unreadable text for them. No golden item is anchored on those pages.
- The page count of `es1980-81` covers only chapter 9 and the statistical appendix, not the full Survey.
- Some documents contradict themselves. In `pay-allowances-2023-24` the 2023-24 total is 295912.54 crore on pages 8 to 10 and 288150.92 crore on page 18. Pay-report items use only the figures on these pages: q052, q057, q071 (page 5); q002, q051, q055 (page 6); q029 and q134 (ten-year expenditure table, page 9); q098 and q131 (five-year components table, page 10); q136 (page 20); q007 (page 24); q005 (page 28); q013 (page 35); q006 (page 37). q134 and q098 name the table they rely on, and q029 and q131 use years (2019-20 to 2022-23, and 2021-22) whose totals agree across pages 9, 10 and 18. No item uses the page 18 totals. Other documents also contain inconsistent figures (for example CUSB supplementary-exam timing and fees); no item relies on them.

## Golden set

`golden.jsonl` has 150 items, one JSON object per line.

| Field | Meaning |
|---|---|
| `id` | `q001` to `q150` |
| `question` | The question as a user would ask it |
| `answer` | Short reference answer |
| `doc`, `page` | Document id and 1-based page that holds the supporting text. `null` for unanswerable items |
| `supporting_quote` | 30 to 300 characters copied from that page's text in `pages.jsonl`. Empty for unanswerable items |
| `type` | `factual`, `table`, `multi-hop` or `unanswerable` |
| `extra_refs` | Multi-hop items only: further `{doc, page, supporting_quote}` entries for the second fact |
| `absence_check` | Unanswerable items only: the terms searched for in `pages.jsonl` to confirm the answer is not in the corpus |

Mix: 70 factual, 30 table, 30 multi-hop, 20 unanswerable. No document supplies more than 25% of the items (the largest supplies 23 of 150). Per primary document: `sebi-mf-master-2011` 23, `cusb-ug-ordinance` 18, `es1980-81` 17, `nitrkl-btech-regs` 17, `pay-allowances-2023-24` 15, `sebi-cra-circular-2016` 14, `sebi-cra-consultation-2017` 14, `dea-frbm-h2-2023-24` 12, and 20 unanswerable items with no document. Four multi-hop items combine two documents.

How it was built: questions were written from the source documents, and every answer is backed by a verbatim quote that `validate_golden.py` checks. Table items quote the extracted row text as it appears in `pages.jsonl`, including OCR artefacts. Unanswerable items are plausible questions on the same topics whose answers are not in the corpus; each lists the terms that were searched for in `pages.jsonl`.

Limits: 150 items over 8 documents, so differences between pipelines of a few points are inside the noise. Answers were checked against the quotes, not against any outside source.

## CI subset

`ci_subset.txt` lists 30 ids: 14 factual, 6 table, 6 multi-hop and 4 unanswerable. Within each type the items were grouped by document, shuffled with a seeded generator and picked round-robin across documents. The seed is in a comment line in the file.

## Rebuild

PDFs are untrusted input. Download them into an empty directory, and read them only inside a container.

```sh
# from the repository root
python3 -I eval/fetch_corpus.py        # downloads with curl, checks SHA-256, exits 1 on any mismatch

docker run --rm -v "$PWD":/work -w /work python:3.11-slim sh -c '
  apt-get update && apt-get install -y --no-install-recommends tesseract-ocr poppler-utils &&
  pip install pypdf==7.0.1 &&
  python -I eval/extract.py'

python3 -I eval/validate_golden.py        # checks every quote and the ci subset
python3 -I eval/test_validate_golden.py   # validator self-test
```

`fetch_corpus.py` needs only the Python standard library; it uses curl when installed because some publisher sites reject Python's default client. If a source site moves a file, the hash check fails and the script exits non-zero. A re-extraction with other Tesseract or pypdf versions can change `pages.jsonl`, and then some quotes may no longer validate. Treat that as a signal to review the affected items.
