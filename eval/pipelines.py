"""Four retrieval pipelines over eval/corpus/pages.jsonl.

P1  fixed 800/100 chunks + dense
P2  parent-child + dense (the app's design): top-5 children -> deduplicated parents
P3  parent-child + BM25 + dense fused with RRF (k=60): top-5 fused children -> parents
P4  P3, then a cross-encoder reranks the top-20 fused children: top-5 reranked children -> parents

All pipelines use the app's embedding model and an in-memory ChromaDB collection. Parent/child sizes are
imported from app.ingest, so the evaluation measures the sizes that ship. Unit ids are deterministic
(<doc>:p<page>:P<k> for parents, :c<j> for children), never random or time based.

Every pipeline exposes retrieve(query) -> list of units (dicts: id, doc, page, text), at most TOP_K, best first.
"""
import itertools
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

from app.ingest import (  # noqa: E402
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
)

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL_REV = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
RERANK_MODEL_ID = "BAAI/bge-reranker-base"
RERANK_MODEL_REV = "2cfc18c9415c912f9d8155881c133215df768a70"

P1_CHUNK_SIZE = 800
P1_CHUNK_OVERLAP = 100
TOP_K = 5          # units returned to the generator (the app searches 5 children)
CANDIDATES = 20    # depth taken from each of BM25 and dense before fusion (P3, P4); also the rerank pool for P4
RRF_K = 60
WARMUP_QUERIES = 3

PIPELINE_NAMES = {
    "P1": "fixed 800/100 chunks + dense",
    "P2": "parent-child + dense (current app design)",
    "P3": "parent-child + BM25 + dense, RRF k=60",
    "P4": "P3 + cross-encoder rerank (bge-reranker-base, top-20)",
}


def model_source(model_id, revision):
    """Local snapshot under $EVAL_MODEL_ROOT/<org>/<repo> if present, else the Hub id pinned to a revision.

    Returns (path_or_id, kwargs) for the loader; kwargs carries revision only for Hub ids.
    """
    root = os.environ.get("EVAL_MODEL_ROOT")
    if root:
        local = os.path.join(root, *model_id.split("/"))
        if os.path.isdir(local):
            return local, {}
    return model_id, {"revision": revision}


def load_pages(path):
    """Rows of pages.jsonl with non-empty text, in file order. Text is kept raw (the app chunks raw page text)."""
    import json

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["text"].strip():
                    rows.append({"doc": r["doc"], "page": r["page"], "text": r["text"]})
    return rows


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


class Embedder:
    """all-MiniLM-L6-v2 with normalised vectors, as the app configures it. Document vectors are memoised by text."""

    def __init__(self):
        from sentence_transformers import SentenceTransformer

        src, kw = model_source(EMBED_MODEL_ID, EMBED_MODEL_REV)
        self.model = SentenceTransformer(src, device="cpu", **kw)
        self._memo = {}

    def embed_documents(self, texts):
        missing = [t for t in dict.fromkeys(texts) if t not in self._memo]
        if missing:
            vecs = self.model.encode(missing, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
            for t, v in zip(missing, vecs):
                self._memo[t] = v.tolist()
        return [self._memo[t] for t in texts]

    def embed_query(self, text):
        return self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0].tolist()


class Reranker:
    def __init__(self):
        from sentence_transformers import CrossEncoder

        src, kw = model_source(RERANK_MODEL_ID, RERANK_MODEL_REV)
        self.model = CrossEncoder(src, max_length=512, device="cpu", **kw)

    def score(self, query, texts):
        return [float(s) for s in self.model.predict([(query, t) for t in texts], batch_size=16, show_progress_bar=False)]


_collection_counter = itertools.count()


# Chroma's default HNSW settings gave slightly different neighbour lists from one index build to the next (a few
# queries out of 150 differed in the top-20 in a repeat test), which would make the CI baseline flaky. A single build
# thread plus a wide graph (M=32, ef 400) made repeated builds return identical results (0 of 150 differed).
# For P2 on the 130 answerable questions, exact search (numpy), this configuration and Chroma's default settings
# returned identical top-5 child sets; the measurement is in eval/results/checks.md (run_checks.py).
HNSW_CONFIG = {"num_threads": 1, "max_neighbors": 32, "ef_construction": 400, "ef_search": 400}


def _new_collection(prefix, hnsw=HNSW_CONFIG):
    import chromadb

    client = chromadb.EphemeralClient()
    # unique name: in-process ephemeral clients share state, so never reuse a collection name
    name = "%s_%d_%d" % (prefix, os.getpid(), next(_collection_counter))
    if hnsw is None:
        return client, client.create_collection(name)  # Chroma defaults
    return client, client.create_collection(name, configuration={"hnsw": dict(hnsw)})


def _chunks(pages, size, overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    out = []
    for p in pages:
        doc = Document(page_content=p["text"], metadata={"doc": p["doc"], "page": p["page"]})
        for k, d in enumerate(splitter.split_documents([doc])):
            out.append((k, p, d.page_content))
    return out


class _DenseIndex:
    """Chroma collection of (id, text) with metadata doc/page; query returns ids best first."""

    def __init__(self, embedder, prefix, rows, hnsw=HNSW_CONFIG):
        self.embedder = embedder
        self.client, self.col = _new_collection(prefix, hnsw)
        ids = [r["id"] for r in rows]
        texts = [r["text"] for r in rows]
        vecs = embedder.embed_documents(texts)
        step = 2000
        for i in range(0, len(ids), step):
            self.col.add(
                ids=ids[i:i + step],
                embeddings=vecs[i:i + step],
                documents=texts[i:i + step],
                metadatas=[{"doc": r["doc"], "page": r["page"]} for r in rows[i:i + step]],
            )
        self.size = len(ids)

    def query(self, query, n):
        res = self.col.query(query_embeddings=[self.embedder.embed_query(query)], n_results=min(n, self.size))
        return res["ids"][0]


class FixedChunkDense:
    """P1."""

    def __init__(self, pages, embedder, **_):
        rows = []
        for k, p, text in _chunks(pages, P1_CHUNK_SIZE, P1_CHUNK_OVERLAP):
            rows.append({"id": "%s:p%d:c%d" % (p["doc"], p["page"], k), "doc": p["doc"], "page": p["page"], "text": text})
        self.units = {r["id"]: r for r in rows}
        self.index = _DenseIndex(embedder, "p1", rows)
        self.stats = {"units": len(rows)}

    def retrieve(self, query):
        return [self.units[i] for i in self.index.query(query, TOP_K)]


class _ParentChild:
    """Shared build for P2 to P4: parents 3000/300 per page, children 400/100 per parent (sizes from app.ingest)."""

    def __init__(self, pages, embedder):
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP)
        self.parents = {}
        self.children = []  # list of dicts in build order; position = child ordinal
        for p in pages:
            doc = Document(page_content=p["text"], metadata={"doc": p["doc"], "page": p["page"]})
            for k, parent in enumerate(parent_splitter.split_documents([doc])):
                pid = "%s:p%d:P%d" % (p["doc"], p["page"], k)
                self.parents[pid] = {"id": pid, "doc": p["doc"], "page": p["page"], "text": parent.page_content}
                for j, child in enumerate(child_splitter.split_documents([parent])):
                    self.children.append({
                        "id": "%s:c%d" % (pid, j), "parent": pid, "doc": p["doc"], "page": p["page"],
                        "text": child.page_content,
                    })
        self.child_by_id = {c["id"]: c for c in self.children}
        self.ordinal = {c["id"]: i for i, c in enumerate(self.children)}
        self.dense = _DenseIndex(embedder, "pc", self.children)
        self.stats = {"parents": len(self.parents), "children": len(self.children)}

    def to_parents(self, child_ids):
        """Deduplicate in rank order, keep the first TOP_K distinct parents."""
        out, seen = [], set()
        for cid in child_ids:
            pid = self.child_by_id[cid]["parent"]
            if pid not in seen:
                seen.add(pid)
                out.append(self.parents[pid])
        return out


class ParentChildDense(_ParentChild):
    """P2: top-5 children by dense similarity, mapped to deduplicated parents (what app/retrieve.py does)."""

    def __init__(self, pages, embedder, **_):
        super().__init__(pages, embedder)

    def retrieve(self, query):
        return self.to_parents(self.dense.query(query, TOP_K))


def rrf_fuse(rankings, k=RRF_K):
    """Reciprocal rank fusion. rankings: lists of ids, best first. Returns ids by fused score desc.

    Ties break on the id's first appearance across the input lists (deterministic).
    """
    scores, first_seen = {}, {}
    for ranking in rankings:
        for rank, i in enumerate(ranking, 1):
            scores[i] = scores.get(i, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(i, len(first_seen))
    return sorted(scores, key=lambda i: (-scores[i], first_seen[i]))


class ParentChildHybrid(_ParentChild):
    """P3 (reranker=None) and P4 (reranker given)."""

    def __init__(self, pages, embedder, reranker=None, **_):
        super().__init__(pages, embedder)
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi([_tokenize(c["text"]) for c in self.children])
        self.reranker = reranker

    def _bm25_rank(self, query, n):
        scores = self.bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [self.children[i]["id"] for i in order[:n] if scores[i] > 0]

    def fused(self, query):
        return rrf_fuse([self.dense.query(query, CANDIDATES), self._bm25_rank(query, CANDIDATES)])

    def retrieve(self, query):
        fused = self.fused(query)
        if self.reranker is None:
            return self.to_parents(fused[:TOP_K])
        pool = fused[:CANDIDATES]
        scores = self.reranker.score(query, [self.child_by_id[i]["text"] for i in pool])
        order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
        return self.to_parents([pool[i] for i in order[:TOP_K]])


def build_pipeline(name, pages, embedder, reranker=None):
    if name == "P1":
        return FixedChunkDense(pages, embedder)
    if name == "P2":
        return ParentChildDense(pages, embedder)
    if name == "P3":
        return ParentChildHybrid(pages, embedder)
    if name == "P4":
        if reranker is None:
            raise ValueError("P4 needs a reranker")
        return ParentChildHybrid(pages, embedder, reranker=reranker)
    raise ValueError("unknown pipeline %r" % name)


def run_queries(pipeline, questions, warmup=WARMUP_QUERIES):
    """Retrieve for every question. Returns [(units, elapsed_ms)].

    Warm-up policy: the first `warmup` questions are run once, untimed and discarded, so one-off costs (lazy
    allocations, first-call kernels) stay out of the numbers. Every question is then run once and timed with
    time.perf_counter around pipeline.retrieve() only (query embedding, search, fusion, rerank, parent mapping);
    index building is excluded.
    """
    for q in questions[:warmup]:
        pipeline.retrieve(q)
    out = []
    for q in questions:
        t0 = time.perf_counter()
        units = pipeline.retrieve(q)
        out.append((units, (time.perf_counter() - t0) * 1000.0))
    return out
