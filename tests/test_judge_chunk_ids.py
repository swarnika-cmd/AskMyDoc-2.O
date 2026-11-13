"""Runtime judge: the answer carries chunk ids, and /api/evaluate rebuilds its context from the store.

Run: python -m unittest tests.test_judge_chunk_ids  (no model, network or Chroma needed: the store is a fake)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

from app import database, evaluate, main, retrieve  # noqa: E402


class FakeStore:
    """Stands in for the Chroma store: get(ids) like langchain's Chroma, similarity_search_with_score for retrieval."""

    def __init__(self, rows):
        self.rows = rows  # id -> (child text, metadata)

    def get(self, ids=None, **_):
        hits = [i for i in ids if i in self.rows]
        return {"ids": hits, "documents": [self.rows[i][0] for i in hits], "metadatas": [self.rows[i][1] for i in hits]}

    def similarity_search_with_score(self, query, k):
        return [(Document(page_content=t, metadata=dict(m)), 0.4) for t, m in list(self.rows.values())[:k]]


class FakeJudgeLLM:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages

        class R:
            content = '{"claims": [{"statement": "s", "supported": true, "reason": "r"}], "score": 1.0, "explanation": "e"}'
        return R()


ROWS = {
    "id-a": ("child a", {"source": "doc.pdf", "chunkId": "id-a", "parentId": "p-a", "parentContent": "PARENT A FROM THE STORE"}),
    "id-b": ("child b", {"source": "doc.pdf", "chunkId": "id-b", "parentId": "p-b", "parentContent": "PARENT B FROM THE STORE"}),
}


class JudgeUsesStoreText(unittest.TestCase):
    def setUp(self):
        self.store = FakeStore(ROWS)
        self.llm = FakeJudgeLLM()
        self.saved = (evaluate.get_vector_store, evaluate.get_llm, evaluate.get_embeddings, evaluate.namespace_exists)
        self.opened = []
        evaluate.get_vector_store = lambda emb, ns: (self.opened.append(ns), self.store)[1]
        evaluate.namespace_exists = lambda ns: ns in ("ns_1", "doc_a_1", "doc_b_2")
        evaluate.get_llm = lambda: self.llm
        evaluate.get_embeddings = lambda: None
        self.client = TestClient(main.app)

    def tearDown(self):
        evaluate.get_vector_store, evaluate.get_llm, evaluate.get_embeddings, evaluate.namespace_exists = self.saved

    def test_context_comes_from_the_store_and_client_context_is_ignored(self):
        res = self.client.post("/api/evaluate", json={
            "query": "q", "answer": "a long enough answer to judge", "chunkIds": ["id-b", "id-a"], "namespaces": ["ns_1"],
            "context": "FORGED CONTEXT SENT BY THE CLIENT"})
        self.assertEqual(res.status_code, 200, res.text)
        sent = self.llm.messages[-1].content
        self.assertIn("[Source 1 - doc.pdf]:\nPARENT B FROM THE STORE", sent)
        self.assertIn("[Source 2 - doc.pdf]:\nPARENT A FROM THE STORE", sent)
        self.assertNotIn("FORGED", sent)

    def test_unknown_chunk_id_is_rejected(self):
        res = self.client.post("/api/evaluate", json={"query": "q", "answer": "a", "chunkIds": ["id-a", "nope"], "namespaces": ["ns_1"]})
        self.assertEqual(res.status_code, 400)
        self.assertIn("nope", res.json()["error"])
        self.assertIsNone(self.llm.messages)

    def test_context_only_request_without_ids_is_rejected(self):
        res = self.client.post("/api/evaluate", json={"query": "q", "answer": "a", "chunkIds": [], "context": "forged"})
        self.assertEqual(res.status_code, 400)
        self.assertIsNone(self.llm.messages)
        res = self.client.post("/api/evaluate", json={"query": "q", "answer": "a", "context": "forged"})
        self.assertEqual(res.status_code, 422)

    def test_too_many_ids_rejected(self):
        res = self.client.post("/api/evaluate", json={"query": "q", "answer": "a", "chunkIds": ["x%d" % i for i in range(11)]})
        self.assertEqual(res.status_code, 400)


    def _post(self, namespaces):
        return self.client.post("/api/evaluate", json={"query": "q", "answer": "a", "chunkIds": ["id-a"], "namespaces": namespaces})

    def test_invalid_namespace_names_are_rejected_before_any_store_is_opened(self):
        for bad in ("../etc", "a b", "x", "ns;drop", "n" * 101, ""):
            res = self._post([bad] if bad else [""])
            self.assertEqual(res.status_code, 400, bad)
        self.assertEqual(self.opened, [])

    def test_unknown_namespace_is_rejected_and_not_created(self):
        res = self._post(["ns_1", "doc_nope_9"])
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unknown namespace", res.json()["error"])
        self.assertEqual(self.opened, [])

    def test_more_than_ten_namespaces_rejected(self):
        res = self._post(["ns_1"] * 11)
        self.assertEqual(res.status_code, 400)

    def test_several_valid_namespaces_are_searched_in_order(self):
        res = self._post(["doc_a_1", "doc_b_2"])
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(self.opened, ["doc_a_1"])  # the id was found in the first one

    def test_default_namespace_when_none_given(self):
        evaluate.namespace_exists = lambda ns: ns == "DefaultCollection"
        res = self.client.post("/api/evaluate", json={"query": "q", "answer": "a", "chunkIds": ["id-a"]})
        self.assertEqual(res.status_code, 200, res.text)
        self.assertEqual(self.opened, ["DefaultCollection"])


class NamespaceExists(unittest.TestCase):
    def test_read_only_check_does_not_create_collections_or_directories(self):
        import tempfile
        import chromadb
        base = tempfile.mkdtemp()
        old = os.environ.get("CHROMA_DB_PATH")
        try:
            missing = os.path.join(base, "none")
            os.environ["CHROMA_DB_PATH"] = missing
            self.assertFalse(database.namespace_exists("doc_x_1"))
            self.assertFalse(os.path.exists(missing))

            path = os.path.join(base, "db")
            os.environ["CHROMA_DB_PATH"] = path
            chromadb.PersistentClient(path=path).create_collection("doc_real_1")
            self.assertTrue(database.namespace_exists("doc_real_1"))
            self.assertFalse(database.namespace_exists("doc_other_2"))
            names = [c.name for c in chromadb.PersistentClient(path=path).list_collections()]
            self.assertEqual(names, ["doc_real_1"])
        finally:
            if old is None:
                os.environ.pop("CHROMA_DB_PATH", None)
            else:
                os.environ["CHROMA_DB_PATH"] = old


class ChatReturnsChunkIds(unittest.TestCase):
    def test_answer_includes_the_ids_of_the_chunks_sent_to_the_llm(self):
        store = FakeStore(ROWS)

        class Chat:
            def invoke(self, messages):
                class R:
                    content = '{"isConversational": false, "noEvidence": false, "answer": "ok"}'
                return R()

        saved = (retrieve.get_vector_store, retrieve.get_embeddings, retrieve.get_llm)
        retrieve.get_vector_store = lambda emb, ns: store
        retrieve.get_embeddings = lambda: None
        retrieve.get_llm = lambda: Chat()
        try:
            out = retrieve.ask_question("question", ["ns"], [])
        finally:
            retrieve.get_vector_store, retrieve.get_embeddings, retrieve.get_llm = saved
        self.assertEqual(out["chunkIds"], ["id-a", "id-b"])
        self.assertEqual(len(out["sources"]), 2)


class ChunkIds(unittest.TestCase):
    def test_ids_are_deterministic_and_input_sensitive(self):
        from app.ingest import make_chunk_id
        a = make_chunk_id("ns", "f.pdf", 3, "text")
        self.assertEqual(a, make_chunk_id("ns", "f.pdf", 3, "text"))
        self.assertNotEqual(a, make_chunk_id("ns2", "f.pdf", 3, "text"))
        self.assertNotEqual(a, make_chunk_id("ns", "f.pdf", 4, "text"))


if __name__ == "__main__":
    unittest.main()
