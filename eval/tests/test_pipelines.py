import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipelines as p  # noqa: E402


class Rrf(unittest.TestCase):
    def test_hand_computed_scores(self):
        # k=60. a: 1/61 + 1/62 ; b: 1/62 + 1/61 (tie with a) ; c: 1/63 ; d: 1/63
        fused = p.rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
        self.assertEqual(fused[:2], ["a", "b"])       # tie -> first seen wins
        self.assertEqual(fused[2:], ["c", "d"])       # tie -> first seen wins

    def test_item_in_both_lists_beats_item_in_one(self):
        fused = p.rrf_fuse([["x", "shared"], ["shared", "y"]], k=60)
        self.assertEqual(fused[0], "shared")

    def test_deterministic(self):
        r = [["a", "b", "c"], ["c", "b", "a"]]
        self.assertEqual(p.rrf_fuse(r), p.rrf_fuse(r))

    def test_k_changes_weight(self):
        # rank 1 vs rank 2 in one list, nothing else: order preserved for any k
        self.assertEqual(p.rrf_fuse([["a", "b"]], k=1), ["a", "b"])


class Sizes(unittest.TestCase):
    def test_sizes_come_from_the_app(self):
        from app import ingest
        self.assertEqual((ingest.PARENT_CHUNK_SIZE, ingest.PARENT_CHUNK_OVERLAP), (3000, 300))
        self.assertEqual((ingest.CHILD_CHUNK_SIZE, ingest.CHILD_CHUNK_OVERLAP), (400, 100))
        self.assertIs(p.PARENT_CHUNK_SIZE, ingest.PARENT_CHUNK_SIZE)


class ParentDedup(unittest.TestCase):
    def test_dedup_keeps_rank_order(self):
        pc = p._ParentChild.__new__(p._ParentChild)
        pc.parents = {"P1": {"id": "P1"}, "P2": {"id": "P2"}}
        pc.child_by_id = {"a": {"parent": "P1"}, "b": {"parent": "P1"}, "c": {"parent": "P2"}}
        self.assertEqual([u["id"] for u in pc.to_parents(["a", "c", "b"])], ["P1", "P2"])
        self.assertEqual([u["id"] for u in pc.to_parents(["c", "a", "b"])], ["P2", "P1"])


class ModelSource(unittest.TestCase):
    def test_hub_id_pinned_to_revision_when_no_local_root(self):
        os.environ.pop("EVAL_MODEL_ROOT", None)
        src, kw = p.model_source(p.EMBED_MODEL_ID, p.EMBED_MODEL_REV)
        self.assertEqual(src, p.EMBED_MODEL_ID)
        self.assertEqual(kw, {"revision": p.EMBED_MODEL_REV})


class Determinism(unittest.TestCase):
    def test_repeated_builds_return_identical_results(self):
        class FakeEmbedder:
            """Deterministic 8-dim vectors from a text hash: no model needed."""
            def _v(self, t):
                import hashlib
                h = hashlib.sha256(t.encode()).digest()
                v = [b / 255.0 - 0.5 for b in h[:8]]
                n = sum(x * x for x in v) ** 0.5
                return [x / n for x in v]

            def embed_documents(self, texts):
                return [self._v(t) for t in texts]

            def embed_query(self, t):
                return self._v(t)

        pages = [{"doc": "d", "page": i, "text": ("Section %d of the rules. " % i) * 120 + "alpha%d beta gamma" % i} for i in range(1, 9)]
        emb = FakeEmbedder()
        runs = []
        for _ in range(2):
            pipe = p.build_pipeline("P3", pages, emb)
            runs.append([[u["id"] for u in pipe.retrieve("rules alpha%d" % i)] for i in range(1, 9)])
        self.assertEqual(runs[0], runs[1])
        self.assertTrue(all(len(r) <= p.TOP_K for r in runs[0]))


if __name__ == "__main__":
    unittest.main()
