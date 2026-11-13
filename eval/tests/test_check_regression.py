import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import check_regression as cr  # noqa: E402


def result(**recalls):
    return {"meta": {"ids": ["a", "b"]}, "pipelines": {k: {"metrics": {"recall_at_5": v}} for k, v in recalls.items()}}


class Compare(unittest.TestCase):
    def test_equal_passes(self):
        rows, fails = cr.compare(result(P1=0.6, P2=0.8), result(P1=0.6, P2=0.8))
        self.assertEqual(fails, [])
        self.assertEqual(len(rows), 2)

    def test_improvement_passes(self):
        self.assertEqual(cr.compare(result(P1=0.6), result(P1=0.9))[1], [])

    def test_drop_of_exactly_three_points_passes(self):
        self.assertEqual(cr.compare(result(P1=0.80), result(P1=0.77))[1], [])

    def test_drop_over_three_points_fails_and_names_the_pipeline(self):
        _, fails = cr.compare(result(P1=0.80, P2=0.80), result(P1=0.80, P2=0.7692))
        self.assertEqual(len(fails), 1)
        self.assertIn("P2", fails[0])

    def test_missing_pipeline_fails(self):
        _, fails = cr.compare(result(P1=0.6, P2=0.8), result(P1=0.6))
        self.assertTrue(any("P2" in f for f in fails))

    def test_changed_subset_fails(self):
        cur = copy.deepcopy(result(P1=0.6))
        cur["meta"]["ids"] = ["a", "c"]
        _, fails = cr.compare(result(P1=0.6), cur)
        self.assertTrue(any("subset" in f for f in fails))


if __name__ == "__main__":
    unittest.main()
