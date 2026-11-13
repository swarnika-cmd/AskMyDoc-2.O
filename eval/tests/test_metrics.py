import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics as m  # noqa: E402


class HitRank(unittest.TestCase):
    UNITS = ["alpha beta gamma", "foo   bar\nbaz qux", "nothing here", "x", "y", "foo bar baz late"]

    def test_first_hit_rank_is_one_based(self):
        self.assertEqual(m.hit_rank(self.UNITS, "alpha beta"), 1)

    def test_whitespace_is_normalised_on_both_sides(self):
        # unit 2 has a triple space and a newline; the quote has single spaces and edge whitespace
        self.assertEqual(m.hit_rank(self.UNITS, "  foo bar baz "), 2)

    def test_earliest_unit_wins_when_several_contain_the_quote(self):
        self.assertEqual(m.hit_rank(self.UNITS, "foo bar baz"), 2)

    def test_hit_beyond_top_k_is_a_miss(self):
        units = ["a", "b", "c", "d", "e", "the quote sits here"]
        self.assertIsNone(m.hit_rank(units, "the quote sits here", k=5))
        self.assertEqual(m.hit_rank(units, "the quote sits here", k=6), 6)

    def test_miss_and_empty_quote(self):
        self.assertIsNone(m.hit_rank(self.UNITS, "absent phrase"))
        self.assertIsNone(m.hit_rank(self.UNITS, ""))


class RecallMrr(unittest.TestCase):
    RANKS = [1, 3, None, 5, None, 6]

    def test_recall_at_5(self):
        # hits within 5: ranks 1, 3, 5 -> 3 of 6
        self.assertAlmostEqual(m.recall_at_k(self.RANKS, 5), 0.5)

    def test_recall_at_3(self):
        # ranks 1 and 3 -> 2 of 6
        self.assertAlmostEqual(m.recall_at_k(self.RANKS, 3), 2 / 6.0)

    def test_mrr_at_5(self):
        # 1/1 + 1/3 + 0 + 1/5 + 0 + 0 (rank 6 is outside k) = 1.5333.. over 6 items
        self.assertAlmostEqual(m.mrr_at_k(self.RANKS, 5), (1 + 1 / 3.0 + 0.2) / 6.0)

    def test_mrr_known_value(self):
        self.assertAlmostEqual(m.mrr_at_k([1, 2, None, 5]), 0.425)

    def test_empty_input(self):
        self.assertEqual(m.recall_at_k([]), 0.0)
        self.assertEqual(m.mrr_at_k([]), 0.0)


class Stats(unittest.TestCase):
    def test_percentile_interpolates(self):
        xs = [10, 20, 30, 40]
        self.assertAlmostEqual(m.percentile(xs, 50), 25.0)
        self.assertAlmostEqual(m.percentile(xs, 95), 38.5)
        self.assertAlmostEqual(m.percentile(xs, 0), 10.0)
        self.assertAlmostEqual(m.percentile(xs, 100), 40.0)

    def test_percentile_unsorted_and_single(self):
        self.assertAlmostEqual(m.percentile([40, 10, 30, 20], 50), 25.0)
        self.assertEqual(m.percentile([7], 95), 7.0)
        self.assertEqual(m.percentile([], 50), 0.0)

    def test_percentile_odd_count(self):
        self.assertAlmostEqual(m.percentile([1, 2, 3, 4, 5], 50), 3.0)
        self.assertAlmostEqual(m.percentile([1, 2, 3, 4, 5], 95), 4.8)

    def test_mean(self):
        self.assertAlmostEqual(m.mean([100, 300, 500]), 300.0)
        self.assertEqual(m.mean([]), 0.0)

    def test_summarise(self):
        s = m.summarise([1, None, 2, None], [1000, 2000, 3000, 2000], [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(s["n"], 4)
        self.assertAlmostEqual(s["recall_at_5"], 0.5)
        self.assertAlmostEqual(s["mrr_at_5"], (1 + 0.5) / 4)
        self.assertAlmostEqual(s["mean_context_chars"], 2000.0)
        self.assertAlmostEqual(s["latency_ms"]["p50"], 25.0)
        self.assertAlmostEqual(s["latency_ms"]["p95"], 38.5)


if __name__ == "__main__":
    unittest.main()
