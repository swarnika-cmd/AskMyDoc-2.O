import copy
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_golden as vg  # noqa: E402

PAGES = {
    ("d1", 1): "Interest rate on the loan is 7.5 per cent per annum, payable monthly.",
    ("d1", 2): "The   applicant must\nsubmit Form 12 within thirty days of the notice date.",
    ("d2", 1): "Penalty for late filing is Rs. 500 per day up to a maximum of Rs. 5,000 per return.",
}

GOOD = [
    {"id": "q1", "question": "What is the interest rate?", "answer": "7.5 per cent.", "doc": "d1", "page": 1,
     "supporting_quote": "Interest rate on the loan is 7.5 per cent per annum", "type": "factual"},
    {"id": "q2", "question": "Which form and by when?", "answer": "Form 12 within thirty days.", "doc": "d1", "page": 2,
     "supporting_quote": "must submit Form 12 within thirty days of the notice date.", "type": "table"},
    {"id": "q3", "question": "Late fee and cap?", "answer": "Rs. 500 per day, max Rs. 5,000.", "doc": "d2", "page": 1,
     "supporting_quote": "Penalty for late filing is Rs. 500 per day up to a maximum of Rs. 5,000", "type": "multi-hop",
     "extra_refs": [{"doc": "d1", "page": 1, "supporting_quote": "Interest rate on the loan is 7.5 per cent per annum"}]},
    {"id": "q4", "question": "Who is the governor?", "answer": "Not stated in the documents.", "doc": None, "page": None,
     "supporting_quote": "", "type": "unanswerable"},
]


def run(items, ci=None, total=4, unans=1, share=0.75, ci_size=2, ci_un=1):
    return vg.validate(items, PAGES, ci, expect_total=total, expect_unanswerable=unans, max_doc_share=share,
                       ci_size=ci_size, ci_unanswerable=ci_un)


class ValidateGolden(unittest.TestCase):
    def test_good_set_passes(self):
        fails, summary = run(copy.deepcopy(GOOD), ci=["q1", "q4"])
        self.assertEqual(fails, [])
        self.assertEqual(summary["by_type"]["unanswerable"], 1)

    def test_whitespace_normalisation(self):
        # q2's quote spans a newline and a triple space in the source page
        fails, _ = run(copy.deepcopy(GOOD))
        self.assertEqual(fails, [])

    def test_broken_quote_fails(self):
        items = copy.deepcopy(GOOD)
        items[0]["supporting_quote"] = "Interest rate on the loan is 9.5 per cent per annum"
        fails, _ = run(items)
        self.assertTrue(any("q1" in f and "not found verbatim" in f for f in fails), fails)

    def test_wrong_page_fails(self):
        items = copy.deepcopy(GOOD)
        items[0]["page"] = 2
        fails, _ = run(items)
        self.assertTrue(any("q1" in f for f in fails), fails)

    def test_short_quote_fails(self):
        items = copy.deepcopy(GOOD)
        items[0]["supporting_quote"] = "7.5 per cent"
        fails, _ = run(items)
        self.assertTrue(any("q1" in f and "length" in f for f in fails), fails)

    def test_bad_extra_ref_fails(self):
        items = copy.deepcopy(GOOD)
        items[2]["extra_refs"][0]["supporting_quote"] = "this sentence is not on the page at all, honestly"
        fails, _ = run(items)
        self.assertTrue(any("extra_refs" in f for f in fails), fails)

    def test_unanswerable_with_doc_fails(self):
        items = copy.deepcopy(GOOD)
        items[3]["doc"] = "d1"
        items[3]["supporting_quote"] = "something"
        fails, _ = run(items)
        self.assertTrue(any("q4" in f for f in fails), fails)

    def test_duplicate_id_and_count_fail(self):
        items = copy.deepcopy(GOOD)
        items[1]["id"] = "q1"
        fails, _ = run(items)
        self.assertTrue(any("duplicate id q1" in f for f in fails), fails)
        fails, _ = run(copy.deepcopy(GOOD)[:3])
        self.assertTrue(any("expected 4 items" in f for f in fails), fails)
        self.assertTrue(any("unanswerable" in f for f in fails), fails)

    def test_unknown_type_and_missing_field_fail(self):
        items = copy.deepcopy(GOOD)
        items[0]["type"] = "essay"
        del items[1]["answer"]
        fails, _ = run(items)
        self.assertTrue(any("type 'essay'" in f for f in fails), fails)
        self.assertTrue(any("missing fields answer" in f for f in fails), fails)

    def test_ci_subset_unknown_id_fails(self):
        fails, _ = run(copy.deepcopy(GOOD), ci=["q1", "q99"])
        self.assertTrue(any("q99" in f for f in fails), fails)

    def test_doc_share_limit(self):
        fails, _ = run(copy.deepcopy(GOOD), share=0.25)
        self.assertTrue(any("over 25%" in f for f in fails), fails)

    def test_deliberately_broken_item_makes_cli_fail(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            gp = os.path.join(tmp, "g.jsonl")
            pp = os.path.join(tmp, "p.jsonl")
            cp = os.path.join(tmp, "ci.txt")
            broken = copy.deepcopy(GOOD[:1])
            broken[0]["supporting_quote"] = "this text does not occur anywhere in the corpus pages"
            with open(gp, "w") as f:
                f.write(json.dumps(broken[0]) + "\n")
            with open(pp, "w") as f:
                for (d, p), t in PAGES.items():
                    f.write(json.dumps({"doc": d, "page": p, "text": t, "method": "pypdf"}) + "\n")
            with open(cp, "w") as f:
                f.write("# none\nq1\n")
            self.assertEqual(vg.main(["--golden", gp, "--pages", pp, "--ci", cp]), 1)


if __name__ == "__main__":
    unittest.main()
