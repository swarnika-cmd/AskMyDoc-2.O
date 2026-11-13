import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aggregate_llm as ag  # noqa: E402

PRICES = {"prices_usd_per_1m_tokens": {"openai/gpt-oss-20b": {"input": 0.1, "output": 0.4}, "openai/gpt-oss-120b": {"input": 0.1, "output": 0.4}},
          "usd_to_inr": {"rate": 100.0, "source": "t", "rate_date": "d"}, "prices_read_on": "d"}


def row(i, typ, status="ok", no_ev=False, corr="correct", judge_refusal=None, claims=(True,), hit=1, finish="stop"):
    r = {"id": i, "type": typ, "status": status, "completed": status != "error",
         "retrieval": {"hit_rank": hit, "retrieval_ms": 10.0},
         "generator": {"no_evidence": no_ev, "answer": "x", "usage": {"prompt_tokens": 1000, "completion_tokens": 100},
                       "latency_s": 1.0, "finish_reason": finish, "retries": [], "rate_headers": {}}}
    if status == "ok":
        refusal = no_ev if judge_refusal is None else judge_refusal
        r["judge"] = {"parse_ok": True, "correctness": corr, "is_refusal": refusal,
                      "claims": [{"claim": "c", "supported": s} for s in claims],
                      "faithfulness": (sum(claims) / len(claims)) if claims else None,
                      "usage": {"prompt_tokens": 2000, "completion_tokens": 200}, "latency_s": 1.0, "finish_reason": "stop"}
    if status == "judge_parse_error":
        r["judge"] = {"parse_ok": False, "usage": {"prompt_tokens": 2000, "completion_tokens": 200}, "finish_reason": "length"}
    if status == "error":
        r.pop("generator")
        r["error"] = {"stage": "generator", "class": "LLMError", "message": "HTTP 413"}
    return r


class Aggregate(unittest.TestCase):
    def test_counts_denominators_and_refusal_signal(self):
        rows = [
            row("a1", "factual", corr="correct", claims=(True, True)),
            row("a2", "factual", corr="partial", claims=(True, False)),
            row("a3", "table", no_ev=True, corr="incorrect", claims=()),          # false refusal
            row("a4", "table", status="no_answer"),                                 # counted incorrect, not a refusal
            row("a5", "factual", status="judge_parse_error"),                       # excluded from correctness and faithfulness
            row("a6", "factual", status="error"),                                   # no generator answer
            row("u1", "unanswerable", no_ev=True, corr="correct", claims=()),
            row("u2", "unanswerable", no_ev=False, corr="incorrect", claims=(False,), hit=None),
            row("u3", "unanswerable", no_ev=True, corr="correct", judge_refusal=False, claims=()),  # judge disagrees with the flag
        ]
        a = ag.aggregate(rows, [r["id"] for r in rows] + ["pending1"], PRICES, "P2", "T")
        k = a["counts"]
        self.assertEqual((k["ok"], k["no_answer"], k["judge_parse_error"], k["error"], k["pending"]), (6, 1, 1, 1, 1))
        # correctness over a1, a2, a3 (judged) + a4 (no_answer = incorrect): 1 correct, 1 partial, 2 incorrect
        c = a["correctness"]
        self.assertEqual((c["correct"], c["partial"], c["incorrect"], c["n"]), (1, 1, 2, 4))
        self.assertAlmostEqual(c["share_correct"], 0.25)
        # faithfulness over judged non-refusals with claims: a1 (1.0), a2 (0.5), a4 has no judge; u2 (0.0)
        self.assertEqual(a["faithfulness"]["n"], 3)
        self.assertAlmostEqual(a["faithfulness"]["mean"], 0.5)
        # refusal by the generator flag: 2 of 3 unanswerable refused; 1 of 5 answerable answered items refused (a3)
        rf = a["refusal"]
        self.assertEqual((rf["n_unanswerable"], rf["n_answerable"]), (3, 5))
        self.assertAlmostEqual(rf["accuracy_on_unanswerable"], 2 / 3.0)
        self.assertAlmostEqual(rf["false_refusal_rate_on_answerable"], 1 / 5.0)
        # judge cross-check over the 6 judged items: only u3 disagrees
        self.assertEqual(rf["judge_crosscheck"]["n"], 6)
        self.assertAlmostEqual(rf["judge_crosscheck"]["agreement"], 5 / 6.0)
        self.assertEqual(rf["judge_crosscheck"]["disagreements"], ["u3"])
        self.assertEqual(a["non_stop_finishes"]["counts"], {"generator": 0, "judge": 1})
        self.assertEqual(a["errors"][0]["id"], "a6")

    def test_rows_without_a_status_count_as_ok(self):
        r = row("a1", "factual")
        r.pop("status")
        self.assertEqual(ag.row_status(r), "ok")
        r["completed"] = False
        self.assertEqual(ag.row_status(r), "error")


if __name__ == "__main__":
    unittest.main()
