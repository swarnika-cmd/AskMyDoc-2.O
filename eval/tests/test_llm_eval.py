import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm_eval as le  # noqa: E402

ITEM = {"id": "q1", "type": "factual", "question": "What is X?", "answer": "X is 5."}
UNANS = {"id": "q2", "type": "unanswerable", "question": "Who?", "answer": "Not stated in the documents."}


class Judge(unittest.TestCase):
    def test_judge_prompt_carries_question_reference_answer_and_exact_context(self):
        ctx = "[Source 1 - d.pdf]:\nX is 5 {context} {question}"
        msgs = le.judge_messages(ITEM, "It is 5.", ctx)
        user = msgs[1]["content"]
        for part in ("What is X?", "X is 5.", "It is 5.", ctx):
            self.assertIn(part, user)
        self.assertIn("answerable (factual)", user)
        self.assertIn("UNANSWERABLE", le.judge_messages(UNANS, "No.", ctx)[1]["content"])

    def test_parse_judge(self):
        good = json.dumps({"correctness": "partial", "correctness_reason": "r", "is_refusal": False,
                           "claims": [{"claim": "a", "supported": True}, {"claim": "b", "supported": False}]})
        p = le.parse_judge(good)
        self.assertTrue(p["parse_ok"])
        self.assertEqual(le.faithfulness(p["claims"]), 0.5)
        self.assertFalse(le.parse_judge('{"correctness": "maybe", "is_refusal": false}')["parse_ok"])
        self.assertFalse(le.parse_judge("not json")["parse_ok"])
        self.assertIsNone(le.faithfulness([]))

    def test_parse_generator(self):
        p = le.parse_generator('```json\n{"isConversational": false, "noEvidence": true, "answer": "I couldn\'t find"}\n```')
        self.assertTrue(p["parse_ok"] and p["no_evidence"])
        p = le.parse_generator("I couldn't find sufficient evidence")
        self.assertFalse(p["parse_ok"])
        self.assertTrue(p["no_evidence"])

    def test_cost(self):
        prices = {"prices_usd_per_1m_tokens": {"m": {"input": 0.1, "output": 0.4}}, "usd_to_inr": {"rate": 100.0}}
        # (1000 * 0.1 + 500 * 0.4) / 1e6 USD = 0.0003 USD = 0.03 INR
        self.assertAlmostEqual(le.cost_inr(prices, "m", {"prompt_tokens": 1000, "completion_tokens": 500}), 0.03)


if __name__ == "__main__":
    unittest.main()
