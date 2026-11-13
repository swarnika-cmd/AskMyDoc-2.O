"""run_llm.py records every item: refusals, empty answers, judge parse errors and API errors are never dropped."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm_client as lc  # noqa: E402
import llm_eval as le  # noqa: E402
import run_llm as rl  # noqa: E402

IDS = ["q010", "q003", "q015", "q016", "q019"]  # answerable, unanswerable, answerable, answerable, answerable
USAGE = {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}


def rec(content, finish="stop"):
    return {"key": "k" + str(abs(hash(content)) % 10**6), "content": content, "finish_reason": finish, "usage": USAGE,
            "latency_s": 0.5, "retries": [], "rate_headers": {}, "cached": False}


GOOD_JUDGE = json.dumps({"correctness": "correct", "correctness_reason": "r", "is_refusal": False,
                         "claims": [{"claim": "c", "supported": True}]})
REFUSAL_JUDGE = json.dumps({"correctness": "incorrect", "correctness_reason": "r", "is_refusal": True, "claims": []})


class FakeClient:
    """Scripted by question text: generator behaviour per id, judge behaviour per id."""
    judge_calls = []

    def __init__(self, *a, **k):
        import collections
        self.api_calls = collections.Counter()
        self.n_429 = collections.Counter()

    def cached(self, model, messages, params):
        return None

    def chat(self, model, messages, max_tokens, response_format=None):
        text = messages[-1]["content"]
        if model == le.GENERATOR_MODEL:
            q = text
            if q == QUESTIONS["q010"]:   # model says no evidence, with its own wording
                return rec(json.dumps({"isConversational": False, "noEvidence": True, "answer": "Not in the document, sorry."}))
            if q == QUESTIONS["q003"]:   # empty answer, not a refusal
                return rec(json.dumps({"isConversational": False, "noEvidence": False, "answer": ""}))
            if q == QUESTIONS["q015"]:
                return rec(json.dumps({"isConversational": False, "noEvidence": False, "answer": "ANSWER15"}))
            if q == QUESTIONS["q016"]:
                raise lc.LLMError("HTTP 413 from m: request too large")
            return rec(json.dumps({"isConversational": False, "noEvidence": False, "answer": "ANSWER19"}))
        FakeClient.judge_calls.append(text)
        if "ANSWER15" in text:
            return rec("this is not json")
        if "ANSWER19" in text:
            raise RuntimeError("boom")
        return rec(REFUSAL_JUDGE)


QUESTIONS = {}


class RunLlm(unittest.TestCase):
    def test_every_item_is_recorded_with_a_status(self):
        from validate_golden import load_golden
        golden = {g["id"]: g for g in load_golden(os.path.join(rl.HERE, "golden.jsonl"))}
        QUESTIONS.update({i: golden[i]["question"] for i in IDS})
        FakeClient.judge_calls = []
        tmp = tempfile.mkdtemp()
        ids_file = os.path.join(tmp, "ids.txt")
        open(ids_file, "w").write("\n".join(IDS))
        unit = {"id": "d:p1:P0", "doc": "d", "page": 1, "text": "context text"}
        saved = (rl.pl.Embedder, rl.pl.build_pipeline, rl.pl.run_queries, rl.lc.GroqClient)
        rl.pl.Embedder = lambda: None
        rl.pl.build_pipeline = lambda *a, **k: None
        rl.pl.run_queries = lambda pipe, qs: [([unit], 1.0) for _ in qs]
        rl.lc.GroqClient = FakeClient
        os.environ["GROQ_API_KEY"] = "test-key-not-real"
        try:
            code = rl.main(["--pipeline", "P2", "--ids", ids_file, "--results-dir", tmp, "--cache-dir", tmp])
        finally:
            rl.pl.Embedder, rl.pl.build_pipeline, rl.pl.run_queries, rl.lc.GroqClient = saved
        rows = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(tmp, "raw", "llm_P2.jsonl"))}
        self.assertEqual(set(rows), set(IDS))  # nothing dropped

        # no-evidence reply: the app's fixed refusal text is what gets judged
        r = rows["q010"]
        self.assertEqual(r["status"], "ok")
        self.assertTrue(r["generator"]["no_evidence"])
        self.assertEqual(r["generator"]["answer"], le.REFUSAL_TEXT)
        self.assertEqual(r["generator"]["raw_answer"], "Not in the document, sorry.")
        self.assertTrue(any(le.REFUSAL_TEXT in t for t in FakeClient.judge_calls))

        # empty answer that is not a refusal: recorded, completed, no judge call
        r = rows["q003"]
        self.assertEqual((r["status"], r["completed"]), ("no_answer", True))
        self.assertNotIn("judge", r)
        self.assertFalse(r["generator"]["no_evidence"])

        # unparseable judge reply
        r = rows["q015"]
        self.assertEqual((r["status"], r["completed"]), ("judge_parse_error", True))
        self.assertFalse(r["judge"]["parse_ok"])
        self.assertIn("not json", r["judge"]["raw_reply"])

        # non-retryable generator error and an unexpected judge exception: recorded with class, retried next run
        r = rows["q016"]
        self.assertEqual((r["status"], r["completed"]), ("error", False))
        self.assertEqual(r["error"]["stage"], "generator")
        self.assertEqual(r["error"]["class"], "LLMError")
        self.assertIn("413", r["error"]["message"])
        r = rows["q019"]
        self.assertEqual((r["status"], r["error"]["stage"], r["error"]["class"]), ("error", "judge", "RuntimeError"))

        prog = json.load(open(os.path.join(tmp, "llm_progress.json")))["pipelines"]["P2"]
        self.assertEqual(sorted(prog["completed_ids"]), ["q003", "q010", "q015"])
        self.assertEqual(sorted(prog["pending_ids"]), ["q016", "q019"])
        self.assertEqual(code, 1)  # errors remain


if __name__ == "__main__":
    unittest.main()
