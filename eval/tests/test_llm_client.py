import glob
import json
import os
import sys
import tempfile
import unittest

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm_client as lc  # noqa: E402

FAKE_KEY = "test-key-not-a-real-credential-0123456789"
MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]


def ok_body(prompt=10, completion=5):
    return {"model": "m", "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}}


class FakeTime:
    def __init__(self):
        self.t = 1000.0
        self.slept = []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def make_client(handler, tmp, ft=None, **kw):
    ft = ft or FakeTime()
    c = lc.GroqClient(FAKE_KEY, tmp, transport=httpx.MockTransport(handler), clock=ft.clock, sleep=ft.sleep, **kw)
    return c, ft


class CacheKey(unittest.TestCase):
    def test_key_is_sha256_of_model_prompt_params(self):
        k = lc.cache_key("m", MSGS, {"temperature": 0})
        self.assertEqual(len(k), 64)
        self.assertEqual(k, lc.cache_key("m", [dict(x) for x in MSGS], {"temperature": 0}))
        self.assertNotEqual(k, lc.cache_key("m2", MSGS, {"temperature": 0}))
        self.assertNotEqual(k, lc.cache_key("m", MSGS, {"temperature": 1}))
        self.assertNotEqual(k, lc.cache_key("m", MSGS[:1], {"temperature": 0}))


class Parsing(unittest.TestCase):
    def test_parse_duration(self):
        self.assertAlmostEqual(lc.parse_duration("7.66s"), 7.66)
        self.assertAlmostEqual(lc.parse_duration("120ms"), 0.12)
        self.assertAlmostEqual(lc.parse_duration("1m30.5s"), 90.5)
        self.assertAlmostEqual(lc.parse_duration("12"), 12.0)
        self.assertIsNone(lc.parse_duration(None))

    def test_daily_limit_detection(self):
        self.assertEqual(lc.daily_limit_kind("... on tokens per day (TPD): Limit 200000, Used 199000 ..."), "tokens per day")
        self.assertEqual(lc.daily_limit_kind("... on requests per day (RPD): Limit 1000 ..."), "requests per day")
        self.assertIsNone(lc.daily_limit_kind("... on tokens per minute (TPM): Limit 8000 ..."))


class Pacing(unittest.TestCase):
    def test_second_call_waits_for_the_window(self):
        ft = FakeTime()
        p = lc.Pacer(budget=100, clock=ft.clock, sleep=ft.sleep)
        h = p.acquire(60)
        p.reconcile(h, 60)
        t0 = ft.t
        p.acquire(60)  # 60 + 60 > 100: must wait until the first event leaves the 60 s window
        self.assertGreaterEqual(ft.t - t0, 59.9)

    def test_fits_without_waiting(self):
        ft = FakeTime()
        p = lc.Pacer(budget=100, clock=ft.clock, sleep=ft.sleep)
        p.acquire(40)
        p.acquire(40)
        self.assertEqual(ft.slept, [])

    def test_reconcile_frees_overestimate(self):
        ft = FakeTime()
        p = lc.Pacer(budget=100, clock=ft.clock, sleep=ft.sleep)
        h = p.acquire(90)
        p.reconcile(h, 20)  # actual usage was lower
        p.acquire(70)
        self.assertEqual(ft.slept, [])

    def test_single_oversized_call_goes_through_on_an_empty_window(self):
        ft = FakeTime()
        p = lc.Pacer(budget=100, clock=ft.clock, sleep=ft.sleep)
        p.acquire(500)
        self.assertEqual(ft.slept, [])


class Client(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_miss_then_hit_makes_one_request_and_caches_without_the_key(self):
        calls = []

        def handler(req):
            calls.append(req)
            self.assertEqual(req.headers["authorization"], "Bearer " + FAKE_KEY)
            body = json.loads(req.content)
            self.assertEqual(body["temperature"], 0)
            self.assertEqual(body["reasoning_effort"], "low")
            return httpx.Response(200, json=ok_body(), headers={"x-ratelimit-remaining-tokens": "7000", "authorization": "x"})

        c, _ = make_client(handler, self.tmp)
        r1 = c.chat("openai/gpt-oss-20b", MSGS, 100)
        r2 = c.chat("openai/gpt-oss-20b", MSGS, 100)
        self.assertEqual(len(calls), 1)
        self.assertFalse(r1["cached"])
        self.assertTrue(r2["cached"])
        self.assertEqual(r2["usage"]["prompt_tokens"], 10)
        self.assertEqual(r2["rate_headers"], {"x-ratelimit-remaining-tokens": "7000"})
        for path in glob.glob(os.path.join(self.tmp, "**", "*.json"), recursive=True):
            text = open(path).read()
            self.assertNotIn(FAKE_KEY, text)
            self.assertNotIn("Bearer", text)
            self.assertNotIn("gs" + "k_", text)

    def test_429_per_minute_backs_off_honouring_retry_after_then_succeeds(self):
        n = {"i": 0}

        def handler(req):
            n["i"] += 1
            if n["i"] == 1:
                return httpx.Response(429, json={"error": {"message": "Rate limit reached ... on tokens per minute (TPM) ..."}},
                                      headers={"retry-after": "7"})
            return httpx.Response(200, json=ok_body())

        c, ft = make_client(handler, self.tmp)
        r = c.chat("m", MSGS, 100)
        self.assertEqual(n["i"], 2)
        self.assertEqual(r["attempts"], 2)
        self.assertEqual(r["retries"][0]["status"], 429)
        self.assertGreaterEqual(max(ft.slept), 7.0)
        self.assertEqual(c.n_429["m"], 1)

    def test_5xx_retries_with_backoff(self):
        n = {"i": 0}

        def handler(req):
            n["i"] += 1
            return httpx.Response(503) if n["i"] < 3 else httpx.Response(200, json=ok_body())

        c, ft = make_client(handler, self.tmp)
        r = c.chat("m", MSGS, 100)
        self.assertEqual(r["attempts"], 3)
        self.assertEqual(len(ft.slept) >= 2, True)

    def test_daily_limit_raises_and_does_not_retry(self):
        n = {"i": 0}

        def handler(req):
            n["i"] += 1
            return httpx.Response(429, json={"error": {"message": "Rate limit reached for model on tokens per day (TPD): Limit 200000, Used 199990"}})

        c, _ = make_client(handler, self.tmp)
        with self.assertRaises(lc.DailyLimitReached) as cm:
            c.chat("m", MSGS, 100)
        self.assertEqual(cm.exception.kind, "tokens per day")
        self.assertEqual(n["i"], 1)

    def test_client_error_is_not_retried(self):
        n = {"i": 0}

        def handler(req):
            n["i"] += 1
            return httpx.Response(400, json={"error": {"message": "bad request"}})

        c, _ = make_client(handler, self.tmp)
        with self.assertRaises(lc.LLMError):
            c.chat("m", MSGS, 100)
        self.assertEqual(n["i"], 1)

    def test_max_calls_counts_only_uncached_calls(self):
        c, _ = make_client(lambda req: httpx.Response(200, json=ok_body()), self.tmp, max_calls=1)
        c.chat("m", MSGS, 100)
        c.chat("m", MSGS, 100)  # cache hit: free
        with self.assertRaises(lc.CallBudgetExhausted):
            c.chat("m", [{"role": "user", "content": "other"}], 100)


class CommittedCache(unittest.TestCase):
    def test_no_credentials_in_committed_cache_or_results(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # split literals so this file itself does not trip the repository secret scan
        bad = ("gs" + "k_", "github" + "_pat_", "gh" + "p_", "Bearer ", "AI" + "za")
        for sub in ("cache", "results"):
            for dirpath, _, names in os.walk(os.path.join(root, sub)):
                for n in names:
                    with open(os.path.join(dirpath, n), encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    for b in bad:
                        self.assertNotIn(b, text, "%s contains %r" % (os.path.join(dirpath, n), b))


if __name__ == "__main__":
    unittest.main()
