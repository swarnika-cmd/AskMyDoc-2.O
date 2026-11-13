"""Groq chat client (OpenAI-compatible endpoint) with a disk cache, a per-model token pacer and retries.

- Cache: eval/cache/<model>/<key>.json, key = sha256(model + "\\n" + full prompt + params). A hit costs nothing and
  makes reruns and resumes free. Records hold the prompt, the reply, usage, timings and rate-limit header values.
  They never hold the API key or any request header.
- Pacer: a sliding 60-second window of tokens per model (budget 8,000 TPM by default). Tokens are estimated before the
  call and reconciled with the usage the API returns.
- Retries: 429 (non-daily) and 5xx/transport errors back off exponentially with jitter and honour retry-after.
- Daily limits: a 429 that names tokens-per-day or requests-per-day raises DailyLimitReached (the caller stops cleanly).
"""
import collections
import hashlib
import json
import math
import os
import random
import re
import threading
import time

import httpx

BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_TPM_BUDGET = 8000
RATE_HEADERS = (
    "x-ratelimit-limit-requests", "x-ratelimit-limit-tokens", "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens", "retry-after",
)


class LLMError(Exception):
    pass


class DailyLimitReached(LLMError):
    def __init__(self, model, kind, message):
        super().__init__("%s: daily limit (%s)" % (model, kind))
        self.model, self.kind, self.message = model, kind, message


class CallBudgetExhausted(LLMError):
    pass


def cache_key(model, messages, params):
    """sha256(model + newline + full prompt + params). Prompt and params are serialised canonically."""
    prompt = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    p = json.dumps(params, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256((model + "\n" + prompt + p).encode("utf-8")).hexdigest()


def estimate_tokens(messages, chars_per_token=3.2):
    chars = sum(len(m.get("content", "")) + 8 for m in messages)
    return int(math.ceil(chars / chars_per_token))


def parse_duration(text):
    """'7.66s', '120ms', '1m30.5s', '2m' -> seconds, or None."""
    if text is None:
        return None
    total, found = 0.0, False
    for num, unit in re.findall(r"([0-9]*\.?[0-9]+)\s*(ms|h|m|s)", str(text)):
        found = True
        total += float(num) * {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]
    if found:
        return total
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def daily_limit_kind(body_text):
    """Return 'tokens per day', 'requests per day' or None for a 429 body."""
    t = (body_text or "").lower()
    if "tokens per day" in t or "(tpd)" in t:
        return "tokens per day"
    if "requests per day" in t or "(rpd)" in t:
        return "requests per day"
    return None


class Pacer:
    """Sliding-window token budget for one model. Thread-safe."""

    def __init__(self, budget=DEFAULT_TPM_BUDGET, window=60.0, clock=time.monotonic, sleep=time.sleep):
        self.budget, self.window, self.clock, self.sleep = budget, window, clock, sleep
        self.events = collections.deque()  # [timestamp, tokens]
        self.lock = threading.Lock()
        self.hold_until = 0.0
        self.waited = 0.0

    def _prune(self, now):
        while self.events and now - self.events[0][0] >= self.window:
            self.events.popleft()

    def _wait_needed(self, est, now):
        self._prune(now)
        used = sum(t for _, t in self.events)
        wait = max(0.0, self.hold_until - now)
        over = used + est - self.budget
        if over > 0 and self.events:
            freed = 0
            for ts, tok in self.events:
                freed += tok
                if freed >= over:
                    wait = max(wait, ts + self.window - now)
                    break
            else:
                wait = max(wait, self.events[-1][0] + self.window - now)
        return wait

    def acquire(self, est_tokens):
        """Block until est_tokens fits in the window, then reserve it. Returns a handle for reconcile()."""
        while True:
            with self.lock:
                now = self.clock()
                wait = self._wait_needed(est_tokens, now)
                if wait <= 0:
                    handle = [now, est_tokens]
                    self.events.append(handle)
                    return handle
            self.waited += wait
            self.sleep(min(wait, 5.0) + 0.01)

    def reconcile(self, handle, actual_tokens):
        with self.lock:
            handle[1] = actual_tokens

    def penalise(self, seconds):
        """After a 429: hold all calls for this many seconds."""
        with self.lock:
            self.hold_until = max(self.hold_until, self.clock() + seconds)


class GroqClient:
    def __init__(self, api_key, cache_dir, tpm_budget=DEFAULT_TPM_BUDGET, max_calls=None, max_attempts=8,
                 base_url=BASE_URL, transport=None, clock=time.monotonic, sleep=time.sleep, log=None):
        self._api_key = api_key
        self.cache_dir = cache_dir
        self.base_url = base_url
        self.max_calls = max_calls
        self.max_attempts = max_attempts
        self.sleep = sleep
        self.log = log or (lambda msg: None)
        self.pacers = collections.defaultdict(lambda: Pacer(tpm_budget, clock=clock, sleep=sleep))
        self.api_calls = collections.Counter()
        self.n_429 = collections.Counter()
        self.counter_lock = threading.Lock()
        self.ratio = {}  # model -> observed prompt tokens per char, for the estimator
        self.completion_mean = {}  # model -> running mean of completion tokens
        verify = os.environ.get("SSL_CERT_FILE") or True
        self.http = httpx.Client(timeout=httpx.Timeout(120.0, connect=15.0), transport=transport,
                                 **({} if transport else {"verify": verify}))

    # ---- cache -------------------------------------------------------------------------------------
    def _path(self, model, key):
        return os.path.join(self.cache_dir, model.replace("/", "__"), key + ".json")

    def cached(self, model, messages, params):
        p = self._path(model, cache_key(model, messages, params))
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return None

    def _store(self, model, record):
        p = self._path(model, record["key"])
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = "%s.%d.tmp" % (p, threading.get_ident())
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=1, ensure_ascii=False, sort_keys=True)
            f.write("\n")
        os.replace(tmp, p)

    # ---- call --------------------------------------------------------------------------------------
    def _estimate(self, model, messages, max_tokens):
        chars = sum(len(m.get("content", "")) + 8 for m in messages)
        r = self.ratio.get(model)
        prompt_est = int(math.ceil(chars * r * 1.05)) if r else estimate_tokens(messages)
        comp_est = int(self.completion_mean.get(model, min(max_tokens, 600)))
        return prompt_est, comp_est

    def chat(self, model, messages, max_completion_tokens, response_format=None, reasoning_effort="low",
             temperature=0):
        params = {"temperature": temperature, "reasoning_effort": reasoning_effort,
                  "max_completion_tokens": max_completion_tokens}
        if response_format:
            params["response_format"] = response_format
        key = cache_key(model, messages, params)
        hit = self.cached(model, messages, params)
        if hit is not None:
            hit = dict(hit, cached=True)
            return hit

        with self.counter_lock:
            if self.max_calls is not None and self.api_calls[model] >= self.max_calls:
                raise CallBudgetExhausted("max-calls reached for %s" % model)
            self.api_calls[model] += 1

        pacer = self.pacers[model]
        prompt_est, comp_est = self._estimate(model, messages, max_completion_tokens)
        handle = pacer.acquire(prompt_est + comp_est)
        waited_before = pacer.waited
        body = {"model": model, "messages": messages, **params}
        events = []
        for attempt in range(1, self.max_attempts + 1):
            t0 = time.perf_counter()
            try:
                resp = self.http.post(
                    self.base_url + "/chat/completions", json=body,
                    headers={"Authorization": "Bearer " + self._api_key, "Content-Type": "application/json"})
            except httpx.TransportError as e:
                events.append({"status": "transport:%s" % type(e).__name__})
                self._backoff(attempt, None, pacer, handle)
                continue
            latency = time.perf_counter() - t0
            headers = {h: resp.headers[h] for h in RATE_HEADERS if h in resp.headers}
            if resp.status_code == 200:
                data = resp.json()
                choice = data["choices"][0]
                usage = data.get("usage", {})
                record = {
                    "key": key, "model": model, "params": params, "messages": messages,
                    "content": choice["message"].get("content") or "",
                    "finish_reason": choice.get("finish_reason"),
                    "usage": usage,
                    "latency_s": round(latency, 3),
                    "attempts": attempt,
                    "retries": events,
                    "pacer_wait_s": round(pacer.waited - waited_before, 3),
                    "rate_headers": headers,
                    "served_model": data.get("model"),
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                actual = int(usage.get("total_tokens") or (prompt_est + comp_est))
                pacer.reconcile(handle, actual)
                if usage.get("prompt_tokens"):
                    chars = sum(len(m.get("content", "")) + 8 for m in messages)
                    self.ratio[model] = usage["prompt_tokens"] / float(chars)
                if usage.get("completion_tokens") is not None:
                    prev = self.completion_mean.get(model)
                    self.completion_mean[model] = usage["completion_tokens"] if prev is None else (0.7 * prev + 0.3 * usage["completion_tokens"])
                self._store(model, record)
                return dict(record, cached=False)
            text = resp.text
            if resp.status_code == 429:
                with self.counter_lock:
                    self.n_429[model] += 1
                kind = daily_limit_kind(text)
                if kind:
                    raise DailyLimitReached(model, kind, text[:300])
                events.append({"status": 429, "retry_after": headers.get("retry-after")})
                wait = parse_duration(headers.get("retry-after"))
                if wait is None:
                    m = re.search(r"try again in ([0-9hms.]+)", text)
                    wait = parse_duration(m.group(1)) if m else None
                self._backoff(attempt, wait, pacer, handle)
                continue
            if resp.status_code >= 500:
                events.append({"status": resp.status_code})
                self._backoff(attempt, parse_duration(headers.get("retry-after")), pacer, handle)
                continue
            pacer.reconcile(handle, 0)
            raise LLMError("HTTP %d from %s: %s" % (resp.status_code, model, text[:300]))
        pacer.reconcile(handle, 0)
        raise LLMError("gave up after %d attempts for %s (%s)" % (self.max_attempts, model, events[-1:] ))

    def _backoff(self, attempt, retry_after, pacer, handle):
        wait = min(60.0, 2.0 ** attempt) * random.uniform(0.5, 1.0)
        if retry_after:
            wait = max(wait, retry_after + random.uniform(0.0, 1.0))
        self.log("backoff %.1fs (attempt %d)" % (wait, attempt))
        pacer.penalise(wait)
        pacer.waited += wait
        self.sleep(wait)
