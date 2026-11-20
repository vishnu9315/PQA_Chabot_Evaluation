"""Client-side pacing for Bedrock: avoid throttles rather than merely survive them.

The reference project (Ragas_Skill_Ministry_PSQ_Chatbot, ISSUE-4) let botocore's adaptive
retry be the pacing mechanism, which turned full runs into hours of back-off. This module
puts the control loop on our side of the wire, in the order the Session 06 spec asks for:

1. **Avoid.** A token bucket caps requests/minute *and* estimated tokens/minute, and an
   AIMD limiter caps in-flight calls — starting small (3–4), ramping +1 after a run of
   clean calls, and halving instantly on the first throttle.
2. **Handle.** Exponential backoff with **full jitter**, honouring a server-supplied
   `Retry-After`/`retryDelay` when present, plus a **fleet-wide cooldown**: one throttle
   pauses every worker briefly, because otherwise the other workers immediately re-trigger
   it and the pool never actually backs off.
3. **Observe.** Every wait, throttle and resize is counted, so the run summary can show
   whether the pacing is tuned right instead of leaving it to guesswork.

Everything here is thread-based, not asyncio-based: `ChatBedrockConverse` has no native
`_agenerate`, so LangChain runs the sync `_generate` in asyncio's default executor and the
calls we need to gate are real threads.
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field

# Botocore error codes that mean "slow down" rather than "this request is wrong".
THROTTLE_CODES = {
    "ThrottlingException", "TooManyRequestsException", "RequestLimitExceeded",
    "ProvisionedThroughputExceededException", "ServiceQuotaExceededException",
    "SlowDown", "LimitExceededException", "ModelTimeoutException",
}
# Transient transport failures: retryable with the same backoff, never a scoring failure.
TRANSIENT_MARKERS = (
    "readtimeout", "read timeout", "connectionerror", "connection reset",
    "connection aborted", "endpointconnectionerror", "serviceunavailable",
    "internalserverexception", "internalfailure", "503", "502", "504",
    "modelnotreadyexception", "connectionclosederror", "incompleteread",
)


def classify_exception(exc: BaseException) -> str:
    """Return 'throttle', 'transient' or 'fatal' for a boto/botocore exception."""
    code = ""
    resp = getattr(exc, "response", None)
    if isinstance(resp, dict):
        code = str(resp.get("Error", {}).get("Code", "") or "")
    name = type(exc).__name__
    if code in THROTTLE_CODES or name in THROTTLE_CODES:
        return "throttle"
    blob = f"{name} {code} {exc}".lower()
    if "throttl" in blob or "too many requests" in blob or "rate exceeded" in blob:
        return "throttle"
    if any(m in blob for m in TRANSIENT_MARKERS):
        return "transient"
    return "fatal"


def retry_after_seconds(exc: BaseException) -> float | None:
    """Server-supplied retry delay, if the response carries one."""
    resp = getattr(exc, "response", None)
    if not isinstance(resp, dict):
        return None
    meta = resp.get("ResponseMetadata") or {}
    headers = {str(k).lower(): v for k, v in (meta.get("HTTPHeaders") or {}).items()}
    for key in ("retry-after", "x-amzn-retry-after", "x-amz-retry-after"):
        if key in headers:
            try:
                return max(0.0, float(headers[key]))
            except (TypeError, ValueError):
                pass
    for key in ("retryDelaySeconds", "retryDelay"):
        if key in resp:
            try:
                return max(0.0, float(resp[key]))
            except (TypeError, ValueError):
                pass
    return None


# ---------------------------------------------------------------------------
class TokenBucket:
    """Continuously-refilling bucket. `take(n)` blocks until n units are available.

    Two of these run in series (requests and estimated tokens), so a burst of large
    prompts is paced by tokens/minute even when the request rate is well under the cap.
    """

    def __init__(self, rate_per_min: float, burst: float | None = None):
        self.rate = max(1e-9, float(rate_per_min)) / 60.0     # units per second
        self.capacity = float(burst) if burst is not None else max(1.0, float(rate_per_min))
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()
        self.total_wait_s = 0.0

    def _refill_locked(self) -> None:
        now = time.monotonic()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now

    def take(self, n: float) -> float:
        """Block until `n` units are available; return seconds waited."""
        n = min(float(n), self.capacity)         # never deadlock on an over-large request
        waited = 0.0
        while True:
            with self._lock:
                self._refill_locked()
                if self._tokens >= n:
                    self._tokens -= n
                    self.total_wait_s += waited
                    return waited
                deficit = n - self._tokens
                sleep_s = min(5.0, max(0.01, deficit / self.rate))
            time.sleep(sleep_s)
            waited += sleep_s


# ---------------------------------------------------------------------------
@dataclass
class PacingStats:
    requests: int = 0
    throttles: int = 0
    transients: int = 0
    retries: int = 0
    backoff_wait_s: float = 0.0
    bucket_wait_s: float = 0.0
    cooldown_wait_s: float = 0.0
    concurrency_increases: int = 0
    concurrency_decreases: int = 0
    max_concurrency_seen: int = 0
    resize_log: list = field(default_factory=list)   # (elapsed_s, limit, reason)

    def as_dict(self, elapsed_s: float | None = None) -> dict:
        d = {
            "requests": self.requests,
            "throttles": self.throttles,
            "transient_errors": self.transients,
            "retries": self.retries,
            "backoff_wait_s": round(self.backoff_wait_s, 2),
            "rate_limit_wait_s": round(self.bucket_wait_s, 2),
            "cooldown_wait_s": round(self.cooldown_wait_s, 2),
            "total_wait_s": round(self.backoff_wait_s + self.bucket_wait_s + self.cooldown_wait_s, 2),
            "concurrency_increases": self.concurrency_increases,
            "concurrency_decreases": self.concurrency_decreases,
            "max_concurrency_reached": self.max_concurrency_seen,
            "resize_log": [{"t": round(t, 1), "limit": l, "reason": r} for t, l, r in self.resize_log],
        }
        if elapsed_s and elapsed_s > 0:
            d["effective_requests_per_min"] = round(self.requests / (elapsed_s / 60.0), 2)
            d["throttle_rate"] = round(self.throttles / max(1, self.requests), 4)
        return d


class AdaptiveLimiter:
    """AIMD in-flight limiter + token buckets + fleet-wide cooldown.

    `run(fn, est_tokens=…)` is the single entry point: it paces, executes, retries
    throttles/transients with full-jitter backoff, and updates the concurrency limit.
    Fatal errors propagate immediately — they are the caller's to record as an NA.
    """

    def __init__(self, cfg, log=None):
        self.min_limit = max(1, int(cfg.pace_concurrency_min))
        self.max_limit = max(self.min_limit, int(cfg.pace_concurrency_max))
        self.limit = min(self.max_limit, max(self.min_limit, int(cfg.pace_concurrency_start)))
        self.successes_to_grow = max(1, int(cfg.pace_aimd_successes))
        self.cooldown_s = float(cfg.pace_cooldown_s)
        self.max_backoff_s = float(cfg.pace_max_backoff_s)
        self.max_attempts = max(1, int(cfg.pace_max_attempts))
        self._log = log or (lambda m: None)

        self._req_bucket = TokenBucket(cfg.pace_rpm, burst=max(2.0, cfg.pace_rpm / 6.0))
        self._tok_bucket = TokenBucket(cfg.pace_tpm, burst=max(1000.0, cfg.pace_tpm / 6.0))

        self._cv = threading.Condition()
        self._inflight = 0
        self._consecutive_ok = 0
        self._cooldown_until = 0.0
        self._t0 = time.monotonic()
        self.stats = PacingStats(max_concurrency_seen=0)
        self.stats.resize_log.append((0.0, self.limit, "start"))

    # -- concurrency ------------------------------------------------------
    def _acquire_slot(self) -> None:
        with self._cv:
            while self._inflight >= self.limit:
                self._cv.wait(timeout=1.0)
            self._inflight += 1
            self.stats.max_concurrency_seen = max(self.stats.max_concurrency_seen, self._inflight)

    def _release_slot(self) -> None:
        with self._cv:
            self._inflight -= 1
            self._cv.notify()

    def _grow(self) -> None:
        with self._cv:
            self._consecutive_ok += 1
            if self._consecutive_ok >= self.successes_to_grow and self.limit < self.max_limit:
                self.limit += 1
                self._consecutive_ok = 0
                self.stats.concurrency_increases += 1
                self.stats.resize_log.append((time.monotonic() - self._t0, self.limit, "aimd_increase"))
                self._cv.notify_all()
                self._log(f"pacing: concurrency +1 -> {self.limit} (clean run of "
                          f"{self.successes_to_grow})")

    def _shrink(self, reason: str) -> None:
        with self._cv:
            self._consecutive_ok = 0
            new = max(self.min_limit, self.limit // 2)
            if new != self.limit:
                self.limit = new
                self.stats.concurrency_decreases += 1
                self.stats.resize_log.append((time.monotonic() - self._t0, self.limit, reason))
                self._log(f"pacing: {reason} -> concurrency halved to {self.limit}")
            self._cooldown_until = max(self._cooldown_until, time.monotonic() + self.cooldown_s)

    def _await_cooldown(self) -> None:
        """Fleet-wide pause. One throttle stalls every worker, not just the one that hit it."""
        while True:
            with self._cv:
                remaining = self._cooldown_until - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, 1.0))
            self.stats.cooldown_wait_s += min(remaining, 1.0)

    # -- main entry point -------------------------------------------------
    def run(self, fn, est_tokens: float = 0.0):
        """Pace, execute `fn()`, retry throttles/transients, and adapt the limit."""
        attempt = 0
        while True:
            attempt += 1
            self._await_cooldown()
            self.stats.bucket_wait_s += self._req_bucket.take(1)
            if est_tokens:
                self.stats.bucket_wait_s += self._tok_bucket.take(est_tokens)
            self._acquire_slot()
            try:
                out = fn()
            except BaseException as exc:                    # noqa: BLE001 - classified below
                kind = classify_exception(exc)
                if kind == "fatal" or attempt >= self.max_attempts:
                    if kind == "throttle":
                        self.stats.throttles += 1
                        self._shrink("throttle")
                    elif kind == "transient":
                        self.stats.transients += 1
                    raise
                if kind == "throttle":
                    self.stats.throttles += 1
                    self._shrink("throttle")
                else:
                    self.stats.transients += 1
                    if attempt >= 3:
                        self._shrink("repeated_transient")
                self.stats.retries += 1
                delay = self._backoff(attempt, retry_after_seconds(exc))
                self._log(f"pacing: {kind} on attempt {attempt}/{self.max_attempts} "
                          f"({type(exc).__name__}); sleeping {delay:.1f}s")
                time.sleep(delay)
                self.stats.backoff_wait_s += delay
                continue
            else:
                self.stats.requests += 1
                # periodic liveness line: the watchdog proves calls are landing, but a log
                # that goes silent for hours is indistinguishable from a wedge when reading
                # it after the fact (observed 03:06-08:26 outage: zero lines, zero evidence)
                if self.stats.requests % 25 == 0:
                    self._log(f"pacing: {self.stats.requests} calls ok, limit "
                              f"{self.limit}, {self.stats.throttles} throttled, "
                              f"{self.stats.transients} transient")
                self._grow()
                return out
            finally:
                self._release_slot()

    def _backoff(self, attempt: int, server_hint: float | None) -> float:
        """Exponential backoff with FULL jitter, floored by any server-supplied delay."""
        ceiling = min(self.max_backoff_s, 1.0 * (2 ** (attempt - 1)))
        delay = random.uniform(0.0, ceiling)
        if server_hint:
            delay = max(delay, server_hint)
        return min(self.max_backoff_s, delay)

    # -- observability ----------------------------------------------------
    def snapshot(self) -> dict:
        return {"limit": self.limit, "inflight": self._inflight,
                **self.stats.as_dict(time.monotonic() - self._t0)}
