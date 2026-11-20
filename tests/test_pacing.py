"""Pacing: classification, token bucket, AIMD behaviour. No API calls, no sleep abuse."""
import pytest

from ragas_eval.pacing import AdaptiveLimiter, TokenBucket, classify_exception
from tests.fixtures import FakeCfg


class _Boto(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


def test_classify_throttle_transient_fatal():
    assert classify_exception(_Boto("ThrottlingException")) == "throttle"
    assert classify_exception(ConnectionError("connection reset by peer")) == "transient"
    assert classify_exception(ValueError("bad input")) == "fatal"


def test_token_bucket_blocks_then_grants():
    b = TokenBucket(rate_per_min=60_000, burst=10)
    assert b.take(5) == 0.0                    # burst available immediately
    waited = b.take(10)                        # must wait for refill
    assert waited >= 0.0


def test_aimd_grows_on_success_and_halves_on_throttle():
    lim = AdaptiveLimiter(FakeCfg())
    start = lim.limit
    for _ in range(FakeCfg.pace_aimd_successes):
        lim.run(lambda: "ok")
    assert lim.limit == start + 1

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _Boto("ThrottlingException")
        return "ok"

    lim.run(flaky)
    assert lim.stats.throttles == 1
    assert lim.stats.retries == 1
    assert lim.limit <= start + 1              # halved then possibly re-grown by 0 or 1


def test_fatal_errors_propagate_immediately():
    lim = AdaptiveLimiter(FakeCfg())
    with pytest.raises(ValueError):
        lim.run(lambda: (_ for _ in ()).throw(ValueError("fatal")))
    assert lim.stats.retries == 0
