"""Out-of-band stall detector. The last line of defence against a wedged run.

Why the per-cell timeout is not enough
-------------------------------------
`runner._score_one` wraps each cell in `asyncio.wait_for(...)`, which looks like a hard
deadline and is not one. On timeout `wait_for` cancels the inner future and then **awaits
that cancellation**. When the coroutine is blocked on `loop.run_in_executor(...)` — which is
how every Bedrock call runs, because `ChatBedrockConverse` has no native `_agenerate` — the
future is already executing in a worker thread, cannot be cancelled, and `wait_for` waits for
it forever. A thread stuck in a native call (a DNS resolution with no timeout, a half-open
socket) therefore disables its own deadline.

This was not theoretical. A smoke run stalled at 02:24 on a DNS outage and sat at exactly zero
CPU for eight hours with 18 established connections, past an 8,460 s cell ceiling, printing
nothing. It looked identical to a run that was still working — which is the worst failure mode
a framework can have, and the one "no silent failures" is meant to exclude.

How this works
--------------
A daemon thread watches a heartbeat that every completed cell and every successful judge call
updates. If nothing at all makes progress for `stall_timeout` seconds, it prints a loud
diagnosis and calls `os._exit()`.

`os._exit` is deliberate. A stuck native call cannot be interrupted from Python: `sys.exit`
only unwinds the calling thread, and a normal shutdown would join the very worker threads that
are wedged. The cell cache is append-only and `fsync`'d per cell, so a hard exit loses at most
the in-flight cells and `--resume` continues from the last durable state. Dying loudly is
strictly better than hanging quietly.
"""
from __future__ import annotations

import os
import sys
import threading
import time

_LOCK = threading.Lock()
_STATE = {"last": None, "label": "startup", "cells": 0, "calls": 0, "started": None}
_THREAD = None
_STOP = threading.Event()


def heartbeat(label: str = "", *, cell: bool = False, call: bool = False) -> None:
    """Record progress. Cheap enough to call from every completed call."""
    with _LOCK:
        _STATE["last"] = time.monotonic()
        if label:
            _STATE["label"] = label
        if cell:
            _STATE["cells"] += 1
        if call:
            _STATE["calls"] += 1


def snapshot() -> dict:
    with _LOCK:
        last = _STATE["last"]
        return {"idle_s": None if last is None else round(time.monotonic() - last, 1),
                "last_progress": _STATE["label"], "cells_completed": _STATE["cells"],
                "calls_completed": _STATE["calls"]}


def start(stall_timeout: float, log=None, exit_code: int = 75) -> None:
    """Arm the watchdog. Idempotent; safe to call once per run."""
    global _THREAD
    if _THREAD is not None or not stall_timeout or stall_timeout <= 0:
        return
    _STOP.clear()
    with _LOCK:
        # counters are per-run: a fresh arm must not inherit a previous run's totals
        _STATE["last"] = time.monotonic()
        _STATE["started"] = time.monotonic()
        _STATE["label"] = "startup"
        _STATE["cells"] = 0
        _STATE["calls"] = 0
    emit = log or (lambda m: print(m, flush=True))

    def _watch():
        while not _STOP.wait(min(30.0, stall_timeout / 4.0)):
            with _LOCK:
                last = _STATE["last"]
                label = _STATE["label"]
                cells, calls = _STATE["cells"], _STATE["calls"]
            if last is None:
                continue
            idle = time.monotonic() - last
            if idle < stall_timeout:
                continue
            emit("")
            emit("=" * 78)
            emit(f"WATCHDOG: no progress for {idle / 60:.1f} min (limit {stall_timeout / 60:.0f} min).")
            emit(f"  last progress : {label}")
            emit(f"  completed     : {cells} cell(s), {calls} judge call(s)")
            emit("  A worker thread is stuck in a native call (DNS, half-open socket). "
                 "asyncio.wait_for cannot cancel an already-running executor job, so the "
                 "per-cell timeouts are disabled and this run would hang indefinitely.")
            emit("  Every completed cell is already fsync'd — re-run with --resume to "
                 "continue from the last durable state.")
            emit("=" * 78)
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
            os._exit(exit_code)

    _THREAD = threading.Thread(target=_watch, name="watchdog", daemon=True)
    _THREAD.start()
    emit(f"watchdog armed: abort if nothing completes for {stall_timeout / 60:.0f} min")


def stop() -> None:
    global _THREAD
    _STOP.set()
    _THREAD = None
