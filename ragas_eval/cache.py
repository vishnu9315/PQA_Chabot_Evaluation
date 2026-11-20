"""Crash-safe, content-addressed per-(sample, metric) cell cache.

The reference project held the whole cache in memory and wrote one JSON file at the end of
a run, so a mid-run crash lost every cell computed since the last `save()`. The Session 06
spec requires the opposite: *"the cache must be written per-cell as results arrive, so a
mid-run crash loses at most the in-flight cell."*

So the store is an **append-only JSONL journal**, one line per cell, `flush()` +
`os.fsync()` on every append. Recovery is a replay where the last line for a key wins;
`compact()` rewrites the journal atomically once a run finishes. A truncated final line
(the classic mid-write crash) is skipped on load rather than aborting recovery.

Keys hash `sample_id | metric | model_id | dataset_hash | variant`, so the cache is safe to
share across runs and invalidates by construction when the data, the model or the context
variant changes. `--resume` is therefore just "run again": every finished cell is a hit and
costs nothing.

**The key does NOT cover the thinking effort or the judge prompt mode.** Two cells scored at
`effort=max` and `effort=medium`, or under `stock` and `no_apparatus`, collide on one key and
the first one written wins forever. This is not theoretical: the whole `context_recall` column
was scored at `effort=max` during the 2026-08-04 full run and silently survived the switch to
`medium`, so the reported matrix mixes efforts across metrics (RCA_CONTEXT_RECALL.md section 1).

Consequences for anyone changing effort or a judge prompt:
  * `--resume` will NOT re-score the affected cells — it cannot see that anything changed;
  * you must drop them explicitly first, e.g.
    `cli.py --clear-cells --metrics context_recall`, which backs the cache up before mutating;
  * the resolved effort and both judge modes are recorded per run in `config_snapshot.json`
    (`runner._judge_provenance`), so a finished run can always be attributed after the fact —
    that record is the only thing standing between you and an unattributable column.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Optional


def cell_key(sample_id: str, metric: str, model_id: str, dhash: str, variant: str = "") -> str:
    raw = f"{sample_id}\x1f{metric}\x1f{model_id}\x1f{dhash}\x1f{variant}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


class CellCache:
    """Append-only, fsync-per-write cell store."""

    def __init__(self, path, model_id: str = "", dhash: str = "", variant: str = ""):
        self.path = Path(path)
        self.model_id = model_id
        self.dhash = dhash
        self.variant = variant
        self._data: dict = {}
        self._lock = threading.Lock()
        self._fh = None
        self.corrupt_lines = 0
        self._load()

    # -- persistence ------------------------------------------------------
    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError:
                    # A crash mid-append leaves one partial trailing line. Skip it; every
                    # complete line before it is still valid, which is the whole point.
                    self.corrupt_lines += 1
                    continue
                k = rec.get("k")
                if k:
                    self._data[k] = rec

    def _open(self):
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")
        return self._fh

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.flush()
                os.fsync(self._fh.fileno())
                self._fh.close()
                self._fh = None

    # -- api --------------------------------------------------------------
    def _k(self, sample_id: str, metric: str) -> str:
        return cell_key(sample_id, metric, self.model_id, self.dhash, self.variant)

    def get(self, sample_id: str, metric: str) -> Optional[dict]:
        rec = self._data.get(self._k(sample_id, metric))
        return None if (rec is not None and rec.get("__deleted__")) else rec

    def has(self, sample_id: str, metric: str) -> bool:
        return self.get(sample_id, metric) is not None

    def is_scored(self, sample_id: str, metric: str) -> bool:
        rec = self.get(sample_id, metric)
        return bool(rec) and rec.get("score") is not None

    def set(self, sample_id: str, metric: str, score, na_reason=None, **extra) -> dict:
        """Write one cell through to disk immediately (flush + fsync)."""
        rec = {"k": self._k(sample_id, metric), "id": sample_id, "metric": metric,
               "score": None if score is None else float(score),
               "na_reason": na_reason, **extra}
        with self._lock:
            self._data[rec["k"]] = rec
            fh = self._open()
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return rec

    def drop(self, sample_id: str, metric: str) -> bool:
        """Tombstone a cell so it is recomputed. Used by `--clear-na` / retry escalation."""
        k = self._k(sample_id, metric)
        if k not in self._data:
            return False
        with self._lock:
            self._data.pop(k, None)
            fh = self._open()
            fh.write(json.dumps({"k": k, "__deleted__": True}) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return True

    def cells(self) -> list:
        return [v for v in self._data.values() if not v.get("__deleted__")]

    def compact(self) -> int:
        """Atomically rewrite the journal with one line per live cell."""
        with self._lock:
            if self._fh is not None:
                self._fh.flush()
                os.fsync(self._fh.fileno())
                self._fh.close()
                self._fh = None
            live = [v for v in self._data.values() if not v.get("__deleted__")]
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                for rec in live:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
            self._data = {r["k"]: r for r in live}
            return len(live)

    def __len__(self) -> int:
        return len(self.cells())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
