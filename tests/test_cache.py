"""Crash-safe cell cache (ported from project A). No API calls."""
from ragas_eval.cache import CellCache


def _cache(tmp_path):
    return CellCache(tmp_path / "cells.jsonl", model_id="m", dhash="d", variant="full")


def test_set_get_roundtrip_and_fsync_persistence(tmp_path):
    c = _cache(tmp_path)
    c.set("GD-1", "faithfulness", 0.9, None, seconds=1.2)
    c.close()
    c2 = _cache(tmp_path)
    assert c2.get("GD-1", "faithfulness")["score"] == 0.9


def test_truncated_trailing_line_is_skipped_not_fatal(tmp_path):
    c = _cache(tmp_path)
    c.set("GD-1", "faithfulness", 0.9, None)
    c.close()
    with open(tmp_path / "cells.jsonl", "a", encoding="utf-8") as f:
        f.write('{"k": "partial-write')          # classic mid-append crash
    c2 = _cache(tmp_path)
    assert c2.corrupt_lines == 1
    assert c2.get("GD-1", "faithfulness")["score"] == 0.9


def test_drop_tombstones_and_compact_rewrites(tmp_path):
    c = _cache(tmp_path)
    c.set("GD-1", "faithfulness", None, "TimeoutError after 1800s")
    assert c.has("GD-1", "faithfulness")
    assert c.drop("GD-1", "faithfulness")
    assert not c.has("GD-1", "faithfulness")
    c.set("GD-2", "completeness", 0.7, None)
    assert c.compact() == 1
    c.close()
    c2 = _cache(tmp_path)
    assert not c2.has("GD-1", "faithfulness")
    assert c2.get("GD-2", "completeness")["score"] == 0.7


def test_key_isolation_by_model_and_dataset(tmp_path):
    a = CellCache(tmp_path / "c.jsonl", model_id="m1", dhash="d1")
    a.set("GD-1", "faithfulness", 0.5, None)
    a.close()
    b = CellCache(tmp_path / "c.jsonl", model_id="m2", dhash="d1")
    assert not b.has("GD-1", "faithfulness")     # different model -> cache miss
