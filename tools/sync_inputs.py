#!/usr/bin/env python3
"""
sync_inputs.py — keep `data/` in step with the authoring kit, and detect when it isn't.

`PQA_ragas/` carries its own copy of every input it evaluates, so the folder can be handed
to someone else and run as-is:

    data/artifacts/     the files the framework reads (dataset, judge prompts, and -
                        for a project that has one - freeze manifest, golden_v2 lineage,
                        KB snapshot sha)
    data/Extracted_KB/  a local KB document corpus, if the project uses one for the
                        parity gate / fabrication check's fingerprint

The cost of a copy is drift: if the dataset is AUTHORED in a separate `../golden_dataset_kit/`
checkout and consumed here, an edit there would otherwise leave this copy quietly a version
behind — and the parity gate would still pass, because it pins the sha recorded in the copy's
OWN freeze manifest. This script closes that gap. This project has no such upstream kit (its
dataset is authored directly in this repo, from the `documents/` folder), so
`upstream_present()` is always False here and this script no-ops - the machinery is retained
for a project set up with a separate authoring kit.

    python3 tools/sync_inputs.py --check     report drift, exit 1 if any (used by --validate)
    python3 tools/sync_inputs.py             copy upstream -> data/, reporting every change

When the authoring kit is not present — the shared/standalone case — there is nothing to
compare against and both modes report "no upstream" and succeed. That is the intended
behaviour: downstream users are not asked to obtain a kit they were never given.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ARTIFACTS = ROOT / "data" / "artifacts"
DATA_KB = ROOT / "data" / "Extracted_KB"
UP_ARTIFACTS = ROOT.parent / "golden_dataset_kit" / "artifacts"
UP_KB = ROOT.parent / "Extracted_KB"

FILES = ("golden_dataset_final_unified_eval.json", "judge_prompts_v1.md",
         "freeze_manifest.json", "golden_v2.json", "kb_snapshot.sha256")


def file_sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def kb_fingerprint(d: Path) -> str | None:
    """The locale-sensitive union recipe, byte-for-byte as the parity gate computes it:
    `cd <kb> && find . -name '*.md' | sort | xargs sha256sum | sha256sum`."""
    if not d.is_dir():
        return None
    out = subprocess.run(
        "find . -name '*.md' | sort | xargs sha256sum | sha256sum",
        shell=True, cwd=d, capture_output=True, text=True,
        env={**os.environ, "LC_ALL": "C"})
    return out.stdout.split()[0] if out.returncode == 0 and out.stdout else None


def upstream_present() -> bool:
    return UP_ARTIFACTS.is_dir() and UP_KB.is_dir()


def drift() -> list[tuple[str, str, str]]:
    """[(what, ours, theirs)] for every input that differs from upstream. Empty when in step."""
    out = []
    for name in FILES:
        ours, theirs = file_sha(DATA_ARTIFACTS / name), file_sha(UP_ARTIFACTS / name)
        if ours != theirs:
            out.append((name, ours or "MISSING", theirs or "MISSING"))
    ours, theirs = kb_fingerprint(DATA_KB), kb_fingerprint(UP_KB)
    if ours != theirs:
        out.append(("Extracted_KB/ (union fingerprint)", ours or "MISSING", theirs or "MISSING"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only; exit 1 on drift")
    args = ap.parse_args()

    if not upstream_present():
        print(f"[sync] no upstream authoring kit beside this folder ({UP_ARTIFACTS.parent}) — "
              f"nothing to compare. This is expected for a shared/standalone copy.")
        return 0

    d = drift()
    if not d:
        print(f"[sync] data/ is in step with {UP_ARTIFACTS.parent} "
              f"({len(FILES)} files + Extracted_KB)")
        return 0

    print(f"[sync] {len(d)} input(s) DIFFER from the authoring kit:")
    for what, ours, theirs in d:
        print(f"    {what}\n        data/   {ours[:16]}…\n        kit     {theirs[:16]}…")
    if args.check:
        print("[sync] run `python3 tools/sync_inputs.py` to bring data/ up to date, then "
              "re-run --validate (the parity gate pins the sha, so it must be re-checked)")
        return 1

    for name in FILES:
        src = UP_ARTIFACTS / name
        if src.is_file():
            shutil.copy2(src, DATA_ARTIFACTS / name)
    if UP_KB.is_dir():
        subprocess.run(["rsync", "-a", "--delete", f"{UP_KB}/", f"{DATA_KB}/"], check=True)
    left = drift()
    print(f"[sync] copied upstream -> data/; remaining differences: {len(left)}")
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main())
