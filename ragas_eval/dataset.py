"""Load, validate and route this project's unified evaluation dataset. No API calls.

The run universe is EXACTLY `config.py`'s `unified_eval_json` (currently
`meity-pq-SHORT-2part-v1`, 66 rows - built by `tools/build_unified_eval.py` from
capture_new_chatbot.py's output; see golden_dataset_README.md). The unified file is
self-contained: every row carries BOTH the golden scoring key (reference,
reference_contexts, must_include_facts, expected_sources, expected_behavior, stratum,
category, difficulty, house) AND the already-captured chatbot output (response,
retrieved_contexts, cited_sources). There is no separate responses file and no join step
for responses.

Routing (derived from the fields - the ragas_eligible flag alone is never trusted):
  * judge phase        — ALL rows;
  * full-RAGAS         — ragas_eligible AND non-empty retrieved_contexts;
  * partial-RAGAS      — ragas_eligible with EMPTY retrieved_contexts: only Answer
                         Correctness, Answer Relevancy and Semantic Similarity run;
                         Faithfulness / Context Precision / Context Recall are emitted as
                         NA with reason "skipped: empty_retrieved_contexts" — never 0.0,
                         never dropped;
  * judge-only         — abstains (ragas_eligible=false). This dataset currently has none.

unsupported_aspects are carried directly on each row (Sample.unsupported_aspects_source
will read "own_row" for all of them) - unlike a differently-shaped dataset where they'd
need joining from a separate lineage file; that join machinery (_join_unsupported,
UNSUPPORTED_JOIN_IDS) is retained as a fallback, not exercised by this dataset.

Provenance (Sample.provenance) is this dataset's own Part 1 (ingested) / Part 2
(not-ingested) split (golden_dataset_README.md section 1/3), read from each row's
`provenance.part` - NOT the GD-V1-/GD-S12- id-prefix scheme kept only as a fallback for a
dataset shaped like the one this framework was originally built for.

The 4-point parity gate (run inside --validate, no API) is SKIPPED for this project until
it does its own freeze (no freeze_manifest.json yet - see config.py's
`parity_gate_available`); see that property's docstring for what it pins once one exists.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

from .kb_index import load_kb_index

_REQUIRED_ROW = ("id", "user_input", "reference", "expected_behavior", "must_include_facts",
                 "expected_sources", "reference_contexts", "response", "retrieved_contexts",
                 "cited_sources", "ragas_eligible", "stratum", "category", "difficulty",
                 "house")

# Rows that are RAGAS-eligible but context-metric-ineligible (empty retrieved_contexts).
# EMPTY for this dataset: every row's capture carries genuine retrieved_contexts (see
# chatbot_endpoint_config.json's `known_from_live_probe`). Kept as fallback machinery for a
# differently-shaped capture where some rows legitimately have no retrieval.
EMPTY_RETRIEVAL_IDS = ()

# Rows whose retrieved_contexts came from a verbatim/fallback answer rather than genuine
# reranked extracts, making their context metrics PARTLY SELF-REFERENTIAL and not comparable
# to other rows. EMPTY for this dataset: no row trips that fallback. Kept as fallback
# machinery; if populated, --validate/report.md/dashboard.html footnote these rows while still
# including them in every mean and the composite.
FULL_ANSWER_CONTEXT_IDS = ()
# Row ids whose unsupported_aspects must be joined from a separate lineage file by id.
# EMPTY for this dataset: every row carries unsupported_aspects directly (Sample.
# unsupported_aspects_source reads "own_row" for all of them) - see _join_unsupported.
# The join machinery is retained as a fallback, not exercised by this dataset.
UNSUPPORTED_JOIN_IDS = ()


class ParityError(RuntimeError):
    """A parity-gate check failed — the inputs are not the frozen ones; refuse to score."""


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Sample:
    id: str
    user_input: str
    response: str
    retrieved_contexts: list
    reference: str
    reference_contexts: list
    expected_behavior: str
    stratum: str
    category: str
    difficulty: str
    house: str
    question_origin: str
    must_include_facts: list
    expected_sources: list
    cited_sources: list
    ragas_eligible: bool
    unsupported_aspects: list = field(default_factory=list)
    # Where unsupported_aspects actually came from - "own_row" | "erratum" | "golden_v2_join"
    # | "none". Tracked explicitly (not inferred from unsupported_aspects/erratum alone)
    # because inferring it broke the moment a project's own rows started carrying the
    # field directly (TEMPLATE: this project's do; the previous project's didn't) - see
    # cli.py's cmd_validate, which used to assume "has it + no erratum => must be joined".
    unsupported_aspects_source: str = "none"
    legacy_source: Optional[dict] = None
    # set when the unified file registers a deliberate correction diverging from frozen
    # golden_v2 (see the row's `erratum` block and validate_unified_eval.py's V2 check)
    erratum: Optional[dict] = None
    # This project's own grouping (golden_dataset_README.md section 1/3): "1-ingested" |
    # "2-not-ingested" | None, read straight from the row's provenance.part. None for a
    # row that doesn't carry one (e.g. a future dataset without this split, or the
    # synthetic fixtures in tests/fixtures.py).
    part: Optional[str] = None
    # Follow-up pairs (golden_dataset_README.md section 3.2): an -F2 row was asked in the
    # same chat right after its -F1, so it is only meaningful with that earlier turn.
    follow_up_of: Optional[str] = None
    prior_turn: Optional[dict] = None     # {"id", "user_input", "response"} of the -F1 row
    query_style: Optional[str] = None     # provenance.query_style: Verbatim / Natural / ...

    @property
    def scoring_user_input(self) -> str:
        """user_input as RAGAS should see it: a follow-up carries its earlier question, so
        relevancy/precision/recall judge it against what was actually being asked."""
        if not self.prior_turn:
            return self.user_input
        return (f"Earlier question in this conversation: {self.prior_turn['user_input']}\n"
                f"Follow-up question: {self.user_input}")

    @property
    def provenance(self) -> str:
        """The label every report/dashboard/aggregate table slices by FIRST (config.py's
        `segments`, always provenance-first - see report.py/aggregate.py/dashboard.py).

        This project's rows all carry `part` (see golden_dataset_README.md section 3:
        "report Part 1 and Part 2 separately, never averaged together") - that split IS
        this project's provenance, and takes priority. The GD-V1-/GD-S12- prefix checks
        below are kept only as a fallback for a differently-shaped dataset (e.g. the
        previous project's own id scheme) that has no `part` field at all.
        """
        if self.part == "1-ingested":
            return "part1-ingested"
        if self.part == "2-not-ingested":
            return "part2-not-ingested"
        if self.id.startswith("GD-V1-"):
            return "legacy-v1"
        if self.id.startswith("GD-S12-"):
            return "rebuilt-s12"
        return "pilot-v2"

    @property
    def has_empty_retrieved_contexts(self) -> bool:
        return len(self.retrieved_contexts) == 0

    @property
    def route(self) -> str:
        """full_ragas | partial_ragas | judge_only — derived, never read off a flag alone."""
        if self.expected_behavior == "abstain" or not self.ragas_eligible:
            return "judge_only"
        if self.has_empty_retrieved_contexts:
            return "partial_ragas"
        return "full_ragas"

    def segment(self, name: str) -> str:
        return str(getattr(self, name, None))

    def to_ragas_kwargs(self) -> dict:
        return {
            "user_input": self.scoring_user_input,
            "response": self.response,
            "retrieved_contexts": list(self.retrieved_contexts),   # rank order preserved
            "reference": self.reference,
            "reference_contexts": list(self.reference_contexts),
        }

    def judge_item(self) -> dict:
        """The field dict shown to the LLM judges (judge_runner assembles per-metric views)."""
        return {
            "id": self.id, "user_input": self.user_input, "response": self.response,
            "reference": self.reference, "retrieved_contexts": list(self.retrieved_contexts),
            "cited_sources": list(self.cited_sources),
            "expected_behavior": self.expected_behavior,
            "must_include_facts": list(self.must_include_facts),
            "unsupported_aspects": list(self.unsupported_aspects),
            "expected_sources": list(self.expected_sources),
            "reference_contexts": list(self.reference_contexts),
            "prior_turn": self.prior_turn,
            "part": self.part,
        }


# ---------------------------------------------------------------------------
def _cfg():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config import load_config
    return load_config()


def dataset_hash(path) -> str:
    """SHA-256 (16 hex chars) of the dataset bytes — the cache invalidation key."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_payload(path=None) -> dict:
    """The raw unified file including its `note` and `amendments` blocks."""
    p = Path(path) if path is not None else _cfg().unified_eval_json
    data = json.loads(p.read_text(encoding="utf-8"))
    for key in ("note", "amendments", "rows"):
        if key not in data:
            raise ValueError(f"{p.name}: missing top-level '{key}' block — this is not the "
                             f"amended 45-row unified eval file the S10 spec pins")
    return data


def _join_unsupported(rows: list, golden_path: Path) -> dict:
    """{id: unsupported_aspects} for answer_partial rows that do NOT already carry their
    own `unsupported_aspects` field directly on the row (the previous project's rows
    didn't; this project's do - see load_samples' TEMPLATE note). Only attempts to open
    `golden_path` (a separate lineage file, e.g. golden_v2.json) when at least one row
    genuinely needs the join - so a dataset where every row is self-contained, like this
    one, never requires that file to exist at all."""
    want = {r["id"] for r in rows
            if not r["id"].startswith("GD-V1-") and r.get("expected_behavior") == "answer_partial"
            and not r.get("unsupported_aspects")}
    if not want:
        return {}
    golden = {g["id"]: g for g in
              json.loads(golden_path.read_text(encoding="utf-8"))["dataset"]}
    out = {}
    for rid in sorted(want):
        g = golden.get(rid)
        if g is None:
            raise ValueError(f"unsupported_aspects join: {rid} not found in golden_v2.json")
        out[rid] = list(g.get("unsupported_aspects") or [])
    return out


def load_samples(path=None, golden_path=None) -> list:
    """Parse the unified file into Samples, enforcing structural invariants (raise on
    breach) and joining unsupported_aspects for the GD-#### answer_partial rows."""
    cfg = None
    if path is None or golden_path is None:
        cfg = _cfg()
    p = Path(path) if path is not None else cfg.unified_eval_json
    gp = Path(golden_path) if golden_path is not None else cfg.golden_v2_json

    data = load_payload(p)
    rows = data["rows"]
    if data.get("n_rows") != len(rows):
        raise ValueError(f"{p.name}: n_rows={data.get('n_rows')} but rows carries {len(rows)}")

    joined = _join_unsupported(rows, gp)

    samples: list = []
    seen: set = set()
    for i, r in enumerate(rows):
        where = f"{p.name} rows[{i}] (id={r.get('id')})"
        for k in _REQUIRED_ROW:
            if k not in r:
                raise ValueError(f"{where}: missing required key '{k}'")
        for k in ("id", "user_input", "reference", "response", "expected_behavior"):
            if not isinstance(r[k], str):
                raise ValueError(f"{where}: '{k}' must be a string")
        for k in ("retrieved_contexts", "reference_contexts", "cited_sources"):
            if not (isinstance(r[k], list) and all(isinstance(x, str) for x in r[k])):
                raise ValueError(f"{where}: '{k}' must be a list[str]")
        for k in ("must_include_facts", "expected_sources"):
            if not isinstance(r[k], list):
                raise ValueError(f"{where}: '{k}' must be a list")
        if r["id"] in seen:
            raise ValueError(f"{where}: duplicate id")
        seen.add(r["id"])

        samples.append(Sample(
            id=r["id"], user_input=r["user_input"], response=r["response"],
            retrieved_contexts=r["retrieved_contexts"], reference=r["reference"],
            reference_contexts=r["reference_contexts"],
            expected_behavior=r["expected_behavior"], stratum=str(r.get("stratum")),
            category=str(r.get("category")), difficulty=str(r.get("difficulty")),
            house=str(r.get("house")), question_origin=str(r.get("question_origin")),
            must_include_facts=r["must_include_facts"],
            expected_sources=r["expected_sources"], cited_sources=r["cited_sources"],
            ragas_eligible=bool(r["ragas_eligible"]),
            # TEMPLATE: the previous project's rows never carried unsupported_aspects
            # directly - only a registered erratum's own block, or the golden_v2 join,
            # could supply them. This project's rows DO carry it directly (confirmed:
            # all answer_partial rows have it) - so that's now the first-choice source,
            # with erratum/join kept as fallbacks for a row that doesn't have it inline
            # (e.g. a future erratum re-authoring a row, the same way the previous
            # project used erratum to override a stale golden_v2 join - see the
            # superseded comment this replaced).
            unsupported_aspects=(r.get("unsupported_aspects")
                                 or (r.get("erratum") or {}).get("unsupported_aspects")
                                 or joined.get(r["id"])
                                 or []),
            unsupported_aspects_source=(
                "own_row" if r.get("unsupported_aspects") else
                "erratum" if (r.get("erratum") or {}).get("unsupported_aspects") else
                "golden_v2_join" if joined.get(r["id"]) else
                "none"),
            legacy_source=r.get("legacy_source"), erratum=r.get("erratum"),
            part=(r.get("provenance") or {}).get("part"),
            follow_up_of=(r.get("provenance") or {}).get("follow_up_of"),
            query_style=(r.get("provenance") or {}).get("query_style")))

    by_id = {s.id: s for s in samples}
    idx = load_kb_index((cfg or _cfg()).documents_dir)
    out = []
    for s in samples:
        changes: dict = {}
        if s.follow_up_of:
            first = by_id.get(s.follow_up_of)
            if first is None:
                raise ValueError(f"{s.id}: follow_up_of={s.follow_up_of} is not in the dataset")
            changes["prior_turn"] = {"id": first.id, "user_input": first.user_input,
                                     "response": first.response}
        if idx is not None:
            resolved = []
            for e in s.expected_sources:
                cid = idx.resolve(e) if isinstance(e, dict) else None
                resolved.append({**e, "document": cid, "document_as_authored": e.get("document")}
                                if cid else e)
            changes["expected_sources"] = resolved
        out.append(replace(s, **changes) if changes else s)
    return out


# ---------------------------------------------------------------------------
def routing(samples: list) -> dict:
    """Route every sample and cross-check the derivation against the file's own flags and
    the amendments block's carve-out list. Raises on any disagreement — the flag alone is
    never trusted, but a flag that CONTRADICTS the derivation means the file changed."""
    routes = {"full_ragas": [], "partial_ragas": [], "judge_only": []}
    for s in samples:
        routes[s.route].append(s.id)

    problems = []
    derived_partial = set(routes["partial_ragas"])
    if derived_partial != set(EMPTY_RETRIEVAL_IDS):
        # not fatal by itself — a re-captured file legitimately removes the carve-out —
        # but it must be surfaced loudly (S10: "if a re-captured file removes the gap,
        # the carve-out disappears and validate prints 39/0/2").
        problems.append(
            f"partial-RAGAS set derived from the data is {sorted(derived_partial)}, the "
            f"amendments block records {sorted(EMPTY_RETRIEVAL_IDS)} — if retrieval was "
            f"re-captured this is expected; otherwise investigate")
    for s in samples:
        if (s.expected_behavior == "abstain") != (not s.ragas_eligible):
            raise ValueError(f"{s.id}: ragas_eligible={s.ragas_eligible} contradicts "
                             f"expected_behavior={s.expected_behavior}")
    return {"routes": routes,
            "counts": {k: len(v) for k, v in routes.items()},
            "notes": problems}


def validate(samples: list, expected_n: Optional[int] = None) -> list:
    """Soft warnings. Empty list == clean. No API calls."""
    warnings: list = []
    if expected_n is not None and len(samples) != expected_n:
        warnings.append(f"expected {expected_n} samples, found {len(samples)}")
    for s in samples:
        for fname in ("user_input", "response", "reference"):
            if not getattr(s, fname).strip():
                warnings.append(f"{s.id}: empty '{fname}'")
        if s.route == "partial_ragas":
            warnings.append(f"{s.id}: EMPTY retrieved_contexts — Faithfulness/Context "
                            f"Precision/Context Recall will be NA (\"empty retrieved_contexts\"), "
                            f"never 0.0; re-capture retrieval to lift the carve-out")
        if s.expected_behavior == "abstain" and s.cited_sources:
            warnings.append(f"{s.id}: abstain row carries citations "
                            f"{s.cited_sources} — Sources Validation will treat them as "
                            f"fabrication per the rubric")
        if s.stratum in ("None", ""):
            warnings.append(f"{s.id}: stratum missing — per-stratum tables will bucket it "
                            f"under 'None'")
        if not s.must_include_facts and s.expected_behavior != "abstain":
            warnings.append(f"{s.id}: no must_include_facts on a non-abstain row")
        if s.id in FULL_ANSWER_CONTEXT_IDS:
            warnings.append(
                f"{s.id}: retrieved_contexts came from the complete event's `full_answer` "
                f"(verbatim-gate row emits no extracts), which is ~99.8% similar to the row's "
                f"own response — faithfulness / context_precision / context_recall here are "
                f"PARTLY SELF-REFERENTIAL and not comparable to the extract-based rows")
    # the unsupported_aspects join contract
    for rid in UNSUPPORTED_JOIN_IDS:
        s = next((x for x in samples if x.id == rid), None)
        if s is None:
            warnings.append(f"{rid}: expected GD-#### answer_partial row not present")
        elif not s.unsupported_aspects:
            warnings.append(f"{rid}: unsupported_aspects join from golden_v2 produced nothing")
    for s in samples:
        if (s.expected_behavior == "answer_partial"   # noqa: E129
                and s.id not in UNSUPPORTED_JOIN_IDS):
            # TEMPLATE: a row carrying unsupported_aspects directly (this project's
            # normal case - see Sample.unsupported_aspects_source) is outside the
            # golden_v2 join list BY DESIGN, same as an erratum row - neither is a
            # warning-worthy situation, unlike a row with genuinely nothing.
            if s.unsupported_aspects_source == "own_row":
                pass
            elif s.erratum:
                warnings.append(f"{s.id}: answer_partial by REGISTERED ERRATUM "
                                f"({s.erratum.get('date')}); {len(s.unsupported_aspects)} "
                                f"unsupported_aspects read from the erratum block, not the "
                                f"golden_v2 join")
            else:
                warnings.append(f"{s.id}: GD-#### answer_partial row outside the documented "
                                f"join list {UNSUPPORTED_JOIN_IDS}")
    return warnings


# ---------------------------------------------------------------------------
# parity gate (no API)
# ---------------------------------------------------------------------------
def kb_fingerprint(kb_dir: Path) -> str:
    """Recompute the union KB fingerprint with the EXACT recipe that produced the recorded
    one: relative paths from inside Extracted_KB, shell `sort` under the user's locale.
    (A pure-Python sorted() differs — GNU sort's locale collation orders `./x` paths
    differently — so the gate shells out to the very pipeline that minted the value.)"""
    out = subprocess.run(
        "find . -name '*.md' | sort | xargs sha256sum | sha256sum",
        shell=True, cwd=str(kb_dir), capture_output=True, text=True, check=True)
    return out.stdout.split()[0]


def parity_gate(cfg) -> dict:
    """The 4-point parity gate. Raises ParityError naming every failed check."""
    manifest = json.loads(cfg.freeze_manifest_json.read_text(encoding="utf-8"))
    amendment = manifest["amendments"][-1]
    expected_unified = amendment["unified_eval"]["sha256_after_amendment"]
    expected_kb = cfg.kb_snapshot_sha_file.read_text(encoding="utf-8").split()[0]
    expected_prompts = manifest["components_versioned_together"]["judge_prompts"]["sha256"]
    expected_golden = manifest["components_versioned_together"]["dataset"]["sha256"] \
        if "dataset" in manifest.get("components_versioned_together", {}) else None

    checks = []
    got_unified = file_sha256(cfg.unified_eval_json)
    checks.append(("unified eval file sha256", expected_unified, got_unified))
    got_kb = kb_fingerprint(cfg.kb_dir)
    checks.append(("KB union fingerprint", expected_kb, got_kb))
    got_prompts = file_sha256(cfg.judge_prompts_md)
    checks.append(("judge_prompts_v1.md sha256", expected_prompts, got_prompts))
    got_golden = file_sha256(cfg.golden_v2_json)
    if expected_golden is None:
        # fall back to the sha recorded inside the unified file itself
        expected_golden = load_payload(cfg.unified_eval_json)["golden_sha256"]
    checks.append(("golden_v2.json sha256 (lineage of the 25 GD-#### rows)",
                   expected_golden, got_golden))

    failures = [f"{name}: expected {exp[:16]}…, got {got[:16]}…"
                for name, exp, got in checks if exp != got]
    if failures:
        raise ParityError(
            "PARITY GATE FAILED — the on-disk inputs are not the frozen ones this framework "
            "was pinned to; scoring them would produce unattributable numbers.\n  - "
            + "\n  - ".join(failures)
            + "\n(Note: the historical 221-doc fingerprint 50cf039… is the golden_v2 "
              "authoring record and is deliberately NOT checked against the on-disk KB.)")
    return {name: {"sha": got, "ok": True} for name, _, got in checks}


def kb_document_names(cfg) -> Optional[set]:
    """The Sources-Validation fabrication floor: canonical ids of every KB document.

    Primary source is cfg.documents_dir (this project's ingested corpus, ids in the
    chatbot's citation form - see kb_index.py); falls back to basenames of .md files under
    cfg.kb_dir. Returns None (not an empty set) when neither exists: None means "cannot
    check fabrication" (attribution.py/judge_runner.py skip that sub-check), whereas set()
    would mean "a real, empty corpus" and flag every citation as fabricated.
    """
    idx = load_kb_index(cfg.documents_dir)
    if idx is not None:
        return idx.names
    if not Path(cfg.kb_dir).is_dir():
        return None
    return {p.name for p in Path(cfg.kb_dir).rglob("*.md")}


def stats(samples: list) -> dict:
    n_ctx = [len(s.retrieved_contexts) for s in samples]
    return {
        "n_samples": len(samples),
        # This project's real split (golden_dataset_README.md section 1/3).
        "part1_ingested": sum(1 for s in samples if s.provenance == "part1-ingested"),
        "part2_not_ingested": sum(1 for s in samples if s.provenance == "part2-not-ingested"),
        # Kept for a differently-shaped dataset with no `part` field (see Sample.provenance) -
        # always 0 for this project, since every row here carries `part`.
        "pilot_v2": sum(1 for s in samples if s.provenance == "pilot-v2"),
        "legacy_v1": sum(1 for s in samples if s.provenance == "legacy-v1"),
        "rebuilt_s12": sum(1 for s in samples if s.provenance == "rebuilt-s12"),
        "retrieved_chunks_total": sum(n_ctx),
        "retrieved_chunks_max": max(n_ctx) if n_ctx else 0,
        "reference_contexts_total": sum(len(s.reference_contexts) for s in samples),
        "response_chars_max": max((len(s.response) for s in samples), default=0),
        "reference_chars_max": max((len(s.reference) for s in samples), default=0),
    }
