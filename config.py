"""Central configuration for the MeitY PQA RAGAS + LLM-judge evaluation framework.

Built from an earlier Commerce-ministry evaluation project's framework as reference (see
README.md); this project and its dataset are independent of that one.

Everything provider / model / path / pacing / cost related lives here so the rest of the
package stays config-driven. Settings load from this project's `.env` (git-ignored);
`python-dotenv` is used when installed, with a tiny built-in parser as a fallback so this
file runs before `pip install -r requirements.txt`.

Inputs are READ-ONLY. Nothing in this package writes outside its own project directory.

The evaluation universe is EXACTLY the rows of your own
`data/artifacts/<unified_eval>.json` (filename customizable below via `unified_eval_json`
- it's self-contained: golden scoring key + captured chatbot output per row. An optional
golden_v2.json-style file can be used for a parity/lineage check and an unsupported_aspects
join (see dataset.py) if your project needs that; otherwise ignore it.

Run directly to print the resolved configuration (no API calls):

    .venv/Scripts/python.exe config.py
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent

# Judge model pinned by the S10 spec (Claude Sonnet on Bedrock + Titan V2 embeddings).
PINNED_MODEL_ID = "global.anthropic.claude-sonnet-5"
PINNED_EMBED_ID = "amazon.titan-embed-text-v2:0"

# Efforts Bedrock accepts for this model, weakest -> strongest (probed by the Parivesh
# project, 2026-08-03).
THINKING_EFFORTS = ("low", "medium", "high", "xhigh", "max")


# ---------------------------------------------------------------------------
# .env loading (python-dotenv preferred; minimal fallback otherwise)
# ---------------------------------------------------------------------------
def _load_env(env_path: Path) -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(env_path)           # does not override existing env vars
        return
    except ModuleNotFoundError:
        pass
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.split(" #", 1)[0].strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), val)


_load_env(ROOT / ".env")


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _i(name: str, default: int) -> int:
    return int(_f(name, default))


def _s(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or v.strip() == "" else v.strip()


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    # --- AWS / Bedrock ---
    aws_region: str = field(default_factory=lambda: _s("AWS_REGION", "ap-south-1"))
    bedrock_model_id: str = field(default_factory=lambda: _s("BEDROCK_MODEL_ID", PINNED_MODEL_ID))
    bedrock_embed_model_id: str = field(
        default_factory=lambda: _s("BEDROCK_EMBED_MODEL_ID", PINNED_EMBED_ID))

    # --- extended thinking ---
    # Probed against this model+region (Parivesh project): thinking.type must be "adaptive";
    # the effort knob is the SIBLING key output_config.effort. Default effort is medium —
    # reasoning and the answer share ONE max_tokens budget on Bedrock, and at `max` the judge
    # can spend the entire budget thinking and emit no text at all (reference ISSUE-P10).
    thinking_mode: str = field(default_factory=lambda: _s("BEDROCK_THINKING", "adaptive").lower())
    thinking_effort: str = field(default_factory=lambda: _s("BEDROCK_THINKING_EFFORT", "medium").lower())
    thinking_budget: int = field(default_factory=lambda: _i("BEDROCK_THINKING_BUDGET", 24576))

    # Output ceiling. On Bedrock max_tokens caps TOTAL output INCLUDING reasoning tokens
    # (reference ISSUE-9c); long answers decompose into long claim/NLI lists (annexure tables),
    # so this sits near the model's output limit. A ceiling, not a reservation.
    ragas_max_tokens: int = field(default_factory=lambda: _i("RAGAS_MAX_TOKENS", 64000))

    # --- pacing: AIMD adaptive concurrency + client-side token buckets (pacing.py) ---
    pace_concurrency_start: int = field(default_factory=lambda: _i("PACE_CONCURRENCY_START", 3))
    pace_concurrency_max: int = field(default_factory=lambda: _i("PACE_CONCURRENCY_MAX", 10))
    pace_concurrency_min: int = field(default_factory=lambda: _i("PACE_CONCURRENCY_MIN", 1))
    pace_rpm: float = field(default_factory=lambda: _f("PACE_RPM", 60.0))
    pace_tpm: float = field(default_factory=lambda: _f("PACE_TPM", 200_000.0))
    pace_aimd_successes: int = field(default_factory=lambda: _i("PACE_AIMD_SUCCESSES", 4))
    pace_cooldown_s: float = field(default_factory=lambda: _f("PACE_COOLDOWN_S", 8.0))
    pace_max_backoff_s: float = field(default_factory=lambda: _f("PACE_MAX_BACKOFF_S", 120.0))
    pace_max_attempts: int = field(default_factory=lambda: _i("PACE_MAX_ATTEMPTS", 8))

    # --- timeouts / retries ---
    # Base per-cell ceiling, for metrics whose cost does not depend on chunk count.
    ragas_timeout: int = field(default_factory=lambda: _i("RAGAS_TIMEOUT", 1800))
    # Context Precision issues one LLM verdict per retrieved chunk SERIALLY inside the cell
    # (reference ISSUE-P7/P12): its deadline is max(ragas_timeout, per_chunk_timeout*n_chunks).
    # MeitY captures carry up to 10 retrieved chunks per row, so 300s x 10 stays well under
    # the cap; the shaping guards against bigger retrieval dumps.
    per_chunk_timeout: int = field(default_factory=lambda: _i("PER_CHUNK_TIMEOUT", 300))
    bedrock_read_timeout: int = field(default_factory=lambda: _i("BEDROCK_READ_TIMEOUT", 900))
    bedrock_connect_timeout: int = field(default_factory=lambda: _i("BEDROCK_CONNECT_TIMEOUT", 60))
    ragas_retry_rounds: int = field(default_factory=lambda: _i("RAGAS_RETRY_ROUNDS", 3))
    judge_retry_rounds: int = field(default_factory=lambda: _i("JUDGE_RETRY_ROUNDS", 2))
    # Out-of-band stall detector (watchdog.py): asyncio.wait_for cannot cancel a running
    # executor job, so a worker wedged in a native call disables every per-cell deadline.
    stall_timeout: int = field(default_factory=lambda: _i("STALL_TIMEOUT", 1800))
    # (sample, metric) cells driven concurrently; each LLM call is additionally gated by
    # the AIMD limiter, so this is a fan-out ceiling, not a rate.
    cell_concurrency: int = field(default_factory=lambda: _i("CELL_CONCURRENCY", 12))
    executor_threads: int = field(default_factory=lambda: _i("EXECUTOR_THREADS", 64))
    boto_pool: int = field(default_factory=lambda: _i("BOTO_POOL", 64))
    judge_max_workers: int = field(default_factory=lambda: _i("JUDGE_MAX_WORKERS", 6))

    # --- paths (inputs are READ-ONLY) ---
    root: Path = ROOT
    artifacts_dir: Path = field(default_factory=lambda: Path(
        _s("MOC_ARTIFACTS_DIR", str(ROOT / "data" / "artifacts"))).resolve())
    kb_dir: Path = field(default_factory=lambda: Path(
        _s("MOC_KB_DIR", str(ROOT / "data" / "Extracted_KB"))).resolve())
    runs_dir: Path = field(default_factory=lambda: Path(
        _s("MOC_RUNS_DIR", str(ROOT / "runs"))).resolve())
    # The chatbot's ingested corpus (every file here is in the KB) - see ragas_eval/kb_index.py.
    documents_dir: Path = field(default_factory=lambda: Path(
        _s("PQA_DOCUMENTS_DIR", str(ROOT / "documents"))).resolve())

    # This project's dataset size (meity-pq-SHORT-2part-v1); override with MOC_EXPECTED_N.
    expected_sample_count: int = field(default_factory=lambda: _i("MOC_EXPECTED_N", 66))

    # --- report segments. Provenance FIRST: Part 1 (ingested) and Part 2 (not ingested)
    # must always be readable separately, never averaged together. ---
    segments: tuple = ("provenance", "expected_behavior", "difficulty", "stratum",
                      "category", "house")

    # --- cost rates ($ per 1K tokens; tracked and reported, NEVER gated) ---
    rate_llm_in_per_1k: float = field(default_factory=lambda: _f("RATE_LLM_IN_PER_1K", 0.003))
    rate_llm_out_per_1k: float = field(default_factory=lambda: _f("RATE_LLM_OUT_PER_1K", 0.015))
    rate_embed_per_1k: float = field(default_factory=lambda: _f("RATE_EMBED_PER_1K", 0.00002))

    # stability check: max tolerated spread across smoke repeats on a composite metric
    stability_tolerance: float = field(default_factory=lambda: _f("STABILITY_TOLERANCE", 0.05))

    # --- Answer Correctness weighting (OWNER OVERRIDE, 2026-08-20) -------------------------
    # ragas ships weights [0.75, 0.25] over [claim-F1, semantic similarity] and beta=1.0
    # (precision and recall equal) inside the F-score. Both are overridden here on the owner's
    # explicit instruction. This is a deliberate departure from the project rule that a ported
    # metric's scoring maths stays untouched, so both values are stamped on every cell and into
    # config_snapshot.json — a score is meaningless if you cannot tell which weighting produced
    # it. Rationale in RCA_ANSWER_CORRECTNESS.md.
    #
    # ac_factuality_weight: share given to claim-F1; semantic similarity gets the remainder.
    ac_factuality_weight: float = field(
        default_factory=lambda: _f("AC_FACTUALITY_WEIGHT", 0.5))
    # ac_recall_weight: share of the F-score given to claim RECALL, precision gets the rest.
    # ragas takes `beta`, not a weight, so this is converted below.
    ac_recall_weight: float = field(default_factory=lambda: _f("AC_RECALL_WEIGHT", 0.6))

    # --- Answer Correctness derived weighting ---
    @property
    def ac_weights(self) -> list:
        """[factuality, semantic_similarity], summing to 1."""
        w = self.ac_factuality_weight
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"AC_FACTUALITY_WEIGHT must be in [0,1], got {w}")
        return [w, 1.0 - w]

    @property
    def ac_beta(self) -> float:
        """ragas F-score beta giving claim recall a share of `ac_recall_weight`.

        F_beta = (1+b^2)PR / (b^2 P + R) is the weighted harmonic mean of precision and
        recall with recall weight b^2/(1+b^2). Solving for a target recall share r gives
        b = sqrt(r / (1-r)) — so r=0.5 -> b=1.0 (ragas default) and r=0.6 -> b=1.2247.
        Returned as a real float: ragas' __post_init__ rejects anything whose
        `type(...) is not float`, so an int would raise.
        """
        import math
        r = self.ac_recall_weight
        if not 0.0 < r < 1.0:
            raise ValueError(f"AC_RECALL_WEIGHT must be strictly between 0 and 1, got {r}")
        return float(math.sqrt(r / (1.0 - r)))

    # --- derived input paths ---
    @property
    def unified_eval_json(self) -> Path:
        return self.artifacts_dir / "golden_dataset_final_unified_eval.json"

    @property
    def judge_prompts_md(self) -> Path:
        return self.artifacts_dir / "judge_prompts_v1.md"

    @property
    def freeze_manifest_json(self) -> Path:
        return self.artifacts_dir / "freeze_manifest.json"

    @property
    def golden_v2_json(self) -> Path:
        return self.artifacts_dir / "golden_v2.json"

    @property
    def parity_gate_available(self) -> bool:
        """False until this project does its own freeze (writes freeze_manifest.json -
        see the previous project's workflow for the shape). cli.py skips the parity gate
        for --all/--smoke while this is False, rather than crashing on a FileNotFoundError
        for a freeze that was never done, and automatically starts enforcing it again the
        moment someone creates a real freeze_manifest.json - no flag to remember to flip."""
        return self.freeze_manifest_json.exists()

    @property
    def kb_snapshot_sha_file(self) -> Path:
        return self.artifacts_dir / "kb_snapshot.sha256"

    @property
    def cache_dir(self) -> Path:
        return self.runs_dir / "_cache"

    @property
    def has_aws_credentials(self) -> bool:
        return bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))

    @property
    def thinking_enabled(self) -> bool:
        return self.thinking_mode in ("adaptive", "enabled")

    def additional_model_request_fields(self) -> dict | None:
        """The exact `additionalModelRequestFields` payload sent on every judge call."""
        if self.thinking_mode == "adaptive":
            return {"thinking": {"type": "adaptive"},
                    "output_config": {"effort": self.thinking_effort}}
        if self.thinking_mode == "enabled":
            return {"thinking": {"type": "enabled", "budget_tokens": self.thinking_budget}}
        return None

    def issues(self) -> list[str]:
        """Non-fatal configuration problems to surface (no API calls)."""
        problems: list[str] = []
        if not self.has_aws_credentials:
            problems.append("AWS credentials not set in .env")
        if self.bedrock_model_id != PINNED_MODEL_ID:
            problems.append(f"judge model is {self.bedrock_model_id!r}, spec pins {PINNED_MODEL_ID!r}")
        if self.bedrock_embed_model_id != PINNED_EMBED_ID:
            problems.append(f"embeddings are {self.bedrock_embed_model_id!r}, spec pins {PINNED_EMBED_ID!r}")
        if self.thinking_mode not in ("adaptive", "enabled", "off"):
            problems.append(f"BEDROCK_THINKING={self.thinking_mode!r} is not adaptive|enabled|off")
        if self.thinking_mode == "adaptive" and self.thinking_effort not in THINKING_EFFORTS:
            problems.append(f"BEDROCK_THINKING_EFFORT={self.thinking_effort!r} not in {THINKING_EFFORTS}")
        if self.pace_concurrency_max > self.boto_pool:
            problems.append("PACE_CONCURRENCY_MAX exceeds BOTO_POOL — calls will starve on sockets")
        for label, p in (("unified eval file", self.unified_eval_json),
                         ("judge prompts", self.judge_prompts_md),
                         ("freeze manifest", self.freeze_manifest_json),
                         ("golden_v2 (lineage/join)", self.golden_v2_json),
                         ("kb snapshot sha", self.kb_snapshot_sha_file)):
            if not p.exists():
                problems.append(f"missing input {label}: {p}")
        if not self.kb_dir.is_dir():
            problems.append(f"missing Extracted_KB directory: {self.kb_dir}")
        return problems

    def snapshot(self) -> dict:
        """Fully-resolved, secret-free config for `config_snapshot.json` (reproducibility)."""
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}
        d["segments"] = list(self.segments)
        d["additional_model_request_fields"] = self.additional_model_request_fields()
        d["thinking_enabled"] = self.thinking_enabled
        d["temperature"] = "unset (Sonnet 5 rejects the parameter; thinking implies temperature=1)"
        return d


def load_config() -> Config:
    return Config()


def main() -> None:
    cfg = load_config()
    print("MeitY PQA RAGAS framework — resolved configuration")
    print(f"  aws_region              : {cfg.aws_region}")
    print(f"  judge model             : {cfg.bedrock_model_id}")
    print(f"  embeddings              : {cfg.bedrock_embed_model_id}")
    print(f"  thinking                : mode={cfg.thinking_mode} effort={cfg.thinking_effort}")
    print(f"  additionalModelRequestFields: {cfg.additional_model_request_fields()}")
    print(f"  max_tokens (ceiling)    : {cfg.ragas_max_tokens:,}")
    print(f"  aws credentials         : {'present' if cfg.has_aws_credentials else 'MISSING'} (values not printed)")
    print(f"  unified eval (read-only): {cfg.unified_eval_json}")
    print(f"  KB (read-only)          : {cfg.kb_dir}")
    print(f"  runs_dir                : {cfg.runs_dir}")
    print(f"  expected samples        : {cfg.expected_sample_count}")
    print(f"  segments                : {', '.join(cfg.segments)}")
    print(f"  pacing                  : concurrency {cfg.pace_concurrency_min}"
          f"/{cfg.pace_concurrency_start}/{cfg.pace_concurrency_max} (min/start/max), "
          f"{cfg.pace_rpm:g} rpm, {cfg.pace_tpm:,.0f} tpm")
    print(f"  timeouts                : cell {cfg.ragas_timeout}s (+{cfg.per_chunk_timeout}s/chunk "
          f"for context_precision), stall {cfg.stall_timeout}s")
    print(f"  retries                 : {cfg.ragas_retry_rounds} RAGAS NA rounds, "
          f"{cfg.judge_retry_rounds} judge rounds, {cfg.pace_max_attempts} transport attempts")
    print(f"  rates ($/1k tok)        : in={cfg.rate_llm_in_per_1k} out={cfg.rate_llm_out_per_1k} "
          f"embed={cfg.rate_embed_per_1k}   (tracked, never gated)")
    issues = cfg.issues()
    if issues:
        print("\n  config notes:")
        for i in issues:
            print(f"    - {i}")
    else:
        print("\n  config OK — inputs present, model pinned, thinking on.")


if __name__ == "__main__":
    main()
