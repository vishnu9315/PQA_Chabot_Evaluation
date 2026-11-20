"""MeitY Parliamentary-Q&A chatbot evaluation framework (RAGAS 6 + LLM judges 3).

Base layout and module names mirror Ragas_Skill_Ministry_PSQ_Chatbot (project B);
hardening — pacing, watchdog, per-metric timeout shaping, the five ragas patches, the
crash-safe cell cache, NA accounting — is ported from Ragas_Parivesh_Chatbot (project A).
Read both projects' KNOWN_ISSUES.md for the forensics behind every patch applied below.
"""

# ISSUE-P15: disable ragas usage telemetry BEFORE ragas is imported.
#
# ragas calls `_analytics.track()` after EVERY `aembed_documents()`, and track() does a
# synchronous `requests.post()` to https://t.explodinggradients.com from inside the async
# event loop. Its `timeout=1` bounds the connect and read but NOT DNS resolution, so on a
# host with no egress every call blocks in getaddrinfo() — measured at 4.84s per call here,
# against 0.00s with tracking off. Because it blocks the loop rather than a worker, it
# serialises the whole phase: the 2026-08-20 run_v4 attempt degraded from 24s to 1,197s per
# cell and completed only 42 of 264 RAGAS cells in 25 minutes before being interrupted.
#
# `do_not_track()` is lru_cache'd and read at first use, so this must be set before the
# import below. setdefault, so an explicit RAGAS_DO_NOT_TRACK=false still wins.
import os as _os  # noqa: E402
_os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

# Apply the ragas 0.4.3 vertexai import shim before anything imports ragas
# (reference project B: packaging bug — ragas 0.4.3 hard-imports a deleted submodule).
from . import _compat as _compat  # noqa: E402
_compat.apply()

# ISSUE-8 (B): fix_output_format's StringIO wrapper is dropped by Claude — rewrap valid,
# non-degenerate JSON instead of burning the repair round. Needs the shim above first.
from . import _ragas_parser_patch as _ragas_parser_patch  # noqa: E402
_ragas_parser_patch.apply()

# ISSUE-9 (B): harden the answer_correctness classifier prompt so the model emits a complete
# TP/FP/FN classification instead of omitting empty categories — fixes the cause without
# fabricating scores (a still-incomplete classification stays an honest NA).
from . import _ragas_prompt_patch as _ragas_prompt_patch  # noqa: E402
_ragas_prompt_patch.apply()

# ISSUE-9b (B): two-call index-based fallback that computes a REAL FN when the single-call
# classifier still won't return one. Fires only on failure; degenerate results stay NA.
from . import _ragas_answer_correctness_patch as _ragas_answer_correctness_patch  # noqa: E402
_ragas_answer_correctness_patch.apply()

# ISSUE-P11 (A): context-judge calibration knobs. BOTH DEFAULT STOCK for this corpus —
# the hardened precision arm was gate-validated on the PARIVESH corpus only; adopt here
# only after a gated pilot on MoC data (see KNOWN_ISSUES.md).
from . import _ragas_ctx_precision_patch as _ragas_ctx_precision_patch  # noqa: E402
_ragas_ctx_precision_patch.apply()
from . import _ragas_ctx_recall_patch as _ragas_ctx_recall_patch  # noqa: E402
_ragas_ctx_recall_patch.apply()

__version__ = "0.1.0"
