"""Bedrock provider factories: Claude Sonnet 5 (extended thinking) + Titan V2 embeddings.

Three things happen here that do not happen in the reference project, and each exists
because of a specific, observed failure mode:

* **Extended thinking is sent explicitly** on every scoring call via
  `additionalModelRequestFields`. Probed against `global.anthropic.claude-sonnet-5` in
  `ap-south-1`: `thinking.type` must be `"adaptive"` (`"enabled"` + `budget_tokens` is
  rejected outright), and the effort knob is the *sibling* key `output_config.effort`,
  not a field inside `thinking`. Accepted efforts, weakest → strongest:
  `low, medium, high, xhigh, max`. We send `max`.

* **Only `text` blocks reach ragas.** With thinking on, Bedrock returns a
  `reasoningContent` block alongside the text block, so `response.content` is a *list*.
  Anything that stringifies it produces a Python repr (`[{'type': ...}]`) whose quoting
  destroys the embedded JSON — that is the reference project's ISSUE-6, and thinking mode
  makes it a certainty rather than an edge case. `_generate` rebuilds the `ChatResult`
  with the concatenated text blocks as a plain string, so no downstream consumer can hit
  the repr at all. Reasoning presence/size is preserved in `response_metadata` for the
  smoke gate's "is thinking actually on?" check.

* **Truncation is an error, not a score.** ragas's own `is_finished` treats
  `MAX_TOKENS` as a normal finish, so a judge output cut off mid-JSON can silently become
  a score. On Bedrock `max_tokens` caps the TOTAL output *including* reasoning tokens
  (KNOWN_ISSUES ISSUE-9c), which thinking makes far more likely. Any `stopReason ==
  "max_tokens"` raises `TruncatedOutputError`, which becomes an honest NA and is retried.

Every call is routed through the `AdaptiveLimiter` (see `pacing.py`) so throttles are
avoided rather than absorbed by botocore back-off.
"""
from __future__ import annotations

import threading

from .pacing import AdaptiveLimiter


class TruncatedOutputError(RuntimeError):
    """The judge's response hit max_tokens — never score a truncated output."""


# ---------------------------------------------------------------------------
# token-usage accumulator (cost is tracked and reported, never gated)
# ---------------------------------------------------------------------------
_USAGE_LOCK = threading.Lock()
_USAGE = {"input_tokens": 0, "output_tokens": 0, "calls": 0,
          "reasoning_blocks": 0, "truncations": 0}


def reset_usage() -> None:
    with _USAGE_LOCK:
        for k in _USAGE:
            _USAGE[k] = 0


def read_usage() -> dict:
    with _USAGE_LOCK:
        return dict(_USAGE)


def _record_usage(in_tok: int, out_tok: int, reasoning: bool) -> None:
    from .watchdog import heartbeat
    heartbeat("judge call returned", call=True)
    with _USAGE_LOCK:
        _USAGE["input_tokens"] += int(in_tok or 0)
        _USAGE["output_tokens"] += int(out_tok or 0)
        _USAGE["calls"] += 1
        if reasoning:
            _USAGE["reasoning_blocks"] += 1


def _record_truncation() -> None:
    with _USAGE_LOCK:
        _USAGE["truncations"] += 1


# ---------------------------------------------------------------------------
# raw-response sampling (smoke gate proof that thinking is actually active)
# ---------------------------------------------------------------------------
_SAMPLE_LOCK = threading.Lock()
_SAMPLES: list = []
_SAMPLE_LIMIT = 0


def enable_raw_sampling(limit: int = 5) -> None:
    global _SAMPLE_LIMIT
    with _SAMPLE_LOCK:
        _SAMPLE_LIMIT = int(limit)
        _SAMPLES.clear()


def read_raw_samples() -> list:
    with _SAMPLE_LOCK:
        return list(_SAMPLES)


def _maybe_sample(block_types, reasoning_chars, signature_len, stop_reason, usage) -> None:
    with _SAMPLE_LOCK:
        if len(_SAMPLES) >= _SAMPLE_LIMIT:
            return
        _SAMPLES.append({
            "block_types": list(block_types),
            "has_reasoning_block": "reasoning_content" in block_types,
            "reasoning_text_chars": reasoning_chars,
            "reasoning_signature_chars": signature_len,
            "stop_reason": stop_reason,
            "usage": dict(usage or {}),
        })


# ---------------------------------------------------------------------------
_LLM_CLS = None
_LIMITER: AdaptiveLimiter | None = None


def set_limiter(limiter: AdaptiveLimiter | None) -> None:
    """Install the process-wide pacing limiter used by every judge call."""
    global _LIMITER
    _LIMITER = limiter


def get_limiter() -> AdaptiveLimiter | None:
    return _LIMITER


def _extract_text_and_reasoning(content):
    """Return (text, reasoning_chars, signature_chars, block_types).

    Concatenates ONLY `text` blocks. A string `content` passes through unchanged.
    """
    if isinstance(content, str):
        return content, 0, 0, ["text"]
    text_parts, rchars, sigchars, kinds = [], 0, 0, []
    for block in content or []:
        if not isinstance(block, dict):
            text_parts.append(str(block))
            kinds.append("raw")
            continue
        btype = block.get("type") or next(iter(block), "")
        kinds.append(btype)
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype in ("reasoning_content", "reasoningContent", "thinking"):
            payload = block.get(btype) or block.get("reasoning_content") or {}
            if isinstance(payload, dict):
                rchars += len(payload.get("text") or "")
                sigchars += len(payload.get("signature") or "")
            elif isinstance(payload, str):
                rchars += len(payload)
    return "".join(text_parts), rchars, sigchars, kinds


def _estimate_tokens(messages) -> float:
    """Cheap char/4 estimate for the token bucket (pacing only, never billing)."""
    total = 0
    for m in messages or []:
        c = getattr(m, "content", "")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):
            total += sum(len(b.get("text", "")) for b in c if isinstance(b, dict))
    return total / 4.0


def _paced_converse_cls():
    """A ChatBedrockConverse that paces, strips reasoning blocks, and refuses truncation."""
    global _LLM_CLS
    if _LLM_CLS is None:
        from langchain_aws import ChatBedrockConverse
        from langchain_core.outputs import ChatGeneration, ChatResult

        class _PacedBedrockConverse(ChatBedrockConverse):
            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                # ragas' LangchainLLMWrapper mutates `temperature` at call time; Sonnet 5
                # rejects the parameter entirely (and thinking implies temperature=1 anyway).
                if getattr(self, "temperature", None) is not None:
                    self.temperature = None

                def _call():
                    return super(_PacedBedrockConverse, self)._generate(
                        messages, stop=stop, run_manager=run_manager, **kwargs)

                limiter = _LIMITER
                result = limiter.run(_call, est_tokens=_estimate_tokens(messages)) \
                    if limiter is not None else _call()

                out_gens = []
                for gen in result.generations:
                    msg = gen.message
                    text, rchars, sigchars, kinds = _extract_text_and_reasoning(msg.content)
                    meta = dict(getattr(msg, "response_metadata", {}) or {})
                    stop_reason = meta.get("stopReason") or meta.get("stop_reason")
                    usage = getattr(msg, "usage_metadata", None) or {}
                    _record_usage(usage.get("input_tokens", 0), usage.get("output_tokens", 0),
                                  reasoning="reasoning_content" in kinds)
                    if _SAMPLE_LIMIT:
                        _maybe_sample(kinds, rchars, sigchars, stop_reason, usage)

                    if stop_reason in ("max_tokens", "MAX_TOKENS"):
                        _record_truncation()
                        raise TruncatedOutputError(
                            f"judge output hit max_tokens ({self.max_tokens}); "
                            f"{usage.get('output_tokens', '?')} output tokens, "
                            f"{len(text)} chars of text — refusing to score a truncated output")

                    # Collapse to a plain-string message: reasoning can never leak into a
                    # parse path, and `.text` is trivially correct (ISSUE-6 made impossible).
                    meta["reasoning_text_chars"] = rchars
                    meta["reasoning_signature_chars"] = sigchars
                    meta["content_block_types"] = kinds
                    msg_copy = msg.model_copy(update={"content": text, "response_metadata": meta})
                    out_gens.append(ChatGeneration(message=msg_copy,
                                                   generation_info=gen.generation_info))
                return ChatResult(generations=out_gens, llm_output=result.llm_output)

        _LLM_CLS = _PacedBedrockConverse
    return _LLM_CLS


def get_llm(cfg, model_id: str | None = None, max_tokens: int | None = None):
    """The pinned judge: Claude Sonnet 5 on Bedrock with extended thinking at max effort."""
    from botocore.config import Config as BotoConfig
    bcfg = BotoConfig(
        read_timeout=cfg.bedrock_read_timeout,
        connect_timeout=cfg.bedrock_connect_timeout,
        max_pool_connections=max(cfg.boto_pool, cfg.pace_concurrency_max * 4),
        # base layer only — our own AIMD/backoff sits above it
        retries={"max_attempts": 3, "mode": "adaptive"},
    )
    kwargs = dict(
        model=model_id or cfg.bedrock_model_id,
        region_name=cfg.aws_region,
        max_tokens=max_tokens or cfg.ragas_max_tokens,
        config=bcfg,
    )
    amrf = cfg.additional_model_request_fields()
    if amrf:
        kwargs["additional_model_request_fields"] = amrf
    return _paced_converse_cls()(**kwargs)


def get_embeddings(cfg):
    """Amazon Titan Text Embeddings V2 on Bedrock."""
    from botocore.config import Config as BotoConfig
    from langchain_aws import BedrockEmbeddings
    bcfg = BotoConfig(
        read_timeout=cfg.bedrock_read_timeout, connect_timeout=cfg.bedrock_connect_timeout,
        max_pool_connections=max(cfg.boto_pool, cfg.pace_concurrency_max * 4),
        retries={"max_attempts": 8, "mode": "adaptive"},
    )
    return BedrockEmbeddings(model_id=cfg.bedrock_embed_model_id,
                             region_name=cfg.aws_region, config=bcfg)


# ---------------------------------------------------------------------------
_THINKING_PROBE_PROMPT = (
    "Decompose this answer into atomic claims and classify each against the context.\n"
    "Context: 'As per DGCI&S data, India's merchandise exports were USD 437.10 billion in "
    "2023-24 and USD 441.79 billion in 2024-25; the Interest Equalisation Scheme ended on "
    "31.12.2024.'\n"
    "Answer: 'Merchandise exports rose from USD 441.79 billion in 2023-24 to USD 437.10 "
    "billion in 2024-25, and the Interest Equalisation Scheme was extended through 2025.'\n"
    'Return strict JSON {"TP": [], "FP": [], "FN": []}.')


def connectivity_smoke(cfg) -> dict:
    """One LLM call + one embed. Proves creds, model id, and that thinking is really on.

    The prompt has to be one the model would *choose* to think about: `thinking.type` is
    `adaptive`, so effort is a ceiling, not a floor — asking it to echo a word produces no
    reasoning block and would make a working configuration look broken. This mirrors the
    shape of a real metric call (claim decomposition + classification).
    """
    enable_raw_sampling(3)
    reset_usage()
    llm = get_llm(cfg, max_tokens=8192)
    resp = llm.invoke(_THINKING_PROBE_PROMPT)
    emb = get_embeddings(cfg)
    vec = emb.embed_query("connectivity probe")
    samples = read_raw_samples()
    usage = read_usage()
    return {
        "model_id": cfg.bedrock_model_id,
        "region": cfg.aws_region,
        "text": (resp.content if isinstance(resp.content, str) else str(resp.content))[:120],
        "embed_dim": len(vec),
        "additional_model_request_fields": cfg.additional_model_request_fields(),
        "raw_samples": samples,
        "thinking_active": any(s["has_reasoning_block"] for s in samples),
        "usage": usage,
    }
