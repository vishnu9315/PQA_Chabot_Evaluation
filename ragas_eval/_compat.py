"""Compatibility shim for a packaging bug in ragas 0.4.3.

ragas 0.4.3's `ragas/llms/base.py` (line 12) does:

    from langchain_community.chat_models.vertexai import ChatVertexAI

but that submodule was removed from `langchain-community` in the 0.4.x / langchain
1.x rewrite (`langchain-community==0.4.2` no longer ships `chat_models/vertexai.py`).
`ChatVertexAI` now lives in `langchain_google_vertexai`. So **stock ragas 0.4.3 fails
to import** on this stack — the DPIIT reference project only worked because its
installed copy of `base.py` had been hand-edited, which a clean `pip install` cannot
reproduce.

`apply()` registers a module alias in `sys.modules` BEFORE ragas is imported, so the
broken import resolves — without editing any file under site-packages. It is
idempotent and safe to call repeatedly. `ragas_eval/__init__.py` calls it on import,
so any `import ragas_eval...` path is safe; standalone scripts that import ragas
directly should call `ragas_eval._compat.apply()` first.

Alternatives if this shim is ever unwanted: pin a `langchain-community` that still
ships `chat_models/vertexai.py` (conflicts with `langchain==1.3.11`), or move to a
ragas version whose `llms/base.py` imports `ChatVertexAI` from
`langchain_google_vertexai` directly.


PORTED VERBATIM from ../Ragas_Skill_Ministry_PSQ_Chatbot/ragas_eval/ (Session 06 spec:
"Port the patches; don't rediscover the bugs"). Sample ids such as PQ-02 in the notes below
refer to that project's dataset; the ragas 0.4.3 bugs they describe are version bugs, not
dataset-specific ones, and PARIVESH's longer answers make them MORE likely, not less.
"""
from __future__ import annotations

import sys
import types

_ALIAS = "langchain_community.chat_models.vertexai"


def apply() -> bool:
    """Make `langchain_community.chat_models.vertexai.ChatVertexAI` resolvable.

    Returns True if a shim module was installed, False if it was unnecessary (a real
    module is present, or a shim is already registered) or impossible (the
    `langchain_google_vertexai` source is not installed).
    """
    if _ALIAS in sys.modules:
        return False
    try:
        __import__(_ALIAS)      # a real module exists (future langchain-community) — leave it
        return False
    except ModuleNotFoundError:
        pass
    try:
        from langchain_google_vertexai import ChatVertexAI
    except ModuleNotFoundError:
        return False            # deps not installed yet; ragas import will fail with a clear error
    shim = types.ModuleType(_ALIAS)
    shim.ChatVertexAI = ChatVertexAI
    sys.modules[_ALIAS] = shim
    return True
