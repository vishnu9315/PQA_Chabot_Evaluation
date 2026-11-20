import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load .env for the WHOLE suite, the same way capture_new_chatbot.py does on import -
# without this, which test file happens to import capture_new_chatbot first (triggering
# its own _load_dotenv as an import side effect) silently changes what env vars (e.g.
# GOLDEN_EXPECTED_TOTAL_ROWS) the rest of the suite sees. Never overrides a variable
# already set in the real environment (os.environ.setdefault semantics).
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(ROOT / ".env")
except ModuleNotFoundError:
    _env_path = ROOT / ".env"
    if _env_path.exists():
        import os
        for _raw in _env_path.read_text(encoding="utf-8").splitlines():
            _line = _raw.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _key, _, _val = _line.partition("=")
            _val = _val.split(" #", 1)[0].strip().strip('"').strip("'")
            os.environ.setdefault(_key.strip(), _val)
