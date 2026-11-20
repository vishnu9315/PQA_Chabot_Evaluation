"""Build a session `Cookie:` header from a Playwright storageState JSON file.

This is the ONLY supported way this project obtains a session, assuming - like the
previous project this was copied from - there's no standalone login API to call directly
and sign-in is only ever exercised through the browser (see ../auth-capture/pages/LoginPage.ts
and ../auth-capture/scripts/capture-auth.ts). A real session is captured there, by a human
relaying a real emailed OTP, via `npm run auth:capture` (from auth-capture/), which writes
a Playwright storageState file to auth-capture/.auth/*.json. This module reads that SAME
file format (it is not a new or guessed format - it is the file Playwright itself writes
and this project's own auth scripts already produce) and turns its cookies into a
`Cookie:` header for plain, non-browser HTTP requests.

Cookie VALUES are secrets. Nothing in this module ever logs, prints, or includes a
cookie value in an exception message - only cookie NAMES (e.g. "pqa_session", which is
not itself secret and is already documented elsewhere in this repo) and file paths.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse


class SessionFileError(RuntimeError):
    """The session file is missing, unreadable, malformed, or has no cookies matching
    the target host. Messages here never include a cookie value."""


def domain_matches(cookie_domain: str, host: str) -> bool:
    """Standard cookie domain-matching, case-insensitive:
    - an exact match ('pqa.myscheme.in' cookie domain, 'pqa.myscheme.in' host), or
    - a leading-dot domain that the host matches or is a subdomain of
      ('.myscheme.in' matches both 'myscheme.in' and 'pqa.myscheme.in'), or
    - a bare (no leading dot) domain that the host is a strict subdomain of, matching
      how Chromium (and therefore Playwright's storageState) actually records cookies
      set with a Domain attribute.
    """
    cookie_domain = cookie_domain.strip().lower()
    host = host.strip().lower()
    if cookie_domain == host:
        return True
    if cookie_domain.startswith("."):
        bare = cookie_domain[1:]
        return host == bare or host.endswith("." + bare)
    return host.endswith("." + cookie_domain)


def load_cookie_header(session_file: Path, base_url: str) -> str:
    """Read a Playwright storageState JSON and return a `Cookie:` header value built
    from every cookie whose domain matches `base_url`'s host.

    Raises SessionFileError - never a bare/uncaught exception, and never with a cookie
    value in its message - if the file is missing, not valid JSON, not a storageState
    file (no 'cookies' list), or has no cookies for the target host.
    """
    if not session_file.exists():
        raise SessionFileError(f"session file not found: {session_file}")

    try:
        data = json.loads(session_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SessionFileError(
            f"session file is not valid JSON: {session_file} ({exc})") from exc
    except OSError as exc:
        raise SessionFileError(f"could not read session file: {session_file} ({exc})") from exc

    cookies = data.get("cookies") if isinstance(data, dict) else None
    if not isinstance(cookies, list):
        raise SessionFileError(
            f"{session_file} has no top-level 'cookies' list - is this a Playwright "
            f"storageState file (e.g. auth-capture/.auth/default.json)?")

    host = urlparse(base_url).hostname or ""
    if not host:
        raise SessionFileError(
            f"could not determine a hostname from PQA_CHATBOT_BASE_URL={base_url!r}")

    matching = [
        c for c in cookies
        if isinstance(c, dict)
        and isinstance(c.get("name"), str) and c["name"]
        and isinstance(c.get("value"), str)
        and isinstance(c.get("domain"), str)
        and domain_matches(c["domain"], host)
    ]
    if not matching:
        raise SessionFileError(
            f"no cookies in {session_file} matched host {host!r} - either this is the "
            f"wrong session file, PQA_CHATBOT_BASE_URL points at the wrong host, or the "
            f"captured session genuinely has no cookies for this host")

    return "; ".join(f"{c['name']}={c['value']}" for c in matching)
