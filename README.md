# PQA_ragas — MeitY Parliamentary Q&A chatbot capture + RAGAS evaluation

This project captures and scores the **MeitY (Ministry of Electronics and Information
Technology) Parliament AI Platform** chatbot at `https://pqa.myscheme.in/meityy`, against a
66-row golden dataset of real parliamentary questions (see `golden_dataset_README.md`).

Built from an earlier Commerce-ministry evaluation project's framework as reference (shared
`myscheme.in` platform pattern, same RAGAS + LLM-judge pipeline shape) — but this project's
dataset, chatbot contract, and scoring config are independent of that one. Historical notes
that still cite the Commerce project (`docs/KNOWN_ISSUES.md`, `docs/README.md`) are kept
because they document real upstream-`ragas` bugs or forensics that remain useful; they're
explicitly labeled as carried-over context, not as this project's own findings.

It has three stages:

1. **Auth capture** (`auth-capture/`) — logs into the portal via email-OTP and saves a
   Playwright browser session. Status: not yet run through `npm run auth:capture` end to
   end — the live session used so far (`auth-capture/.auth/default.json`) was captured
   manually via browser automation during setup. Before relying on `npm run auth:capture`
   for a future refresh, verify `pages/LoginPage.ts`'s selectors against the real login page
   (`npx playwright codegen https://pqa.myscheme.in/meityy`).
2. **Chatbot capture** (`capture_new_chatbot.py` + `capture/`) — calls the confirmed
   `POST /meityy/api/assistant/chat/stream` endpoint for every golden dataset row (bare
   question, no wrapper prompt; see `chatbot_endpoint_config.json`'s `known_from_live_probe`
   for how the contract was confirmed) and records the response, including real session
   continuity for the dataset's follow-up (`-F1`/`-F2`) pairs. Status: done — all 66 rows
   captured into `data/artifacts/golden_dataset_new_chatbot_captured.json`.
3. **Scoring** (`cli.py` + `ragas_eval/`) — RAGAS 6 metrics + 3 LLM judges on AWS Bedrock.
   Status: `--validate` and `--estimate-cost` pass clean (~$19.60 estimated); the paid
   `--all` run has not been launched yet.

## Quickstart

Not in this repo, and needed locally:

- `.env` — copy `.env.example` and fill in your own AWS Bedrock credentials and settings.
- `documents/` — the 2,250 Lok Sabha / Rajya Sabha reply PDFs (395 MB), shared separately.
  Without them scoring still runs, but the knowledge-base citation check is skipped.
- Captured chatbot answers (`data/artifacts/golden_dataset_*`) and run outputs (`runs/`) —
  produce your own with the capture step below, using your own portal login.

```bash
# one-time setup (Python 3.10.x required — see requirements.lock.txt)
py -3.10 -m venv .venv
.venv/Scripts/pip install -r requirements.lock.txt
cp .env.example .env      # fill in AWS_*, PQA_CHATBOT_*, GOLDEN_EXPECTED_*

.venv/Scripts/python.exe cli.py --validate          # free: schema + routing, no API
.venv/Scripts/python.exe cli.py --estimate-cost     # free: token/$ estimate
.venv/Scripts/python.exe cli.py --connectivity      # 1 small paid call: creds/model check
.venv/Scripts/python.exe cli.py --all               # PAID full run — only once approved
```

To re-capture chatbot responses (needs a fresh `auth-capture/.auth/default.json` — the
`pqa_session` cookie's real lifetime is short, ~30 minutes after login):

```bash
.venv/Scripts/python.exe capture_new_chatbot.py --dry-run     # sanity check, no network
.venv/Scripts/python.exe capture_new_chatbot.py --ids <one-row-id>   # one real call first
.venv/Scripts/python.exe capture_new_chatbot.py                # full capture, skips done rows
.venv/Scripts/python.exe tools/build_unified_eval.py            # promote capture -> scoring input
```

## Golden dataset schema

`capture/dataset_io.py` and `ragas_eval/dataset.py` expect each row to carry at least:
`id`, `user_input`, `reference`, `reference_contexts`, `must_include_facts`,
`expected_sources`, `expected_behavior`, `ragas_eligible`, `stratum`, `category`,
`difficulty`, `house` — plus the capture-populated `response`, `retrieved_contexts`,
`cited_sources`, `cited_sources_raw`. See `ragas_eval/dataset.py`'s `_REQUIRED_ROW` tuple
and `capture/dataset_io.py`'s `PROTECTED_FIELDS` if the dataset's shape ever changes — keep
both in sync with whatever the rows actually carry. `documents/` (Lok Sabha / Rajya Sabha)
holds the source parliamentary documents the dataset was built from, kept for reference.

## A few hard-won lessons (from building this and the project it's built from)

- **Request shape matters more than it looks.** A request missing fields, or with wrong
  framing, can silently produce a measurably worse answer than the real app while still
  returning HTTP 200 with the same retrieved sources. Always validate a new or changed
  endpoint contract against a real captured browser request, not just "it returns 200 and
  something that looks like an answer."
- **Multi-turn continuity needs the prior turn replayed into `messages`**, not just a
  reused `session_id`/`user_id` — confirmed via a live two-call test on this chatbot (same
  `session_id`, context only carried when the prior turn was replayed).
- **LLM sampling variance is real even with a correct, identical request.** Expect small
  wording/length differences between otherwise-identical repeat calls; don't chase
  eliminating this; do expect it when comparing two captures of the same question.
- **A thin or truncated answer isn't automatically a pipeline bug.** It can be a genuine
  retrieval gap (source never found) or a generation-length limit on the backend. Isolate
  the cause (same sources? same request bytes? reproducible across repeats?) before
  assuming the capture script is at fault.
