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
- Run outputs (`runs/`) — produced by `cli.py`.

Included: the 20 Nov 2026 capture of the chatbot's answers
(`data/artifacts/golden_dataset_new_chatbot_captured.json`) and the scoring input built from
it (`data/artifacts/golden_dataset_final_unified_eval.json`), so you can run `--validate` and
`--all` without capturing. Re-capture with the steps below only if you want fresh answers.

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

Scoring never contacts the portal. A portal login is needed only to capture new answers.

## Capturing fresh answers

### 1. Log in to the portal (saves a session for about 30 minutes)

```bash
cd auth-capture
npm install
npx playwright install chromium
cp .env.example .env          # nothing to fill in; the portal address is preset
npm run login                 # asks for YOUR portal email, then the emailed 6-digit code
```

You have about 3 minutes to enter the code. The session is saved to
`auth-capture/.auth/default.json` (git-ignored), which `PQA_CHATBOT_SESSION_FILE` points at.
Start the capture straight away: the session expires about 30 minutes after login.
Verified end to end against the MeitY portal on 20 Nov 2026.

### 2. Capture

The repository already holds a capture, and the script skips rows that already have an
answer. **Use `--force` to re-ask every question**, and `--out` to keep the old capture:

```bash
.venv/Scripts/python.exe capture_new_chatbot.py --dry-run          # sanity check, no network
.venv/Scripts/python.exe capture_new_chatbot.py --ids <one-row-id> --force --out data/artifacts/capture_<date>.json
.venv/Scripts/python.exe capture_new_chatbot.py --force --out data/artifacts/capture_<date>.json
.venv/Scripts/python.exe tools/build_unified_eval.py --captured data/artifacts/capture_<date>.json
.venv/Scripts/python.exe cli.py --validate
.venv/Scripts/python.exe cli.py --all
```

`build_unified_eval.py` overwrites `data/artifacts/golden_dataset_final_unified_eval.json`,
the file `cli.py` scores. Scores are cached per dataset content, so a new capture is always
scored fresh.

### Using a new dataset

1. Give each row the fields listed under [Golden dataset schema](#golden-dataset-schema).
   Optional fields the framework uses when present: `provenance.part` (`1-ingested` /
   `2-not-ingested`, reported separately), `provenance.query_style`,
   `provenance.follow_up_of` (asks a follow-up in the same chat as its first question) and
   `unsupported_aspects` on `answer_partial` rows.
2. Give each `expected_sources` entry `session` (e.g. `ls18-viii`, `rs-271`), `qno` and
   `parl_type`, so citations can be matched to the knowledge base. Add the documents for any
   new ingested session to `documents/`, with that session's `metadata.csv`.
3. Set the dataset's counts in `.env`: `GOLDEN_EXPECTED_TOTAL_ROWS`,
   `GOLDEN_EXPECTED_RAGAS_ELIGIBLE`, `GOLDEN_EXPECTED_JUDGE_ONLY_IDS` and `MOC_EXPECTED_N`.
   The scripts stop with a clear message if they don't match the file.
4. Capture and score:

```bash
.venv/Scripts/python.exe capture_new_chatbot.py --golden new_dataset.json --out data/artifacts/new_captured.json
.venv/Scripts/python.exe tools/build_unified_eval.py --captured data/artifacts/new_captured.json
.venv/Scripts/python.exe cli.py --validate
.venv/Scripts/python.exe cli.py --all
```

The Part 2 check reads the not-ingested sessions from the Part 2 rows' own
`expected_sources`, so it follows the dataset. The framework is built for the MeitY portal:
another ministry's chatbot needs its own endpoint (`chatbot_endpoint_config.json`), judge
wording (`ragas_eval/judge_runner.py`) and report titles.

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
