# Setting up PQA_ragas

This repository scores the MeitY Parliament AI chatbot (`https://pqa.myscheme.in/meityy`)
on 66 real parliamentary questions, using 6 RAGAS metrics and 3 LLM judges on AWS Bedrock.
The chatbot's answers from 20 Nov 2026 are already included, so you can score them without
logging in to the portal.

Commands below are for Windows (Git Bash or PowerShell). On macOS or Linux, use
`.venv/bin/python` wherever you see `.venv/Scripts/python.exe`.

## What you need

| For | You need |
|---|---|
| Scoring (the main use) | Git, **Python 3.10.x exactly** (newer versions break the pinned `ragas` 0.4.3), and AWS credentials with Bedrock access in `ap-south-1` to `global.anthropic.claude-sonnet-5` and `amazon.titan-embed-text-v2:0` |
| Capturing fresh answers (optional) | Node.js 18 or newer, and your own login on the MeitY portal |
| Reading the source documents (optional) | `documents.zip` from the repo owner (395 MB of PDFs) |

## 1. Get the code

```bash
git clone https://github.com/vishnu9315/PQA_Chabot_Evaluation.git
cd PQA_Chabot_Evaluation
```

## 2. Create the Python environment

```bash
py -3.10 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.lock.txt
```

Check it with `.venv/Scripts/python.exe --version`; it must say 3.10.

## 3. Add your AWS credentials

```bash
cp .env.example .env
```

Open `.env` and fill in `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`. Leave everything
else as it is; the defaults are this project's settings. Never commit `.env`.

## 4. Check the setup

```bash
.venv/Scripts/python.exe -m pytest tests/ -q          # all should pass, no API calls
.venv/Scripts/python.exe cli.py --validate            # free; expect "VALIDATE PASSED"
.venv/Scripts/python.exe cli.py --estimate-cost       # free; about $20 (actual runs cost about $29)
.venv/Scripts/python.exe cli.py --connectivity        # one small paid call
```

`--validate` should report `66 rows parsed (44 part1-ingested + 22 part2-not-ingested)` and
0 warnings. `--connectivity` should end with `thinking active on this call: True`.

## 5. Score

```bash
.venv/Scripts/python.exe cli.py --all
```

This takes about 20 minutes and costs about $29 on Bedrock. Results go to
`runs/run_vN/<timestamp>/`:

- `dashboard.html`: the shareable report. Open it in a browser.
- `report.md`: the same results as text.
- `evaluation_report.xlsx`: the same results as a spreadsheet.

If a run stops partway, `cli.py --resume` continues it without paying again for finished
scores. Read `docs/RAGAS_SCORING.md` before quoting any numbers: it explains each metric
and how to check a run is trustworthy. Part 1 and Part 2 are always reported separately.

## Optional: the source documents

Citation matching works without the PDFs: the repository includes each session's
`metadata.csv`, which lists every document. You only need the PDFs to read the source
replies. Unzip `documents.zip` so the folders sit at `documents/Lok Sabha/...` and
`documents/Rajya Sabha/...`, next to the `metadata.csv` files already there.

## Optional: capture fresh answers from the chatbot

You need your own portal login. The session lasts about 30 minutes, so capture straight
after logging in.

```bash
cd auth-capture
npm install
npx playwright install chromium
cp .env.example .env
npm run login        # asks for your portal email, then the 6-digit code from your inbox
cd ..
```

Then capture. **Use `--force`**: the included capture already answers every question, and
without `--force` the script skips them all.

```bash
.venv/Scripts/python.exe capture_new_chatbot.py --ids GD-MTY-LS17-2-U2536-V --force --out data/artifacts/capture_<date>.json
.venv/Scripts/python.exe capture_new_chatbot.py --force --out data/artifacts/capture_<date>.json
.venv/Scripts/python.exe tools/build_unified_eval.py --captured data/artifacts/capture_<date>.json
.venv/Scripts/python.exe cli.py --validate
.venv/Scripts/python.exe cli.py --all
```

The first capture command asks a single question, so you can check the login works before
asking all 66. For a different dataset, see "Using a new dataset" in `README.md`.

## If something goes wrong

| Problem | Fix |
|---|---|
| `ModuleNotFoundError ... vertexai` or install errors | Wrong Python version. Recreate `.venv` with Python 3.10. |
| `[error] no scoring input yet` | `data/artifacts/golden_dataset_final_unified_eval.json` is missing. Pull the repository again, or run `tools/build_unified_eval.py` after a capture. |
| Bedrock `AccessDeniedException` | Your AWS account doesn't have access to the model in `ap-south-1`. Request access in the Bedrock console. |
| Capture fails with 401 or 403 | The portal session expired. Run `npm run login` in `auth-capture/` again. |
| `npm run login` times out | Enter the code within 3 minutes of it being sent, then run it again. |
