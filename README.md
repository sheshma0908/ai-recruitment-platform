# AI Recruitment Platform -- ML Service

FastAPI service wrapping all 7 AI/ML modules from the project report. Built and tested
end-to-end against the models trained in the accompanying notebooks.

## Performance / latency

All models (spaCy NER, XGBoost matching/attrition/salary, TF-IDF vectorizer) load **once, at
server startup** -- not on each request. Startup typically takes a couple of seconds (the spaCy
model is the slowest piece, ~2-12s depending on disk cache state). After that, individual
requests are fast: single-digit to low-double-digit milliseconds for most endpoints.

**If requests seem consistently slow, check these in order:**

1. **Is the server actually staying up between requests?** If something is restarting the
   `uvicorn` process between calls (a process manager set to restart on every request, a
   serverless-style deployment that spins up fresh each time, `--reload` triggering restarts from
   file changes), every request pays the full startup cost again. Run with a plain
   `uvicorn app.main:app --port 8100` (no `--reload`) for anything other than active local
   development.
2. **Is `explain=true` being sent on every `/match-score` call?** SHAP's explanation computation
   is meaningfully slower than a plain score -- only request it for the specific view that
   actually displays the explanation, not on every call.
3. **Is `resume_text` being sent repeatedly for the same candidate** on every `/match-score` /
   `/skill-gap` / `/rank-candidates` call, instead of parsing once and reusing the structured
   result? Each `resume_text` call re-runs the full NER pipeline. If a candidate's profile is
   already parsed and stored in the database, pass the structured `resume` object instead --
   it skips NER entirely and is faster.
4. **Concurrent load:** a single `uvicorn` process handles requests one worker's worth of CPU at
   a time. For concurrent traffic beyond a light demo, run with multiple workers:
   `uvicorn app.main:app --port 8100 --workers 4` (each worker loads its own copy of the models,
   so this increases memory use accordingly).
5. **Network vs. compute:** if the ML service and backend are on different machines, some of the
   perceived latency may be network round-trip, not model computation -- time a direct `curl` to
   the ML service (see `GET /health`) versus the full request path through the backend to isolate
   which side the delay is actually on.

## Setup

```bash
pip install -r requirements.txt
```

The trained models are already bundled inside `app/models/` -- no retraining needed to run
this service. If you retrain any model, drop the new `.pkl` file / `resume_ner_model/` folder
in `app/models/` with the same filename and restart the service.

## Running

```bash
uvicorn app.main:app --reload --port 8100
```

Then open **http://localhost:8100/docs** for interactive API documentation (Swagger UI) --
you can try every endpoint directly from the browser there.

## Testing

```bash
python3 test_api.py
```

This runs realistic sample requests against every endpoint and prints PASS/FAIL for each.
All 9 checks (8 endpoints + 1 error-handling case) pass as of this build.

## Backward compatibility

Every endpoint that previously required a structured `resume` object still works with
**exactly the same request body** as before -- nothing your team has already integrated
will need to change. What's new is *optional*:

- `/match-score`, `/skill-gap` now also accept `resume_text` instead of `resume` (raw text
  gets parsed via the Module 5 NER model internally). Provide exactly one of the two.
- `/rank-candidates` now also accepts an optional `candidate_texts` list of raw resume texts,
  which can be combined with the existing `candidates` list -- both pools get merged before
  ranking.

Example -- old style (still works, unchanged):
```json
{"job": {...}, "resume": {"resume_id": "R1", "role": "...", "skills": "['Python']", ...}}
```

Example -- new style (raw text instead):
```json
{
  "job": {...},
  "resume_text": "Priya Sharma\nSenior Backend Engineer\n...",
  "resume_seniority": "Senior",
  "resume_industry": "IT"
}
```

`resume_seniority` / `resume_industry` are optional hints -- the NER model's label set
doesn't include these fields, so they can't be reliably extracted from raw text. Pass them
if you know them; otherwise they're treated as unknown rather than guessed.

## File upload -- same URLs, no separate endpoints

`/parse-resume`, `/screen-resume`, `/match-score`, `/skill-gap`, and `/generate-interview-questions`
each accept **either** a JSON body (exactly as documented below -- this is what the backend's
existing integration sends, completely unchanged) **or** an uploaded resume file
(`multipart/form-data`), at the **same URL**. There are no separate `-file` endpoints.

**In Swagger UI (`/docs`):** open any of these endpoints and click "Try it out" -- a dropdown
appears letting you choose the request format:
- **`application/json`** -- the usual JSON body, a text box to paste it into
- **`multipart/form-data`** -- an actual file picker ("Choose File") to upload a `.pdf` or `.docx`
  directly, plus a `job` field where you paste the job details as a JSON string (since a file
  upload and a nested JSON object can't travel in the same multipart field)

Example multipart fields for `/screen-resume`:
```
file: <choose a .pdf or .docx from your computer>
job: {"job_id": "J_1", "job_title": "Backend Developer", "seniority": "Mid", "industry": "IT", "must_have_skills": "python, django", "nice_to_have_skills": "docker", "description": "", "responsibilities": "", "requirements": ""}
```

**Supported file types:** `.pdf` and `.docx`. Legacy `.doc` (old binary Word format) returns a
clear error asking for `.docx` or `.pdf` instead. A scanned/image-only PDF with no selectable text
layer also returns a clear error rather than silently producing empty results.

**Why this doesn't affect the backend integration at all:** the JSON path is handled by the exact
same validation and logic as before -- nothing about it changed. The endpoint inspects the
incoming request's `Content-Type` header and only takes the file-upload path when it sees
`multipart/form-data`; any `application/json` request is processed exactly as it always was.

## Endpoints

| Endpoint | Module | Description |
|---|---|---|
| `GET /health` | -- | Service health check |
| `POST /parse-resume` | 5 | Raw resume text -> structured profile (Name, Skills, Companies, etc.) |
| `POST /screen-resume` | 6 | Job + resume -> dimension-by-dimension screening report (Skills, Seniority, Industry, Role Alignment, Content Relevance) -- accepts either a structured resume or raw resume text |
| `POST /match-score` | 7 | Job + resume -> single weighted match probability, with optional SHAP explanation |
| `POST /skill-gap` | 17 | Job + resume -> missing must-have/nice-to-have skills |
| `POST /rank-candidates` | 18 | Job + list of candidates -> ranked shortlist |
| `POST /generate-interview-questions` | 19 | Job + resume -> tailored interview questions |
| `POST /predict-attrition` | 34 | Employee record -> attrition risk (LOW/MEDIUM/HIGH) |
| `POST /predict-salary` | 35 | Job posting -> estimated salary range |
| `POST /recommend-candidate-decision` | 23 (support) | Match score + interview feedback -> suggested recommendation (Recommended / Hold / Not Recommended) |

Full request/response schemas (with example values) are visible at `/docs`.

### About `/recommend-candidate-decision`

This supports the Hiring Manager's "Approve/Reject Candidates" permission, but it does **not**
make or store that decision -- it returns a *suggested* recommendation for the Hiring Manager to
review, by combining:
- The AI match score (40% weight) -- either pass `match_probability_pct` directly if you already
  have it from a prior `/match-score` call, or pass `job` + `resume` to compute it fresh
- The interviewer's structured feedback (60% weight) -- Technical Skills, Communication, Problem
  Solving, Teamwork, Leadership, each 0-10, matching the report's Module 23 example

Interview feedback is weighted higher since it's a direct human judgment made after actually
meeting the candidate. The **actual** approve/reject click, and its permanent storage against the
candidate's record, is the backend team's responsibility -- this endpoint is stateless like every
other one in this service, and only ever returns a suggestion.

### Module 6 vs Module 7 -- what's the difference?

`/screen-resume` (Module 6) and `/match-score` (Module 7) look at the same underlying signals
but serve different purposes:
- **`/screen-resume`** returns a *report*: each dimension (skills, seniority, industry, role,
  content relevance) shown separately, for a human recruiter to read and understand *why*.
- **`/match-score`** returns a *single number*: all of that combined by a trained XGBoost model
  into one weighted match probability, for ranking/sorting candidates at scale.

Run one after the other in the demo to show the full pipeline: parse -> screen -> score.

## Environment variables (optional)

`/generate-interview-questions` uses a real LLM if a key is available, and falls back to a
template-based generator otherwise (still returns a valid response either way):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export OPENAI_API_KEY="sk-..."
```

Then pass `"provider": "anthropic"` or `"provider": "openai"` in the request body.

## Field name conventions

Request field names match the original training CSVs **exactly**, so the backend team can
pass database records straight through without renaming fields:

- `/match-score`, `/skill-gap`, `/rank-candidates`, `/generate-interview-questions` -- fields
  match `jobs.csv` and `resume.csv` (e.g. skills are Python-list-repr strings like
  `"['Python' 'Django']"`, matching the original dataset format).
- `/predict-attrition` -- fields match the IBM HR Attrition dataset columns.
- `/predict-salary` -- fields match the India job market dataset columns.

## Project structure

```
app/
  main.py            - FastAPI app and route definitions
  schemas.py         - Pydantic request/response models
  ml_utils.py        - shared text-processing helpers (skill parsing, normalization)
  matching.py        - candidate-job matching (Modules 6/7)
  skill_gap.py       - skill gap analysis + ranking (Modules 17/18)
  attrition.py       - attrition prediction (Module 34)
  salary.py          - salary prediction (Module 35)
  interview.py       - interview question generation (Module 19)
  resume_parsing.py  - resume NER (Module 5)
  models/            - trained model files:
    match_model.ubj          - matching XGBoost model, native format
    match_engine_meta.pkl    - vectorizer + feature list + threshold (no XGBoost objects)
    salary_point_model.ubj   - salary XGBoost model (point estimate), native format
    salary_lower_model.ubj   - salary XGBoost model (25th percentile), native format
    salary_upper_model.ubj   - salary XGBoost model (75th percentile), native format
    salary_meta.pkl          - feature list + category columns (no XGBoost objects)
    attrition_bundle.pkl     - scikit-learn GradientBoostingClassifier (plain pickle is fine here)
    resume_ner_model/        - spaCy NER model directory
test_api.py          - end-to-end endpoint tests
requirements.txt
```

**Why the XGBoost models are `.ubj` files instead of pickled inside a bundle:** a pickled
XGBoost booster can fail to load on a different machine even when both machines report the
same xgboost version -- we hit this directly during testing (an `XGBoostError: input stream
corrupted` on Windows/Python 3.14, despite an identical, checksum-verified `.pkl` file and
matching xgboost version). XGBoost's own `save_model()`/`load_model()` format is explicitly
designed to be portable across platforms, OS, and versions, so the two XGBoost-backed modules
(matching, salary) use that instead. The attrition model uses scikit-learn's
`GradientBoostingClassifier`, which isn't affected by this particular issue, so it stays as a
plain pickle.

## Notes for the backend integration

- All models are loaded once at import time (not per-request), so the first request after
  startup is fast -- there's no lazy-loading delay per call.
- `/match-score` accepts an optional `explain=true` query parameter to include a SHAP-based
  breakdown of why a score was produced (useful for the "explainable match score" requirement
  in the project report). It's off by default since it's noticeably slower than a plain score.
- `/rank-candidates` expects the full candidate pool in the request body (no database lookup
  built in) -- the backend is expected to fetch candidates and pass them in.

## Candidate Chatbot (`POST /chat`)

Own trained model -- no Groq / OpenAI. For **candidates only**.

- **Dataset:** Career Guidance QA Dataset by Pradeep (2025), CC BY 4.0 --
  https://huggingface.co/datasets/Pradeep016/career-guidance-qa-dataset
  (1,620 Q&A rows, 54 job roles), stored in `training/chatbot/dataset/career_guidance_qa.csv`.
- **Model:** `app/models/candidate_chatbot.joblib` -- topic classifier (TF-IDF + Logistic
  Regression, 10 topics) + job-role detector (54 roles). Code: `app/career_model.py`.
- **Service code:** `app/chatbot.py` (answer logic), `app/chatbot_smalltalk.py` (greetings,
  ok/okay, thanks, bye, help, typos like "hlo"/"thnks"), `app/chatbot_guardrails.py` (candidate-only rules).
- **Accuracy:** 100% topic accuracy on `training/chatbot/tests/holdout_questions.csv` (20/20) and
  `user_test_questions.csv` (40/40), using keyword topic rules first and the classifier as backup.
- **v3.0 features:** context-aware small talk; step-by-step roadmap for "how to become a X";
  role overview for "what is a X" / just the role name; several topics at once ("skills and salary
  of X"); role comparison ("X vs Y"); "list career roles"; career follow-ups ("and for a data
  scientist?"); suggested next questions; links to a matching current opening.
- **Conversation tests:** `python training/chatbot/tests/test_conversation.py`

Request:
```json
{
  "message": "How much does a web developer earn?",
  "role": "PUBLIC",
  "conversation_history": [{"role": "user", "content": "..."}],
  "job_context": "Java Developer | Hyderabad | 3-5 yrs\nData Scientist | Bangalore | 2-4 yrs"
}
```
Response: `{"response": "..."}`

Rules: only `role` = `PUBLIC` / `CANDIDATE` gets answers; staff roles get a "candidates only"
message. Questions mentioning recruiters, HR, hiring managers, interviewers, employees or admins
get a "can't answer" message. Job openings come only from `job_context`. Salary answers in the
dataset are in US dollars.

Retrain (from the project root, with the pinned versions in `requirements.txt`):
```bash
python training/chatbot/train_chatbot.py
```
Optional flan-t5 answer rewording: `pip install -r training/chatbot/requirements-generative.txt`
then `python training/chatbot/train_generative.py` (saves to `app/models/flan_t5_candidate/`).
