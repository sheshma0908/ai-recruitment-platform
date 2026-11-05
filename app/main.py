"""
AI Recruitment & Employee Management Platform -- ML Service

Wraps all 7 AI/ML modules from the project report as REST endpoints:
  5.  Resume Parsing (NER)          -> POST /parse-resume
  6.  AI Resume Screening           -> POST /screen-resume
  7.  Candidate Matching            -> POST /match-score
  17. Skill Gap Analysis            -> POST /skill-gap
  18. Candidate Ranking             -> POST /rank-candidates
  19. Interview Question Generation -> POST /generate-interview-questions
  34. Attrition Prediction          -> POST /predict-attrition
  35. Salary Prediction             -> POST /predict-salary
  --  Candidate Decision Recommendation -> POST /recommend-candidate-decision
  --  Candidate Chatbot                 -> POST /chat

Run with: uvicorn app.main:app --port 8000   (do NOT use --reload in production/demo --
see the note in the "why imports are at the top" comment below)
Docs available at http://localhost:8000/docs once running.

PERFORMANCE NOTE: every `from app.X import Y` below is at the TOP of this file,
not inside each endpoint function. This is deliberate and matters a lot for
latency: importing a module runs its top-level code exactly once, the first
time it's imported. app.resume_parsing's spaCy model alone takes ~12 SECONDS
to load from disk. If these imports were inside the endpoint functions instead
(as an earlier version of this file had them), that 12s load would happen on
the very FIRST request to whichever endpoint touches resume parsing --
appearing as one random, extremely slow request -- rather than once, predictably,
at server startup before any real traffic arrives. Importing everything here
means all models (spaCy NER, XGBoost matching/attrition/salary, TF-IDF
vectorizer) finish loading during `uvicorn` startup, so even the very first
real request is fast.
"""
import time
import logging
import json
import re

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError

from app.schemas import (
    JobInput, ResumeInput, MatchScoreRequest, MatchScoreResponse,
    ScreeningRequest, ScreeningResponse,
    SkillGapRequest, SkillGapResponse, RankCandidatesRequest, RankedCandidate,
    InterviewQuestionsRequest, InterviewQuestionsResponse,
    ResumeParseRequest, ResumeParseResponse,
    EmployeeInput, AttritionResponse,
    JobPostingInput, SalaryResponse,
    CandidateDecisionRequest, CandidateDecisionResponse,
    ChatRequest, ChatResponse,
)

logger = logging.getLogger("uvicorn")

# --- All model loading happens HERE, once, at import time (= server startup) ---
_t0 = time.time()

from app.resume_parsing import parse_resume, build_resume_dict_from_text
from app.screening import (
    extract_job_requirements, extract_resume_profile,
    extract_resume_profile_from_text, screen_resume_against_job,
)
from app.matching import score_candidate
from app.skill_gap import skill_gap_analysis, rank_candidates_for_job
from app.interview import generate_interview_questions
from app.attrition import predict_attrition_risk
from app.salary import predict_salary_range
from app.decision import recommend_candidate_decision
from app.file_extraction import extract_resume_text
from app.chatbot import get_chatbot_response, load_chatbot_model, CHATBOT_VERSION
from app.job_search import parse_jobs

# Load the chatbot model at startup too (fast). If the .joblib is missing,
# the rest of the service still starts and /chat returns a clear 503.
try:
    load_chatbot_model()
except Exception as exc:  # noqa: BLE001
    logger.warning(f"Chatbot model not loaded: {exc}")

logger.info(f"All models loaded in {time.time() - _t0:.2f}s")
logger.info(f"Chatbot version: {CHATBOT_VERSION}")
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Recruitment Platform -- ML Service",
    description="AI/ML endpoints for resume parsing, candidate matching, skill gap analysis, "
                 "candidate ranking, interview question generation, attrition prediction, "
                 "and salary prediction.",
    version="1.1.0",
)


@app.get("/health", tags=["health"])
def health_check():
    """Confirms the service is up and all models loaded successfully at startup."""
    return {"status": "ok", "chatbot_version": CHATBOT_VERSION}


# ---------------------------------------------------------------------------
# Shared helpers for the endpoints below that accept EITHER a JSON body
# (application/json, unchanged from before -- this is what the backend's
# existing integration sends) OR an uploaded resume file (multipart/form-data
# with a 'file' part, for manual testing without typing resume_text). Both
# content types are accepted on the SAME path, so no separate "-file" routes
# and no change needed on the backend's side at all.
#
# FastAPI/Swagger doesn't have a high-level shortcut for "either JSON or
# multipart on one route", so this is done manually: the endpoint takes the
# raw Request, checks the Content-Type header, and branches. openapi_extra
# declares both content types so Swagger's "Try it out" shows a dropdown to
# pick JSON (a text box) or multipart (an actual file picker).
# ---------------------------------------------------------------------------

def _is_multipart(request: Request) -> bool:
    return request.headers.get("content-type", "").lower().startswith("multipart/form-data")


async def _load_json_body(request: Request, model_cls):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body is not valid JSON.")
    try:
        return model_cls(**body)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))


async def _read_uploaded_file_text(request: Request) -> str:
    form = await request.form()
    file = form.get("file")
    if file is None:
        raise HTTPException(status_code=400, detail="Missing 'file' in multipart form data.")
    file_bytes = await file.read()
    return extract_resume_text(file.filename, file_bytes)


async def _read_job_form_field(request: Request) -> JobInput:
    form = await request.form()
    job_str = form.get("job")
    if job_str is None:
        raise HTTPException(status_code=400, detail="Missing 'job' field in multipart form data (a JSON string).")
    try:
        job_dict = json.loads(job_str)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"'job' field is not valid JSON: {e}")
    try:
        return JobInput(**job_dict)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"'job' field failed validation: {e}")


async def _read_optional_form_field(request: Request, field_name: str, default=None):
    form = await request.form()
    value = form.get(field_name)
    return value if value not in (None, "") else default


_SAMPLE_JOB_EXAMPLE = {
    "job_id": "J_000123", "job_title": "Backend Engineer", "seniority": "Senior", "industry": "FinTech",
    "must_have_skills": "python, django, postgresql", "nice_to_have_skills": "docker, aws",
    "description": "We are hiring a Backend Engineer to build scalable APIs.",
    "responsibilities": "Design and maintain REST APIs.", "requirements": "5+ years of Python experience.",
}
_SAMPLE_RESUME_EXAMPLE = {
    "resume_id": "R_004512", "role": "Backend Developer", "seniority": "Senior", "industry": "FinTech",
    "years_experience": 6, "education": "B.Tech", "skills": "python, django, postgresql, docker, git",
    "summary": "Backend developer with 6 years building scalable APIs.", "experience_bullets": "",
}


# ---------------------------------------------------------------------------
# Module 5 -- Resume Parsing
# ---------------------------------------------------------------------------

@app.post(
    "/parse-resume", response_model=ResumeParseResponse, tags=["resume-parsing"],
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {"type": "object", "properties": {"resume_text": {"type": "string"}}, "required": ["resume_text"]},
                    "example": {"resume_text": "Priya Sharma\nSenior Backend Engineer\n\nSKILLS\nPython, Django, PostgreSQL"},
                },
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {"file": {"type": "string", "format": "binary", "description": "Resume file: .pdf or .docx"}},
                        "required": ["file"],
                    }
                },
            },
        }
    },
)
async def parse_resume_endpoint(request: Request):
    if _is_multipart(request):
        resume_text = await _read_uploaded_file_text(request)
    else:
        payload = await _load_json_body(request, ResumeParseRequest)
        resume_text = payload.resume_text

    return parse_resume(resume_text)


# ---------------------------------------------------------------------------
# Module 6 -- AI Resume Screening (standalone from Module 7 matching)
# ---------------------------------------------------------------------------

@app.post(
    "/screen-resume", response_model=ScreeningResponse, response_model_exclude_none=True, tags=["screening"],
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {"type": "object"},
                    "example": {"job": _SAMPLE_JOB_EXAMPLE, "resume": _SAMPLE_RESUME_EXAMPLE},
                },
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string", "format": "binary", "description": "Resume file: .pdf or .docx"},
                            "job": {"type": "string", "description": "Job details as a JSON string", "example": json.dumps(_SAMPLE_JOB_EXAMPLE)},
                        },
                        "required": ["file", "job"],
                    }
                },
            },
        }
    },
)
async def screen_resume_endpoint(request: Request):
    if _is_multipart(request):
        job_obj = await _read_job_form_field(request)
        resume_text = await _read_uploaded_file_text(request)
        resume_profile = extract_resume_profile_from_text(resume_text)
        resume_text_for_similarity = None
    else:
        payload = await _load_json_body(request, ScreeningRequest)
        if (payload.resume is None) == (payload.resume_text is None):
            raise HTTPException(status_code=400, detail="Provide exactly one of 'resume' or 'resume_text'.")
        job_obj = payload.job
        if payload.resume is not None:
            resume_profile = extract_resume_profile(payload.resume.dict())
            resume_text_for_similarity = f"{payload.resume.summary} {payload.resume.experience_bullets}"
        else:
            resume_profile = extract_resume_profile_from_text(payload.resume_text)
            resume_text_for_similarity = None

    job_req = extract_job_requirements(job_obj.dict())
    job_text = f"{job_obj.description} {job_obj.responsibilities} {job_obj.requirements}"
    return screen_resume_against_job(resume_profile, job_req, resume_text=resume_text_for_similarity, job_text=job_text)


# ---------------------------------------------------------------------------
# Modules 6/7 -- Candidate Matching
# ---------------------------------------------------------------------------

@app.post(
    "/match-score", response_model=MatchScoreResponse, response_model_exclude_none=True, tags=["matching"],
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {"type": "object"},
                    "example": {"job": _SAMPLE_JOB_EXAMPLE, "resume": _SAMPLE_RESUME_EXAMPLE},
                },
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string", "format": "binary", "description": "Resume file: .pdf or .docx"},
                            "job": {"type": "string", "description": "Job details as a JSON string", "example": json.dumps(_SAMPLE_JOB_EXAMPLE)},
                            "resume_seniority": {"type": "string", "description": "Optional -- can't be reliably extracted from resume text"},
                            "resume_industry": {"type": "string", "description": "Optional -- can't be reliably extracted from resume text"},
                        },
                        "required": ["file", "job"],
                    }
                },
            },
        }
    },
)
async def match_score_endpoint(request: Request, explain: bool = False):
    if _is_multipart(request):
        job_obj = await _read_job_form_field(request)
        resume_text = await _read_uploaded_file_text(request)
        resume_seniority = await _read_optional_form_field(request, "resume_seniority")
        resume_industry = await _read_optional_form_field(request, "resume_industry")
        resume_dict = build_resume_dict_from_text(resume_text, seniority=resume_seniority, industry=resume_industry)
    else:
        payload = await _load_json_body(request, MatchScoreRequest)
        if (payload.resume is None) == (payload.resume_text is None):
            raise HTTPException(status_code=400, detail="Provide exactly one of 'resume' or 'resume_text'.")
        job_obj = payload.job
        if payload.resume is not None:
            resume_dict = payload.resume.dict()
        else:
            resume_dict = build_resume_dict_from_text(
                payload.resume_text, seniority=payload.resume_seniority, industry=payload.resume_industry
            )

    result = score_candidate(job_obj.dict(), resume_dict, explain=explain)
    return {
        "job_id": job_obj.job_id,
        "resume_id": resume_dict["resume_id"],
        "result": result["result"],
        "match_probability_pct": result["match_probability_pct"],
        "features": result["features"],
        "explanation": result.get("explanation"),
    }


# ---------------------------------------------------------------------------
# Module 17 -- Skill Gap Analysis
# ---------------------------------------------------------------------------

@app.post(
    "/skill-gap", response_model=SkillGapResponse, tags=["skill-gap"],
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {"type": "object"},
                    "example": {"job": _SAMPLE_JOB_EXAMPLE, "resume": _SAMPLE_RESUME_EXAMPLE},
                },
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string", "format": "binary", "description": "Resume file: .pdf or .docx"},
                            "job": {"type": "string", "description": "Job details as a JSON string", "example": json.dumps(_SAMPLE_JOB_EXAMPLE)},
                        },
                        "required": ["file", "job"],
                    }
                },
            },
        }
    },
)
async def skill_gap_endpoint(request: Request):
    if _is_multipart(request):
        job_obj = await _read_job_form_field(request)
        resume_text = await _read_uploaded_file_text(request)
        resume_dict = build_resume_dict_from_text(resume_text)
    else:
        payload = await _load_json_body(request, SkillGapRequest)
        if (payload.resume is None) == (payload.resume_text is None):
            raise HTTPException(status_code=400, detail="Provide exactly one of 'resume' or 'resume_text'.")
        job_obj = payload.job
        if payload.resume is not None:
            resume_dict = payload.resume.dict()
        else:
            resume_dict = build_resume_dict_from_text(payload.resume_text)

    return skill_gap_analysis(job_obj.dict(), resume_dict)


# ---------------------------------------------------------------------------
# Module 18 -- Candidate Ranking
# ---------------------------------------------------------------------------

@app.post("/rank-candidates", response_model=list[RankedCandidate], tags=["ranking"])
def rank_candidates_endpoint(payload: RankCandidatesRequest):
    candidates_dicts = [c.dict() for c in payload.candidates]
    for i, text in enumerate(payload.candidate_texts):
        candidates_dicts.append(build_resume_dict_from_text(text, resume_id=f"TEXT_{i}"))

    if not candidates_dicts:
        raise HTTPException(status_code=400, detail="Provide at least one candidate via 'candidates' or 'candidate_texts'.")

    job_dict = payload.job.dict()
    return rank_candidates_for_job(job_dict, candidates_dicts, top_n=payload.top_n)


# ---------------------------------------------------------------------------
# Module 19 -- Interview Question Generation
# ---------------------------------------------------------------------------

@app.post(
    "/generate-interview-questions", response_model=InterviewQuestionsResponse, response_model_exclude_none=True, tags=["interview-questions"],
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {"type": "object"},
                    "example": {"job": _SAMPLE_JOB_EXAMPLE, "resume": _SAMPLE_RESUME_EXAMPLE, "num_questions": 5, "provider": "anthropic"},
                },
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string", "format": "binary", "description": "Resume file: .pdf or .docx"},
                            "job": {"type": "string", "description": "Job details as a JSON string", "example": json.dumps(_SAMPLE_JOB_EXAMPLE)},
                            "num_questions": {"type": "integer", "default": 5},
                            "provider": {"type": "string", "default": "anthropic"},
                        },
                        "required": ["file", "job"],
                    }
                },
            },
        }
    },
)
async def generate_interview_questions_endpoint(request: Request):
    if _is_multipart(request):
        job_obj = await _read_job_form_field(request)
        resume_text = await _read_uploaded_file_text(request)
        resume_dict = build_resume_dict_from_text(resume_text)
        num_questions = int(await _read_optional_form_field(request, "num_questions", default=5))
        provider = await _read_optional_form_field(request, "provider", default="anthropic")
    else:
        payload = await _load_json_body(request, InterviewQuestionsRequest)
        job_obj = payload.job
        resume_dict = payload.resume.dict()
        num_questions = payload.num_questions
        provider = payload.provider

    result = generate_interview_questions(job_obj.dict(), resume_dict, num_questions=num_questions, provider=provider)
    return {
        "job_id": job_obj.job_id,
        "resume_id": resume_dict["resume_id"],
        "source": result["source"],
        "questions": result["questions"],
        "fallback_reason": result.get("fallback_reason"),
    }


# ---------------------------------------------------------------------------
# Module 34 -- Attrition Prediction
# ---------------------------------------------------------------------------

@app.post("/predict-attrition", response_model=AttritionResponse, tags=["attrition"])
def predict_attrition_endpoint(employee: EmployeeInput):
    return predict_attrition_risk(employee.dict())


# ---------------------------------------------------------------------------
# Module 35 -- Salary Prediction
# ---------------------------------------------------------------------------

@app.post("/predict-salary", response_model=SalaryResponse, tags=["salary"])
def predict_salary_endpoint(job_posting: JobPostingInput):
    return predict_salary_range(job_posting.dict())


# ---------------------------------------------------------------------------
# Candidate Decision Recommendation -- supports Hiring Manager's
# "Approve/Reject Candidates" permission. Returns a SUGGESTED recommendation
# only; the actual approve/reject decision is made and stored by the backend.
# ---------------------------------------------------------------------------

@app.post("/recommend-candidate-decision", response_model=CandidateDecisionResponse, response_model_exclude_none=True, tags=["decision"])
def recommend_candidate_decision_endpoint(payload: CandidateDecisionRequest):
    if payload.match_probability_pct is None and (payload.job is None or payload.resume is None):
        raise HTTPException(
            status_code=400,
            detail="Provide either 'match_probability_pct' directly, or both 'job' and 'resume' to compute it fresh."
        )
    if payload.match_probability_pct is not None and (payload.job is not None or payload.resume is not None):
        raise HTTPException(
            status_code=400,
            detail="Provide either 'match_probability_pct' OR 'job'+'resume', not both."
        )

    job_id = payload.job.job_id if payload.job else None
    resume_id = payload.resume.resume_id if payload.resume else None

    if payload.match_probability_pct is not None:
        match_pct = payload.match_probability_pct
    else:
        result = score_candidate(payload.job.dict(), payload.resume.dict())
        match_pct = result["match_probability_pct"]

    feedback_dict = {k: v for k, v in payload.interview_feedback.dict().items() if v is not None}
    result = recommend_candidate_decision(match_pct, feedback_dict)
    result["job_id"] = job_id
    result["resume_id"] = resume_id
    return result


# ---------------------------------------------------------------------------
# Candidate Chatbot
# ---------------------------------------------------------------------------

_USER_PREFIX = re.compile(r"^\s*(user|candidate|you|human|me)\s*[:\-]\s*", re.I)
_BOT_PREFIX = re.compile(r"^\s*(assistant|bot|talentai|ai|system)\s*[:\-]\s*", re.I)
# Openings of the chatbot's own replies, used when a plain string has no prefix.
_BOT_REPLY_START = re.compile(
    r"^\s*(hello! i can help|hi! .*talentai|i found|there (is|are) \d+|here are|i'm not sure|"
    r"sorry,|to apply|to upload|to register|you can track|your interview|you're welcome|goodbye|"
    r"this assistant is|current job listings|which job role|what would you like to know|"
    r"no job matches|i couldn't find|ask me|\*\*|hello! \U0001F44B|\U0001F44D|i'm doing great|i'm the candidate|"
    r"here's what i can|glad i could|alright!|great!|i can give career|there's no|"
    r"sorry about that|sorry, i didn't)", re.I)


def _history_to_dicts(items) -> list:
    """Accept [{"role","content"}] objects or Java List<String> and return role/content dicts."""
    out = []
    for m in items or []:
        if isinstance(m, str):
            text = m.strip()
            if not text:
                continue
            if _USER_PREFIX.match(text):
                out.append({"role": "user", "content": _USER_PREFIX.sub("", text, count=1)})
            elif _BOT_PREFIX.match(text) or _BOT_REPLY_START.match(text):
                out.append({"role": "assistant", "content": _BOT_PREFIX.sub("", text, count=1)})
            else:
                out.append({"role": "user", "content": text})
        else:
            d = m if isinstance(m, dict) else m.dict()
            d = {"role": d.get("role") or d.get("sender") or "user",
                 "content": d.get("content") or d.get("message") or d.get("text") or ""}
            role = str(d.get("role", "user")).lower()
            d["role"] = "assistant" if role in ("assistant", "bot", "ai", "system") else "user"
            out.append(d)
    return out


@app.post("/chat", response_model=ChatResponse, tags=["chatbot"])
def chat_endpoint(payload: ChatRequest):
    # Accept camelCase / alternative names the Java DTO might use.
    extra = getattr(payload, "model_extra", None) or {}
    jobs = [j.dict(exclude_none=True) for j in (payload.jobs or [])]
    if not jobs:
        for key in ("jobList", "jobsList", "job_list", "availableJobs", "openJobs", "jobDetails",
                    "jobPostings", "jobOpenings"):
            if isinstance(extra.get(key), list):
                jobs = [j for j in extra[key] if isinstance(j, dict)]
                break
    raw_history = payload.conversation_history or extra.get("conversationHistory") or extra.get("history")
    history = _history_to_dicts(raw_history)
    job_context = payload.job_context or extra.get("jobContext")
    if job_context is not None and not isinstance(job_context, str):
        job_context = json.dumps(job_context, default=str)      # list/object sent instead of text

    # Sanitized diagnostics: counts and field NAMES only, never messages or job contents.
    parsed = len(parse_jobs(job_context, jobs))
    logger.info(
        f"/chat role={payload.role} history_items={len(history)} jobs_received={len(jobs)} "
        f"job_fields={sorted(jobs[0].keys()) if jobs else []} "
        f"job_context_chars={len(job_context or '')} jobs_usable={parsed} "
        f"other_fields={sorted(extra.keys())}")
    if (jobs or job_context) and parsed == 0:
        logger.warning("/chat received job data but no usable job (each job needs a title / jobTitle).")

    try:
        reply = get_chatbot_response(
            message=payload.message,
            conversation_history=history,
            job_context=job_context,
            role=payload.role,
            jobs=jobs,
        )
    except ValueError as exc:
        # e.g. empty message
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        # e.g. chatbot model file missing
        logger.error(f"/chat failed: {exc}")
        raise HTTPException(status_code=503, detail=str(exc))
    return {"response": reply}
