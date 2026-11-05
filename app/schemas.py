"""
Request/response schemas for the AI Recruitment Platform's ML service.

Field names match the original training CSVs exactly (jobs.csv, resume.csv, the
IBM attrition dataset, and the India job market dataset) so the backend team can
pass data straight through from the database without renaming anything.
"""
from typing import Any, Optional, List, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared job/resume shapes (used by matching, skill gap, ranking, interview Qs)
# ---------------------------------------------------------------------------

class JobInput(BaseModel):
    job_id: str = Field(..., example="J_000123")
    job_title: str = Field(..., example="Backend Engineer")
    seniority: str = Field(..., example="Senior", description="Junior | Mid | Senior")
    industry: str = Field(..., example="FinTech")
    must_have_skills: str = Field(
        ..., example="['Python' 'Django' 'PostgreSQL']",
        description="Python-list-repr string, matching jobs.csv's original format"
    )
    nice_to_have_skills: str = Field(..., example="['Docker' 'AWS']")
    description: str = Field(default="", example="We are looking for a backend engineer...")
    responsibilities: str = Field(default="", example="Design and maintain REST APIs...")
    requirements: str = Field(default="", example="5+ years of Python experience...")


class ResumeInput(BaseModel):
    resume_id: str = Field(..., example="R_004512")
    role: str = Field(..., example="Software Engineer")
    seniority: str = Field(..., example="Senior")
    industry: str = Field(..., example="FinTech")
    years_experience: int = Field(..., example=6)
    education: str = Field(default="", example="B.Tech")
    skills: str = Field(..., example="['Python' 'Django' 'REST APIs' 'Docker']")
    summary: str = Field(default="", example="Backend engineer with 6 years of experience...")
    experience_bullets: str = Field(default="", example="['Built scalable APIs' 'Led a team of 3']")


class MatchScoreRequest(BaseModel):
    job: JobInput
    resume: Optional[ResumeInput] = Field(
        default=None, description="Provide this OR resume_text, not both."
    )
    resume_text: Optional[str] = Field(
        default=None, description="Raw resume text, parsed via the Module 5 NER model. "
                                    "Provide this OR resume, not both."
    )
    resume_seniority: Optional[str] = Field(
        default=None, description="Only used with resume_text -- seniority can't be reliably "
                                    "extracted from raw text, so pass it here if known."
    )
    resume_industry: Optional[str] = Field(
        default=None, description="Only used with resume_text -- same reasoning as resume_seniority."
    )


class MatchScoreResponse(BaseModel):
    job_id: str
    resume_id: str
    result: str
    match_probability_pct: float
    features: dict
    explanation: Optional[List[dict]] = None


# ---------------------------------------------------------------------------
# Candidate Decision Recommendation (supports Hiring Manager approve/reject)
# ---------------------------------------------------------------------------

class InterviewFeedback(BaseModel):
    technical_skills: Optional[float] = Field(default=None, ge=0, le=10, example=9)
    communication: Optional[float] = Field(default=None, ge=0, le=10, example=8)
    problem_solving: Optional[float] = Field(default=None, ge=0, le=10, example=9)
    teamwork: Optional[float] = Field(default=None, ge=0, le=10, example=8)
    leadership: Optional[float] = Field(default=None, ge=0, le=10, example=7)


class CandidateDecisionRequest(BaseModel):
    job: Optional[JobInput] = Field(
        default=None, description="Provide job+resume to compute the match score fresh, "
                                    "OR provide match_probability_pct directly -- not both."
    )
    resume: Optional[ResumeInput] = None
    match_probability_pct: Optional[float] = Field(
        default=None, ge=0, le=100, description="Use this if you already have a match score "
                                                   "from a previous /match-score call, instead "
                                                   "of recomputing it here."
    )
    interview_feedback: InterviewFeedback


class CandidateDecisionResponse(BaseModel):
    job_id: Optional[str] = None
    resume_id: Optional[str] = None
    match_probability_pct: float
    interview_average_score: float
    combined_score_pct: float
    recommendation: str = Field(description="Recommended | Hold | Not Recommended")
    rationale: str
    note: str


# ---------------------------------------------------------------------------
# Module 6 -- AI Resume Screening (standalone from Module 7 matching)
# ---------------------------------------------------------------------------

class ScreeningRequest(BaseModel):
    job: JobInput
    resume: Optional[ResumeInput] = Field(
        default=None, description="Provide this OR resume_text, not both."
    )
    resume_text: Optional[str] = Field(
        default=None, description="Raw resume text -- parsed via the Module 5 NER model. "
                                    "Provide this OR resume, not both."
    )


class ScreeningDimension(BaseModel):
    dimension: str
    match_pct: Optional[float] = None
    matched: Optional[List[str]] = None
    missing: Optional[List[str]] = None
    resume_value: Optional[object] = None
    job_requirement: Optional[object] = None
    match: Optional[bool] = None
    similarity_pct: Optional[float] = None
    similarity_score: Optional[float] = None
    note: Optional[str] = None


class ScreeningResponse(BaseModel):
    job_id: str
    resume_id: Optional[str] = None
    extracted_resume_profile: dict
    extracted_job_requirements: dict
    screening_dimensions: List[ScreeningDimension]


class SkillGapRequest(BaseModel):
    job: JobInput
    resume: Optional[ResumeInput] = Field(default=None, description="Provide this OR resume_text, not both.")
    resume_text: Optional[str] = Field(default=None, description="Raw resume text. Provide this OR resume, not both.")


class SkillGapResponse(BaseModel):
    job_id: str
    resume_id: str
    must_have_coverage_pct: float
    nice_to_have_coverage_pct: float
    missing_must_have: List[str]
    missing_nice_to_have: List[str]


class RankCandidatesRequest(BaseModel):
    job: JobInput
    candidates: List[ResumeInput] = Field(
        default_factory=list, description="Structured candidate records."
    )
    candidate_texts: List[str] = Field(
        default_factory=list, description="Raw resume texts, parsed via the Module 5 NER model. "
                                            "Can be used together with 'candidates' -- both pools "
                                            "are merged before ranking."
    )
    top_n: int = Field(default=10, ge=1, le=200)


class RankedCandidate(BaseModel):
    rank: int
    resume_id: str
    candidate_role: str
    match_probability_pct: float
    must_have_coverage_pct: float
    missing_must_have: List[str]


class InterviewQuestionsRequest(BaseModel):
    job: JobInput
    resume: ResumeInput
    num_questions: int = Field(default=5, ge=1, le=10)
    provider: str = Field(default="anthropic", description="'anthropic' or 'openai'")


class InterviewQuestion(BaseModel):
    question: str
    category: str
    targets_skill: Optional[str] = None


class InterviewQuestionsResponse(BaseModel):
    job_id: str
    resume_id: str
    source: str = Field(description="'llm:<provider>' if a real LLM call succeeded, else 'offline_fallback'")
    questions: List[InterviewQuestion]
    fallback_reason: Optional[str] = Field(default=None, description="Why the LLM call failed (only present when source is 'offline_fallback')")


# ---------------------------------------------------------------------------
# Resume parsing (NER)
# ---------------------------------------------------------------------------

class ResumeParseRequest(BaseModel):
    resume_text: str = Field(..., description="Raw resume text extracted from a PDF/DOCX upload")


class ResumeParseResponse(BaseModel):
    name: List[str]
    skills_raw: List[str] = Field(description="Skill mentions exactly as the NER model found them (may be multi-line blocks)")
    skills_cleaned: List[str] = Field(description="Individual, deduplicated skills -- ready for matching/skill-gap")
    companies_worked_at: List[str]
    designation: List[str]
    degree: List[str]
    college_name: List[str]
    graduation_year: List[str]
    years_of_experience: List[str]
    email_address: List[str]
    location: List[str]


# ---------------------------------------------------------------------------
# Attrition prediction -- field names match the IBM HR Attrition CSV exactly
# ---------------------------------------------------------------------------

class EmployeeInput(BaseModel):
    Age: int = Field(..., example=34)
    BusinessTravel: str = Field(..., example="Travel_Rarely")
    DailyRate: int = Field(..., example=800)
    Department: str = Field(..., example="Research & Development")
    DistanceFromHome: int = Field(..., example=5)
    Education: int = Field(..., example=3, description="1=Below College ... 5=Doctor")
    EducationField: str = Field(..., example="Life Sciences")
    EnvironmentSatisfaction: int = Field(..., example=3, description="1=Low ... 4=Very High")
    Gender: str = Field(..., example="Male")
    HourlyRate: int = Field(..., example=65)
    JobInvolvement: int = Field(..., example=3)
    JobLevel: int = Field(..., example=2)
    JobRole: str = Field(..., example="Research Scientist")
    JobSatisfaction: int = Field(..., example=3)
    MaritalStatus: str = Field(..., example="Married")
    MonthlyIncome: int = Field(..., example=5500)
    MonthlyRate: int = Field(..., example=15000)
    NumCompaniesWorked: int = Field(..., example=2)
    OverTime: str = Field(..., example="No")
    PercentSalaryHike: int = Field(..., example=14)
    PerformanceRating: int = Field(..., example=3)
    RelationshipSatisfaction: int = Field(..., example=3)
    StockOptionLevel: int = Field(..., example=1)
    TotalWorkingYears: int = Field(..., example=8)
    TrainingTimesLastYear: int = Field(..., example=2)
    WorkLifeBalance: int = Field(..., example=3)
    YearsAtCompany: int = Field(..., example=5)
    YearsInCurrentRole: int = Field(..., example=3)
    YearsSinceLastPromotion: int = Field(..., example=1)
    YearsWithCurrManager: int = Field(..., example=3)


class AttritionResponse(BaseModel):
    attrition_probability_pct: float
    risk_level: str = Field(description="LOW | MEDIUM | HIGH")


# ---------------------------------------------------------------------------
# Salary prediction -- field names match the India job market CSV exactly
# ---------------------------------------------------------------------------

class JobPostingInput(BaseModel):
    Job_Title: str = Field(..., example="Backend Developer")
    Company_Type: str = Field(..., example="MNC", description="MNC | Indian Unicorn | PSU/Govt | Startup")
    Industry: str = Field(..., example="Information Technology")
    City: str = Field(..., example="Bangalore")
    Location_Tier: str = Field(..., example="Tier 1", description="Tier 1 | Tier 2 | Remote")
    Experience_Level: str = Field(
        ..., example="Senior (6-10 yrs)",
        description="Fresher (0-1 yr) | Junior (1-3 yrs) | Mid (3-6 yrs) | Senior (6-10 yrs) | Lead (10+ yrs)"
    )
    Job_Type: str = Field(..., example="Full-Time")
    Work_Mode: str = Field(..., example="Hybrid")
    Education_Required: str = Field(..., example="B.Tech/B.E.")
    Skills_Required: str = Field(..., example="Python, Django, PostgreSQL, AWS")
    Openings: int = Field(default=1, example=2)
    Applicants: int = Field(default=100, example=150)
    Company_Rating: float = Field(default=3.8, example=4.1)


class SalaryResponse(BaseModel):
    estimated_range_lpa: str
    lower_bound_lpa: float
    upper_bound_lpa: float


# ---------------------------------------------------------------------------
# Candidate chatbot (POST /chat)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str = Field(..., example="user", description="'user' or 'assistant'")
    content: str = Field(..., example="How much does a web developer earn?")


class ChatJob(BaseModel):
    """One open job, as the backend already has it. Every field accepts any JSON shape
    (string, number, list, or an object like {"name": "Engineering"}), so a Java entity
    or DTO can be sent as is and never causes a 422. camelCase names from Java
    (jobTitle, requiredSkills, minExperience, ...) are also accepted."""
    class Config:
        extra = "allow"     # keep any other fields the backend sends (camelCase etc.)

    job_id: Optional[Any] = Field(default=None, example="7")
    title: Optional[Any] = Field(default=None, example="Java Full Stack Developer")
    department: Optional[Any] = Field(default=None, example="Engineering")
    location: Optional[Any] = Field(default=None, example="Hyderabad")
    experience: Optional[Any] = Field(default=None, example="2 - 5 Years")
    min_experience: Optional[Any] = Field(default=None, example=2)
    max_experience: Optional[Any] = Field(default=None, example=5)
    employment_type: Optional[Any] = Field(default=None, example="Full-Time")
    work_mode: Optional[Any] = Field(default=None, example="Onsite")
    required_skills: Optional[Any] = Field(default=None, example=["Java", "Spring Boot", "React", "PostgreSQL"])
    preferred_skills: Optional[Any] = Field(default=None, example=["Docker", "Git", "REST API"])
    education: Optional[Any] = Field(default=None, example="B.Tech Computer Science")
    openings: Optional[Any] = Field(default=None, example=2)
    salary_range: Optional[Any] = Field(default=None, example="Not specified")
    closing_date: Optional[Any] = Field(default=None, example="Not specified")
    description: Optional[Any] = Field(default=None, example="We are looking for a Java Full Stack Developer...")
    responsibilities: Optional[Any] = Field(default=None, example="Develop and maintain full stack applications")


class ChatRequest(BaseModel):
    class Config:
        extra = "allow"     # jobList / conversationHistory / jobContext etc. are picked up in main.py
    message: str = Field(..., example="How much does a web developer earn?")
    role: str = Field(default="PUBLIC", example="PUBLIC",
                      description="Logged-in user's role from the backend. Only PUBLIC / CANDIDATE / USER get answers.")
    conversation_history: Optional[List[Any]] = Field(
        default=None,
        description="Earlier messages, oldest first, used for follow-up questions. Either objects "
                    "{\"role\": \"user\"|\"assistant\", \"content\": \"...\"} (also sender/message/text keys) "
                    "or plain strings "
                    "(Java List<String>), optionally prefixed with 'user:' / 'assistant:' / 'bot:'.")
    job_context: Optional[Any] = Field(
        default=None, example="Java Developer | Hyderabad | 3-5 yrs\nData Scientist | Bangalore | 2-4 yrs",
        description="Current job openings from the backend: one job per line ('Title | Dept | City | "
                    "2 - 5 Years | Full-Time | Skills: Java, React | Preferred: Docker'), or a JSON list. "
                    "Job answers come only from this and/or 'jobs'.")
    jobs: Optional[List[ChatJob]] = Field(
        default=None, description="Structured open jobs (recommended over job_context). "
                                  "Used for search by experience, skills, city, job type and title.")


class ChatResponse(BaseModel):
    response: str
