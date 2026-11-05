"""
Exercises every endpoint of the ML service with realistic sample data.
Run with: python3 test_api.py
"""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

SAMPLE_JOB = {
    "job_id": "J_TEST001",
    "job_title": "Backend Engineer",
    "seniority": "Senior",
    "industry": "FinTech",
    "must_have_skills": "['Python' 'Django' 'PostgreSQL']",
    "nice_to_have_skills": "['Docker' 'AWS']",
    "description": "We are hiring a Backend Engineer to build scalable APIs for our FinTech platform.",
    "responsibilities": "Design and maintain REST APIs. Own database schema design.",
    "requirements": "5+ years of Python experience. Strong SQL skills.",
}

SAMPLE_RESUME = {
    "resume_id": "R_TEST001",
    "role": "Backend Developer",
    "seniority": "Senior",
    "industry": "FinTech",
    "years_experience": 6,
    "education": "B.Tech",
    "skills": "['Python' 'Django' 'PostgreSQL' 'Docker' 'Git']",
    "summary": "Backend developer with 6 years building scalable APIs in FinTech.",
    "experience_bullets": "['Built REST APIs serving 1M+ requests/day' 'Led a team of 3 engineers']",
}

SAMPLE_CANDIDATE_2 = {
    "resume_id": "R_TEST002",
    "role": "Frontend Developer",
    "seniority": "Junior",
    "industry": "Retail",
    "years_experience": 1,
    "education": "B.Sc",
    "skills": "['HTML' 'CSS' 'JavaScript']",
    "summary": "Junior frontend developer.",
    "experience_bullets": "['Built landing pages']",
}

SAMPLE_EMPLOYEE = {
    "Age": 34, "BusinessTravel": "Travel_Rarely", "DailyRate": 800,
    "Department": "Research & Development", "DistanceFromHome": 5, "Education": 3,
    "EducationField": "Life Sciences", "EnvironmentSatisfaction": 3, "Gender": "Male",
    "HourlyRate": 65, "JobInvolvement": 3, "JobLevel": 2, "JobRole": "Research Scientist",
    "JobSatisfaction": 3, "MaritalStatus": "Married", "MonthlyIncome": 5500,
    "MonthlyRate": 15000, "NumCompaniesWorked": 2, "OverTime": "No",
    "PercentSalaryHike": 14, "PerformanceRating": 3, "RelationshipSatisfaction": 3,
    "StockOptionLevel": 1, "TotalWorkingYears": 8, "TrainingTimesLastYear": 2,
    "WorkLifeBalance": 3, "YearsAtCompany": 5, "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 3,
}

SAMPLE_JOB_POSTING = {
    "Job_Title": "Backend Developer", "Company_Type": "MNC",
    "Industry": "Information Technology", "City": "Bangalore",
    "Location_Tier": "Tier 1", "Experience_Level": "Senior (6-10 yrs)",
    "Job_Type": "Full-Time", "Work_Mode": "Hybrid",
    "Education_Required": "B.Tech/B.E.",
    "Skills_Required": "Python, Django, PostgreSQL, AWS",
    "Openings": 2, "Applicants": 150, "Company_Rating": 4.1,
}

SAMPLE_RESUME_TEXT = """Govardhana K
Senior Software Engineer

Bengaluru, Karnataka - Email me on Indeed: indeed.com/r/Govardhana-K/b2de315

Total IT experience 5 Years 6 Months
Cloud Lending Solutions INC 4 Month - Salesforce Developer
Oracle 5 Years 2 Month - Core Java Developer

EDUCATION
B.E in Computer Science
B.V.B College of Engineering and Technology - Hubli, Karnataka

SKILLS
Java (5 years), SQL (5 years), Salesforce (1 year), Python (2 years)
"""


def check(name, response, expect_status=200):
    status = "PASS" if response.status_code == expect_status else "FAIL"
    print(f"[{status}] {name} -> HTTP {response.status_code}")
    if status == "FAIL":
        print("   Response:", response.text[:500])
    else:
        print("   Sample output:", str(response.json())[:300])
    print()
    return response


print("=" * 70)
print("1. Health check")
print("=" * 70)
check("GET /health", client.get("/health"))

print("=" * 70)
print("2. Resume Parsing (Module 5)")
print("=" * 70)
check("POST /parse-resume", client.post("/parse-resume", json={"resume_text": SAMPLE_RESUME_TEXT}))

print("=" * 70)
print("3. AI Resume Screening (Module 6) -- structured resume input")
print("=" * 70)
check("POST /screen-resume (structured)", client.post(
    "/screen-resume",
    json={"job": SAMPLE_JOB, "resume": SAMPLE_RESUME}
))

print("=" * 70)
print("3b. AI Resume Screening (Module 6) -- raw resume text input")
print("=" * 70)
check("POST /screen-resume (raw text)", client.post(
    "/screen-resume",
    json={"job": SAMPLE_JOB, "resume_text": SAMPLE_RESUME_TEXT}
))

print("=" * 70)
print("3c. AI Resume Screening -- error case: both resume and resume_text given")
print("=" * 70)
check(
    "POST /screen-resume (both provided, should fail)",
    client.post("/screen-resume", json={"job": SAMPLE_JOB, "resume": SAMPLE_RESUME, "resume_text": SAMPLE_RESUME_TEXT}),
    expect_status=400,
)

print("=" * 70)
print("4. Candidate Matching (Modules 6/7) -- old-style structured call (regression check)")
print("=" * 70)
check("POST /match-score (structured, old style)", client.post(
    "/match-score?explain=true",
    json={"job": SAMPLE_JOB, "resume": SAMPLE_RESUME}
))

print("=" * 70)
print("4b. Candidate Matching -- NEW: raw resume_text instead of structured resume")
print("=" * 70)
check("POST /match-score (resume_text)", client.post(
    "/match-score",
    json={
        "job": SAMPLE_JOB,
        "resume_text": SAMPLE_RESUME_TEXT,
        "resume_seniority": "Senior",
        "resume_industry": "IT",
    }
))

print("=" * 70)
print("4c. Candidate Matching -- error case: neither resume nor resume_text given")
print("=" * 70)
check(
    "POST /match-score (neither given, should fail)",
    client.post("/match-score", json={"job": SAMPLE_JOB}),
    expect_status=400,
)

print("=" * 70)
print("5. Skill Gap Analysis (Module 17) -- old-style structured call (regression check)")
print("=" * 70)
check("POST /skill-gap (structured, old style)", client.post(
    "/skill-gap",
    json={"job": SAMPLE_JOB, "resume": SAMPLE_CANDIDATE_2}
))

print("=" * 70)
print("5b. Skill Gap Analysis -- NEW: raw resume_text")
print("=" * 70)
check("POST /skill-gap (resume_text)", client.post(
    "/skill-gap",
    json={"job": SAMPLE_JOB, "resume_text": SAMPLE_RESUME_TEXT}
))

print("=" * 70)
print("6. Candidate Ranking (Module 18) -- old-style structured call (regression check)")
print("=" * 70)
check("POST /rank-candidates (structured, old style)", client.post(
    "/rank-candidates",
    json={"job": SAMPLE_JOB, "candidates": [SAMPLE_RESUME, SAMPLE_CANDIDATE_2], "top_n": 5}
))

print("=" * 70)
print("6b. Candidate Ranking -- NEW: mix of structured candidates AND raw texts")
print("=" * 70)
check("POST /rank-candidates (mixed)", client.post(
    "/rank-candidates",
    json={
        "job": SAMPLE_JOB,
        "candidates": [SAMPLE_CANDIDATE_2],
        "candidate_texts": [SAMPLE_RESUME_TEXT],
        "top_n": 5,
    }
))

print("=" * 70)
print("6. Interview Question Generation (Module 19)")
print("=" * 70)
check("POST /generate-interview-questions", client.post(
    "/generate-interview-questions",
    json={"job": SAMPLE_JOB, "resume": SAMPLE_RESUME, "num_questions": 5, "provider": "anthropic"}
))

print("=" * 70)
print("7. Attrition Prediction (Module 34)")
print("=" * 70)
check("POST /predict-attrition", client.post("/predict-attrition", json=SAMPLE_EMPLOYEE))

print("=" * 70)
print("8. Salary Prediction (Module 35)")
print("=" * 70)
check("POST /predict-salary", client.post("/predict-salary", json=SAMPLE_JOB_POSTING))

print("=" * 70)
print("9. Candidate Decision Recommendation -- using an existing match score")
print("=" * 70)
check("POST /recommend-candidate-decision (with match_probability_pct)", client.post(
    "/recommend-candidate-decision",
    json={
        "match_probability_pct": 83.0,
        "interview_feedback": {
            "technical_skills": 9, "communication": 8, "problem_solving": 9,
            "teamwork": 8, "leadership": 7
        }
    }
))

print("=" * 70)
print("9b. Candidate Decision Recommendation -- computing match score fresh from job+resume")
print("=" * 70)
check("POST /recommend-candidate-decision (with job+resume)", client.post(
    "/recommend-candidate-decision",
    json={
        "job": SAMPLE_JOB,
        "resume": SAMPLE_RESUME,
        "interview_feedback": {
            "technical_skills": 4, "communication": 5, "problem_solving": 3,
            "teamwork": 6, "leadership": 5
        }
    }
))

print("=" * 70)
print("9c. Candidate Decision -- error case: neither match_probability_pct nor job+resume given")
print("=" * 70)
check(
    "POST /recommend-candidate-decision (missing input, should fail)",
    client.post("/recommend-candidate-decision", json={"interview_feedback": {"technical_skills": 8}}),
    expect_status=400,
)

print("=" * 70)
print("9d. Candidate Decision -- error case: BOTH match_probability_pct AND job+resume given")
print("=" * 70)
check(
    "POST /recommend-candidate-decision (both given, should fail)",
    client.post("/recommend-candidate-decision", json={
        "match_probability_pct": 80, "job": SAMPLE_JOB, "resume": SAMPLE_RESUME,
        "interview_feedback": {"technical_skills": 8}
    }),
    expect_status=400,
)

print("=" * 70)
print("10. Error handling -- empty candidate list should return 400")
print("=" * 70)
check(
    "POST /rank-candidates (empty list)",
    client.post("/rank-candidates", json={"job": SAMPLE_JOB, "candidates": [], "top_n": 5}),
    expect_status=400,
)

print("=" * 70)
print("11. FILE UPLOAD -- same URLs as above, sent as multipart/form-data instead of JSON")
print("=" * 70)

import io
import json as _json
from reportlab.pdfgen import canvas as _canvas
import docx as _docx

def _make_test_pdf_bytes():
    buf = io.BytesIO()
    c = _canvas.Canvas(buf)
    for i, line in enumerate([
        "Test Candidate", "Backend Developer", "",
        "SKILLS", "Python, Django, PostgreSQL, Docker, Git",
    ]):
        c.drawString(72, 750 - i * 20, line)
    c.save()
    return buf.getvalue()

def _make_test_docx_bytes():
    buf = io.BytesIO()
    d = _docx.Document()
    d.add_paragraph("Test Candidate")
    d.add_paragraph("Backend Developer")
    d.add_paragraph("SKILLS")
    d.add_paragraph("Python, Django, PostgreSQL, Docker, Git")
    d.save(buf)
    return buf.getvalue()

pdf_bytes = _make_test_pdf_bytes()
docx_bytes = _make_test_docx_bytes()
SAMPLE_JOB_JSON = _json.dumps(SAMPLE_JOB)

check("POST /parse-resume (file upload, PDF)", client.post(
    "/parse-resume", files={"file": ("resume.pdf", pdf_bytes, "application/pdf")}
))
check("POST /parse-resume (file upload, DOCX)", client.post(
    "/parse-resume", files={"file": ("resume.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
))
check("POST /screen-resume (file upload)", client.post(
    "/screen-resume", files={"file": ("resume.pdf", pdf_bytes, "application/pdf")}, data={"job": SAMPLE_JOB_JSON}
))
check("POST /match-score (file upload)", client.post(
    "/match-score", files={"file": ("resume.pdf", pdf_bytes, "application/pdf")}, data={"job": SAMPLE_JOB_JSON}
))
check("POST /skill-gap (file upload)", client.post(
    "/skill-gap", files={"file": ("resume.pdf", pdf_bytes, "application/pdf")}, data={"job": SAMPLE_JOB_JSON}
))
check("POST /generate-interview-questions (file upload)", client.post(
    "/generate-interview-questions", files={"file": ("resume.pdf", pdf_bytes, "application/pdf")}, data={"job": SAMPLE_JOB_JSON}
))
check(
    "POST /parse-resume (file upload, unsupported .txt, should fail)",
    client.post("/parse-resume", files={"file": ("resume.txt", b"plain text", "text/plain")}),
    expect_status=422,
)
check(
    "POST /screen-resume (file upload, invalid job JSON, should fail)",
    client.post("/screen-resume", files={"file": ("resume.pdf", pdf_bytes, "application/pdf")}, data={"job": "not json"}),
    expect_status=400,
)

print("=" * 70)
print("Candidate Chatbot")
print("=" * 70)
check("POST /chat (career question)", client.post(
    "/chat", json={"message": "How much does a web developer earn?", "role": "PUBLIC"}))
check("POST /chat (follow-up uses earlier role)", client.post(
    "/chat", json={"message": "What about the skills?",
                   "conversation_history": [{"role": "user", "content": "How much does a web developer earn?"}]}))
check("POST /chat (job openings from job_context)", client.post(
    "/chat", json={"message": "Any data scientist openings?",
                   "job_context": "Java Developer | Hyderabad | 3-5 yrs\nData Scientist | Bangalore | 2-4 yrs"}))
check("POST /chat (question about recruiters -> restricted message)", client.post(
    "/chat", json={"message": "What does a recruiter do?"}))
check("POST /chat (staff role -> candidates only)", client.post(
    "/chat", json={"message": "Hello", "role": "HR"}))
check("POST /chat (empty message, should fail)", client.post(
    "/chat", json={"message": "   "}), expect_status=400)

print("All endpoint tests completed.")
