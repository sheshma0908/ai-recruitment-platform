"""
Conversation regression tests for the candidate chatbot (no server needed).

Run from the EMS project root:
    python training/chatbot/tests/test_conversation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.chatbot import get_chatbot_response  # noqa: E402

JOBS = [
    {"title": "Java Full Stack Developer", "department": "Engineering", "location": "Hyderabad",
     "experience": "2 - 5 years", "employmentType": "Full-Time",
     "skills": "Java, Spring Boot, React, PostgreSQL"},
    {"title": "data analyst", "department": "engineering", "location": "Pune",
     "experience": "2 - 4 years", "employmentType": "Full Time", "skills": "python aiml"},
    {"title": "data science", "department": "engineering", "location": "Hyderabad",
     "experience": "0 - 1 years", "employmentType": "Internship", "skills": "python aiml"},
]

# (message, text the reply must contain, text the reply must NOT contain)
SINGLE = [
    # greetings incl. short forms / typos
    ("hlo", "Hello!", "didn't quite get"), ("hii", "Hello!", None), ("hiiii", "Hello!", None),
    ("helo", "Hello!", None), ("hey", "Hello!", None), ("good morning", "Hello!", None),
    # acknowledgements
    ("ok", "\U0001F44D", "didn't quite get"), ("okay", "\U0001F44D", None), ("okk", "\U0001F44D", None),
    ("k", "\U0001F44D", None), ("got it", "\U0001F44D", None), ("alright", "\U0001F44D", None),
    # thanks / bye / wellbeing / help
    ("thank you", "welcome", None), ("thnks", "welcome", None), ("tq", "welcome", None),
    ("bye", "Goodbye", None), ("ok bye", "Goodbye", None), ("how are you", "doing great", None),
    ("what can you do", "help you with", None), ("menu", "help you with", None),
    ("sure", "\U0001F44D", None), ("welcome", "Thank you!", "couldn't find"),
    ("wel come", "Thank you!", "couldn't find"), ("you're welcome", "Thank you!", None),
    ("thank you for helping", "welcome", "couldn't find"),
    ("thanks for the info", "welcome", None), ("thank you so much for your help", "welcome", None),
    ("thank you for the jobs list", "welcome", None), ("sorry", "No problem", None),
    ("have a nice day", "Goodbye", None), ("hello sir", "Hello!", None),
    ("hi bot how are you", "doing great", None), ("nice to meet you", "Thank you!", None),
    ("thank you for helping, show me python jobs", "python jobs", "helping"),
    ("help me with python jobs", "python jobs", "help python"),
    ("2", "1) Open the Jobs page", None),
    # career: roadmap vs skills (the original bug)
    ("how to become a data analyst", "step-by-step roadmap", None),
    ("process to become a data analyst", "step-by-step roadmap", None),
    ("steps to become a software engineer", "step-by-step roadmap", None),
    ("what skills do data analyst should have", "Skills needed", "roadmap"),
    ("skills for data analyst", "Skills needed", "I found"),
    ("data analyst salary", "Salary", None),
    ("skills and salary of a data analyst", "Salary:", None),
    ("what is a data analyst", "role overview", None),
    ("tell me about data scientist", "role overview", None),
    ("tester", "QA Engineer", None),
    ("data analyst vs data scientist", "Data Analyst vs Data Scientist", None),
    ("list career roles", "job roles", None),
    # job search still works
    ("how many job openings are there", "3 job listings", None),
    ("python jobs in hyderabad", "data science", None),
    ("hi, show me python jobs", "python jobs", "Hello!"),
    ("about data analyst", "Pune", None),
    ("how do I apply", "To apply", None),
]

FLOWS = [
    # "ok" after a job list -> suggests asking for details
    (["how many job openings are there", "okay"], "details of"),
    # "okay" right after the greeting -> general ack, not the career-follow-up ack
    (["hlo", "okay"], "Anything else I can help with"),
    # career follow-up keeps the topic, changes the role
    (["how to become a data analyst", "and for a data scientist?"], "become a Data Scientist"),
    # follow-up keeps the role, changes the topic
    (["how to become a data analyst", "ok", "what about salary"], "Data Analyst — Salary"),
    # job follow-up still works
    (["python jobs", "and in hyderabad?"], "Hyderabad"),
]


def main() -> int:
    failures = 0
    for msg, must, must_not in SINGLE:
        r = get_chatbot_response(msg, [], role="CANDIDATE", jobs=JOBS)
        ok = must.lower() in r.lower() and not (must_not and must_not.lower() in r.lower())
        if not ok:
            failures += 1
            print(f"FAIL  {msg!r}\n      expected {must!r}, not {must_not!r}\n      got: {r[:200]!r}")
    for msgs, must in FLOWS:
        history = []
        for m in msgs:
            r = get_chatbot_response(m, history, role="CANDIDATE", jobs=JOBS)
            history += [{"role": "user", "content": m}, {"role": "assistant", "content": r}]
        if must.lower() not in r.lower():
            failures += 1
            print(f"FAIL  flow {msgs}\n      expected {must!r}\n      got: {r[:200]!r}")
    total = len(SINGLE) + len(FLOWS)
    print(f"\n{total - failures}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
