"""
Rules shared by training and the chatbot.

The chatbot is ONLY for candidates (job seekers). Anything about the
platform's internal roles - recruiter, HR, hiring manager, super admin,
interviewer, employee - is not answered.
"""

import re

# Roles the Java backend may send. Only these may use the chatbot.
CANDIDATE_ROLES = {"PUBLIC", "CANDIDATE", "USER"}

# Questions mentioning these internal roles are not answered.
RESTRICTED_TOPIC = re.compile(
    r"\b("
    r"recruiters?|recruiting team|recruitment team|talent acquisition|"
    r"hr|h\.r|hrs|human resources?|hr ?managers?|hr ?team|hr ?executives?|"
    r"hiring ?managers?|"
    r"super ?admins?|admins?|admin ?(panel|dashboard|access|rights)|"
    r"interviewers?|interview ?panel|panelists?|"
    r"employees?|staff members?|staff"
    r")\b",
    re.IGNORECASE,
)

# Requests for secrets / internal instructions.
SECRETS_REQUEST = re.compile(
    r"(api ?key|password of|db password|database password|credentials?|"
    r"jwt|secret ?key|access token|system prompt|your instructions|"
    r"ignore (all |the )?(previous|above) instructions|env(ironment)? variables?)",
    re.IGNORECASE,
)


def is_restricted_topic(text: str) -> bool:
    return bool(RESTRICTED_TOPIC.search(text or ""))


def is_secrets_request(text: str) -> bool:
    return bool(SECRETS_REQUEST.search(text or ""))
