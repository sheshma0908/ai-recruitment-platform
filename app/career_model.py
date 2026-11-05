"""
Career question model used by train_chatbot.py and Chatbot.py.

The Career Guidance QA Dataset has 54 job roles, and every role has the
same 10 question topics, each asked 3 ways (30 rows per role, same order).

The model answers a question in two steps:
  1. Topic classifier (TF-IDF + Logistic Regression, trained on the dataset
     questions with the role name masked) -> which of the 10 topics.
  2. Role detector -> which of the 54 roles the question is about.
Answer = the dataset answer for that role and topic.
"""

import re
from difflib import SequenceMatcher

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

# Row position within each role (0-29) -> topic
TOPIC_BY_POSITION = (
    ["responsibilities"] * 3 + ["skills"] * 2 + ["tools_technologies"] * 4 +
    ["career_growth"] * 3 + ["education"] * 3 + ["daily_tasks"] * 3 +
    ["salary"] * 3 + ["work_environment"] * 3 + ["challenges"] * 3 +
    ["getting_started"] * 3
)

TOPIC_LABELS = {
    "responsibilities": "what the role involves",
    "skills": "skills needed",
    "tools_technologies": "tools and technologies",
    "career_growth": "career growth",
    "education": "education and qualifications",
    "daily_tasks": "a typical day",
    "salary": "salary",
    "work_environment": "where they work",
    "challenges": "challenges",
    "getting_started": "how to get started",
}

ROLE_MASK = "jobrole"

# Everyday words -> words used in the dataset questions.
SYNONYMS = [
    (r"\bhow much\b", "salary earn"), (r"\b(earn|earns|earning|make|makes)\b", "earn salary"),
    (r"\b(pay|paid|package|ctc|income|stipend|compensation)\b", "salary"),
    (r"\b(become|becoming)\b", "start career become"),
    (r"\b(growth|promotion|promoted|promote|future|scope|grow)\b", "career growth path progress"),
    (r"\b(duties|duty|role involve|job role|work on)\b", "responsibilities"),
    (r"\b(degree|course|study|studies|certification|certifications)\b", "degree qualifications education"),
    (r"\b(language|languages|framework|frameworks|stack)\b", "programming languages technologies"),
    (r"\b(daily|everyday|every day|day to day|routine)\b", "daily tasks typical day"),
    (r"\b(companies|company|industries|industry|employer|employers|office|remote)\b", "companies work environment"),
    (r"\b(hard|tough|difficult|cons|problems|issues)\b", "challenges difficulties problems"),
    (r"\b(fresher|freshers|beginner|beginners|entry level|no experience|switch)\b", "start entry-level begin"),
]

# Extra names people use for the dataset roles (lowercase).
ROLE_ALIASES = {
    "Machine Learning Engineer": ["ml engineer", "ml engineers"],
    "AI Software Engineer": ["ai engineer", "ai developer"],
    "AI Researcher": ["ai scientist", "ai research"],
    "UX/UI Designer": ["ux designer", "ui designer", "ui ux designer", "ux ui designer", "ui/ux designer", "product designer"],
    "UX Researcher": ["user researcher", "ux research"],
    "QA Engineer": ["qa", "tester", "software tester", "test engineer", "quality assurance engineer"],
    "Site Reliability Engineer (SRE)": ["sre", "reliability engineer"],
    "Systems Administrator": ["system administrator", "sysadmin", "system admin"],
    "Business Intelligence Analyst": ["bi analyst", "bi developer"],
    "Mobile App Developer": ["app developer", "android developer", "ios developer", "mobile developer"],
    "Web Developer": ["frontend developer", "front end developer", "backend developer", "full stack developer", "web designer"],
    "Software Engineer": ["software developer", "programmer", "sde", "software engineering"],
    "Data Visualization Specialist": ["data visualization", "data viz specialist"],
    "SEO Specialist": ["seo", "seo expert", "seo analyst"],
    "SEM (Search Engine Marketing) Specialist": ["sem specialist", "search engine marketing", "sem"],
    "Corporate Social Responsibility (CSR) Manager (Entry-level)": ["csr manager", "csr"],
    "Technical Support Engineer": ["tech support", "technical support", "support engineer"],
    "Application Support Analyst": ["application support", "app support"],
    "E-commerce Specialist": ["ecommerce specialist", "e commerce specialist"],
    "Cloud Consultant (Junior level)": ["cloud engineer"],
    "Cloud Database Engineer": ["database engineer", "dba", "database administrator"],
    "Digital Marketing Manager (Junior level)": ["digital marketer", "digital marketing"],
}


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9+#/\-\s]", " ", text)
    text = text.replace("/", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def expand_synonyms(text: str) -> str:
    for pattern, extra in SYNONYMS:
        if re.search(pattern, text):
            text = f"{text} {extra}"
    return text


def base_role_name(role: str) -> str:
    """'Product Manager (Entry-level)' -> 'Product Manager'."""
    return re.sub(r"\s*\((junior level|entry-level|entry level)\)", "", role, flags=re.I).strip()


def role_names(role: str) -> list:
    """All lowercase names that refer to a dataset role."""
    base = base_role_name(role)
    names = {role, base, base + "s", "junior " + base}
    # "Site Reliability Engineer (SRE)" -> also without the acronym, and the acronym itself
    inner = re.findall(r"\(([^)]+)\)", base)
    no_paren = re.sub(r"\s*\([^)]*\)", "", base).strip()
    names.update({no_paren, no_paren + "s"})
    names.update(i for i in inner if len(i) <= 5)
    names.update(ROLE_ALIASES.get(role, []))
    return sorted({normalize_text(n) for n in names if n}, key=len, reverse=True)


class CareerModel:
    def __init__(self):
        self.topic_clf = Pipeline([
            ("features", FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
            ])),
            ("clf", LogisticRegression(C=10, max_iter=3000)),
        ])

    # ---------------- training ----------------

    @staticmethod
    def mask_role(question: str, role: str) -> str:
        q = normalize_text(question)
        for name in role_names(role):
            q = re.sub(rf"\b{re.escape(name)}\b", ROLE_MASK, q)
        q = re.sub(r"\bjunior\b", "", q)
        return re.sub(r"\s+", " ", q).strip()

    def topic_features(self, masked_question: str) -> str:
        return expand_synonyms(masked_question)

    def fit(self, roles, questions, answers, topics):
        self.roles = sorted(set(roles))
        self.answers = {}            # (role, topic) -> [(question, answer), ...]
        for r, q, a, t in zip(roles, questions, answers, topics):
            self.answers.setdefault((r, t), []).append((q, a))

        self.role_lookup = []        # (name, role), longest names first
        for r in self.roles:
            for n in role_names(r):
                self.role_lookup.append((n, r))
        self.role_lookup.sort(key=lambda x: len(x[0]), reverse=True)

        X = [self.topic_features(self.mask_role(q, r)) for r, q in zip(roles, questions)]
        self.topic_clf.fit(X, topics)
        return self

    # ---------------- inference ----------------

    def detect_role(self, text: str):
        """Return (role, matched_name) or (None, None). Handles small typos."""
        q = f" {normalize_text(text)} "
        for name, role in self.role_lookup:
            if f" {name} " in q:
                return role, name

        # fuzzy match on word windows (e.g. "mobile app developr")
        words = q.split()
        best = (0.0, None, None)
        for name, role in self.role_lookup:
            n = len(name.split())
            if n < 2 and len(name) < 6:
                continue            # skip short names like "qa" for fuzzy matching
            for i in range(len(words) - n + 1):
                window = " ".join(words[i:i + n])
                score = SequenceMatcher(None, window, name).ratio()
                if score > best[0]:
                    best = (score, role, name)
        if best[0] >= 0.86:
            return best[1], best[2]
        return None, None

    def predict_topic(self, text: str, role_name: str = None):
        q = normalize_text(text)
        if role_name:
            q = re.sub(rf"\b{re.escape(role_name)}\b", ROLE_MASK, q)
        probs = self.topic_clf.predict_proba([self.topic_features(q)])[0]
        i = int(np.argmax(probs))
        return self.topic_clf.classes_[i], float(probs[i])

    def answer(self, role: str, topic: str, text: str = "") -> tuple:
        """Pick the dataset answer for (role, topic) whose question is closest to the user's text."""
        options = self.answers.get((role, topic), [])
        if not options:
            return None, None
        q = normalize_text(text)
        best = max(options, key=lambda qa: SequenceMatcher(None, q, normalize_text(qa[0])).ratio())
        return best   # (dataset_question, dataset_answer)
