"""
Resume Parsing / NER (Module 5) -- wraps the spaCy model trained in
resume_ner.ipynb. Takes raw resume text and returns a structured profile.
"""
import os
import re
from datetime import date

import spacy

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "resume_ner_model")
nlp = spacy.load(_MODEL_PATH)

# Maps this module's entity labels to the ResumeParseResponse field names
_LABEL_TO_FIELD = {
    "Name": "name",
    "Skills": "skills_raw",
    "Companies worked at": "companies_worked_at",
    "Designation": "designation",
    "Degree": "degree",
    "College Name": "college_name",
    "Graduation Year": "graduation_year",
    "Years of Experience": "years_of_experience",
    "Email Address": "email_address",
    "Location": "location",
}

# ---------------------------------------------------------------------------
# Everything below is MODULE-LEVEL (built exactly once, when this file is
# first imported at server startup) rather than inside the function -- these
# used to be rebuilt from scratch on every single call, including once per
# candidate in a /rank-candidates request with hundreds of resumes.
# ---------------------------------------------------------------------------

_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

_COLON_LABEL_PATTERN = re.compile(r"^[A-Za-z /&]+:\s*")

_CATEGORY_LABELS = (
    r"languages?|frameworks?|tools?|databases?|technologies|technology|"
    r"libraries|library|skills?|platforms?|proficiencies|proficient in|"
    r"frontend|backend|front[- ]end|back[- ]end"
)
_HYPHEN_LABEL_PATTERN = re.compile(rf"^(?:{_CATEGORY_LABELS})\s*-\s*", re.IGNORECASE)

_BULLET_PATTERN = re.compile(r"^[\s\-_*•●○▪◦·‣–—]+")
_PAREN_SUFFIX_PATTERN = re.compile(r"\s*\([^)]*\)\s*$")
_SPLIT_PATTERN = re.compile(r"[,;\n\u2022]")


# ============================================================================
# FALSE POSITIVE FILTERS
# ============================================================================

# A piece whose FIRST word is one of these almost always means the NER span
# swallowed part of a sentence (a project description, a responsibility, an
# instruction) rather than a genuine skill name.
_LEADING_REJECT_WORDS = frozenset({
    "and", "the", "a", "an", "of", "to", "in", "on", "for", "with", "as",
    "is", "are", "using", "based", "this", "that", "from", "by", "at",
    "implementing", "implemented", "integrating", "integrated", "viewing",
    "editing", "deleting", "building", "developed", "designed", "created",
    "added", "built", "mark", "marking", "pause", "paused", "scroll",
    "scrolling", "missing", "allows", "allowing", "enables", "enabling",
    "fostering", "impactful", "proficient",
})

# Whole pieces that are themselves clearly a generic action/description word
# rather than a skill, even standing alone.
_NONSKILL_WHOLE_WORDS = frozenset({
    "delete", "deleting", "edit", "editing", "pause", "paused", "mark",
    "marking", "missing", "scroll", "scrolling", "view", "viewing",
    "implement", "implementing", "implemented", "integrate", "integrating",
    "integrated", "and", "the", "with", "using", "declaration", "declare",
    "signature", "reference", "references",
})
_MAX_WORDS = 4

# Indian state/UT names as a supplementary location signal (deployment-
# specific heuristic -- the platform's data is India-focused).
_INDIAN_STATES_AND_UTS = (
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand",
    "karnataka", "kerala", "madhya pradesh", "maharashtra", "manipur",
    "meghalaya", "mizoram", "nagaland", "odisha", "punjab", "rajasthan",
    "sikkim", "tamil nadu", "telangana", "tripura", "uttar pradesh",
    "uttarakhand", "west bengal", "delhi", "puducherry", "chandigarh",
    "jammu", "kashmir", "ladakh",
)
_STATE_WORDS = frozenset(word for state in _INDIAN_STATES_AND_UTS for word in state.split())

# Common spoken-language names -- rejected when a piece is made ENTIRELY of them.
_SPOKEN_LANGUAGE_WORDS = frozenset({
    "english", "hindi", "telugu", "tamil", "kannada", "malayalam", "marathi",
    "bengali", "gujarati", "punjabi", "urdu", "odia", "assamese", "sanskrit",
    "french", "german", "spanish", "chinese", "japanese", "korean", "arabic",
    "russian", "portuguese", "italian", "mandarin", "konkani", "sindhi",
})

# A piece that is EXACTLY one of these bare category-label words means a
# section header leaked through as its own split piece.
_BARE_CATEGORY_LABELS = frozenset({
    "languages", "language", "skills", "skill", "tools", "tool", "frameworks",
    "framework", "databases", "database", "technologies", "technology",
    "libraries", "library", "platforms", "platform", "proficiencies",
})


# ============================================================================
# KNOWN SKILLS
# ============================================================================
#
# This vocabulary is NOT replacing the trained NER model. It is used to
# recover explicit skills from the Technical Skills section and to prevent
# technical skill lists from being mistaken for section headings.
# ============================================================================

_KNOWN_SKILL_TERMS = frozenset({
    # Programming languages / Web
    "python", "java", "c", "c++", "c#", "go", "golang", "ruby", "php",
    "javascript", "java script", "typescript", "kotlin", "swift",
    "html", "html5", "css", "css3", "sql", "r", "bash", "shell scripting",

    # Computer Science / Data / Databases
    "dsa", "oops", "object oriented programming", "object-oriented programming",
    "data structures", "data structures and algorithms", "algorithms", "dbms",
    "computer networks", "operating systems",
    "mysql", "postgresql", "oracle", "mongodb", "sqlite", "redis", "nosql",
    "pandas", "numpy", "matplotlib", "seaborn",
    "scikit-learn", "sklearn", "tensorflow", "keras", "pytorch", "opencv",
    "nlp", "natural language processing", "machine learning", "deep learning",
    "artificial intelligence", "generative ai",
    "bert", "roberta", "cnn", "rnn", "llm",
    "power bi", "tableau", "excel",

    # Frameworks / Libraries / Development
    "react", "react.js", "reactjs", "angular", "vue", "vue.js",
    "node", "node.js", "nodejs", "express", "express.js", "expressjs",
    "flask", "fastapi", "django", "spring", "spring boot",
    "bootstrap", "tailwind", "tailwind css",
    "rest api", "rest apis", "restful api", "jwt",

    # Tools
    "git", "github", "gitlab", "bitbucket", "docker", "kubernetes",
    "linux", "windows", "vs code", "visual studio code",
    "jupyter", "jupyter notebooks", "google colab", "postman",
    "android studio", "sublime", "figma", "microsoft office",

    # Cloud
    "aws", "azure", "gcp", "vercel", "firebase",

    # Software / Engineering
    "api integration", "frontend development", "front-end development",
    "backend development", "back-end development", "web development",
    "ui/ux", "responsive design", "version control",
    "debugging", "testing", "test automation",
    "data analysis", "data analytics", "data visualization", "reporting", "forecasting",

    # Soft skills
    "communication", "communicator", "interpersonal skills", "teamwork",
    "leadership", "adaptability", "problem solving", "problem-solving",
    "collaboration", "time management", "critical thinking", "quick learning",
    "learning agility", "decision making", "decision-making",
    "presentation skills", "organizational skills",
})

_SOFT_SKILL_PHRASES = (
    "interpersonal skills", "problem-solving", "problem solving", "time management",
    "critical thinking", "quick learning", "learning agility", "decision-making",
    "decision making", "presentation skills", "organizational skills",
    "communication", "teamwork", "leadership", "adaptability", "collaboration",
    "debugging",
)


# ============================================================================
# SKILLS SECTION HEADERS
# ============================================================================

# One list of section names, used by header detection, the stop check, and
# _restore_section_breaks, so adding a new heading here works everywhere.
_SKILL_SECTION_NAMES = (
    "technical skills", "technical skill set", "key skills", "core skills",
    "professional skills", "relevant skills", "soft skills", "programming skills",
    "computer skills", "it skills", "software skills", "skills and tools",
    "skills & tools", "skills and technologies", "skills & technologies",
    "tools and technologies", "tools & technologies", "technical proficiency",
    "technical proficiencies", "technical expertise", "technical knowledge",
    "technical summary", "tech stack", "technology stack", "skill set", "skillset",
    "skills", "areas of expertise", "core competencies", "competencies",
    "expertise", "relevant coursework", "coursework", "core subjects",
    "relevant courses", "areas of interest",
)

# Headings that START A DIFFERENT section (the skills scan stops at these).
_OTHER_SECTION_NAMES = (
    "work experience", "professional experience", "employment history",
    "experience", "academic details", "academic background",
    "academic qualifications", "educational qualifications",
    "educational background", "education", "qualifications",
    "academic projects", "personal projects", "key projects", "major projects",
    "mini projects", "projects", "certifications", "certification",
    "certificates", "courses and certifications", "achievements",
    "accomplishments", "awards", "internships", "internship", "declaration",
    "hobbies", "interests", "extra-curricular activities",
    "extracurricular activities", "co-curricular activities", "activities",
    "personal details", "personal information", "personal profile",
    "career objective", "objective", "professional summary", "profile summary",
    "summary", "profile", "about me", "publications", "strengths", "references",
    "workshops", "trainings", "training", "languages known",
    "positions of responsibility", "volunteer experience", "volunteering",
    "contact details", "contact information",
)


def _names_alternation(names) -> str:
    """Regex alternation for a list of section names, longest first, with
    any run of whitespace allowed between words."""
    return "|".join(
        r"\s+".join(re.escape(w) for w in name.split())
        for name in sorted(set(names), key=len, reverse=True)
    )


# ---------------------------------------------------------------------------
# Section-break restoration. Every section scanner in this module works LINE
# BY LINE, which breaks when the caller collapsed all line breaks (the Spring
# backend's text-cleaning step does this) or when PDF extraction glued a
# header onto the previous line. _restore_section_breaks() puts each
# recognisable section header back on its own line BEFORE any scanning.
# ---------------------------------------------------------------------------

_SECTION_HEADER_PHRASES = _SKILL_SECTION_NAMES + _OTHER_SECTION_NAMES
_ALL_SECTION_NAMES = frozenset(n.lower() for n in _SECTION_HEADER_PHRASES)

# Words that, placed directly before "Skills", make it part of a phrase in a
# sentence or soft-skills list ("Communication Skills") rather than a heading.
_SKILLS_MODIFIER_WORDS = frozenset({
    "communication", "interpersonal", "leadership", "analytical",
    "problem-solving", "presentation", "management", "organizational",
    "organisational", "writing", "people", "language", "social",
    "negotiation", "listening", "time", "team", "teamwork", "my", "our", "his",
    "her", "their", "strong", "good", "excellent", "great",
})

_SECTION_HEADER_FINDER = re.compile(
    r"(?<![A-Za-z])(" + _names_alternation(_SECTION_HEADER_PHRASES) + r")(?![A-Za-z])(\s*:)?",
    re.IGNORECASE,
)


def _neighbour_tokens(text: str, start: int, end: int):
    """The whitespace-separated token just before `start` and just after
    `end` (empty string when there is none)."""
    before = text[:start].rstrip().rsplit(None, 1)
    after = text[end:].lstrip().split(None, 1)
    return (before[-1] if before else ""), (after[0] if after else "")


def _looks_like_header(phrase: str, has_colon: bool, prev_tok: str, next_tok: str) -> bool:
    """Decides whether a section-name phrase found in running text is really
    a heading (ALL CAPS, followed by a colon, multi-word Title Case, or
    single-word Title Case at a clear section boundary)."""
    words = phrase.split()
    letters = [c for c in phrase if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        return True
    if has_colon:
        return True
    if not all(w[0].isupper() for w in words):
        return False
    if len(words) > 1:
        return True

    if prev_tok and prev_tok[0].islower():
        return False
    if words[0].lower() == "skills" and prev_tok.lower().strip(",.;:") in _SKILLS_MODIFIER_WORDS:
        return False
    if not next_tok:
        return True
    first = next_tok[0]
    return first.isupper() or first.isdigit() or not first.isalnum()


def _restore_section_breaks(text: str) -> str:
    """Puts recognisable section headings back on their own line. Text whose
    headings are already on their own lines is effectively unchanged."""
    def _replace(m):
        phrase, colon = m.group(1), m.group(2)
        prev_tok, next_tok = _neighbour_tokens(m.string, m.start(), m.end())
        if not _looks_like_header(phrase, bool(colon), prev_tok, next_tok):
            return m.group(0)
        return "\n" + re.sub(r"\s+", " ", phrase) + (":" if colon else "") + "\n"

    return _SECTION_HEADER_FINDER.sub(_replace, text)


_SKILLS_HEADER_PATTERN = re.compile(
    r"^\s*(?:technical\s+|key\s+|core\s+|professional\s+|relevant\s+)?"
    r"(?:skills?\b|skill\s*set\b|areas?\s+of\s+expertise\b|expertise\b|competenc(?:y|ies)\b)"
    r"\s*:?\s*",
    re.IGNORECASE,
)

_INLINE_SKILLS_HEADER_PATTERN = re.compile(
    r"\b(?:"
    r"technical\s+skills?"
    r"|key\s+skills?"
    r"|core\s+skills?"
    r"|professional\s+skills?"
    r"|relevant\s+skills?"
    r"|skills?\s+set"
    r"|skill\s+set"
    r"|skills?\s*&\s*(?:expertise|competencies|technologies)"
    r"|skills?\s+and\s+(?:expertise|competencies|technologies)"
    r"|areas?\s+of\s+expertise"
    r"|technical\s+competenc(?:y|ies)"
    r"|technical\s+proficienc(?:y|ies)"
    r"|technical\s+expertise"
    r"|tools?\s*&\s*technologies"
    r"|core\s+competenc(?:y|ies)"
    r"|core\s+proficienc(?:y|ies)"
    r")\b\s*:?",
    re.IGNORECASE,
)


# ============================================================================
# GENERAL RESUME SECTION STOP PATTERN
# ============================================================================

_SECTION_STOP_PATTERN = re.compile(
    r"^\s*(?:"
    r"declaration|projects?|experience|work\s+experience|education|"
    r"certifications?|references?|achievements?|hobbies|interests?|"
    r"extra[\s-]?curricular|personal\s+(?:details|information)|objective|"
    r"career\s+objective|summary|profile|publications?|awards?|"
    r"internships?|academic\s+(?:details|background)|languages?|"
    r"date\s+of\s+birth|dob|father'?s\s+name|mother'?s\s+name|"
    r"marital\s+status|nationality|gender"
    r")\s*:?\s*$",
    re.IGNORECASE,
)

_NON_SKILL_SECTION_LINE_PATTERN = re.compile(
    r"^\s*(?:"
    r"academic\s+coursework"
    r"|coursework"
    r"|participation(?:\s*&|\s+and)?\s+certifications?"
    r"|certifications?"
    r"|certificates?"
    r"|achievements?"
    r"|experience(?:\s*&|\s+and)?\s+internships?"
    r"|professional\s+experience"
    r"|work\s+experience"
    r"|projects?"
    r"|education"
    r"|declaration"
    r"|references?"
    r"|objective"
    r"|career\s+objective"
    r"|summary"
    r"|profile"
    r"|personal\s+(?:details|information)"
    r"|contact(?:\s+details?)?"
    r")\s*:?.*$",
    re.IGNORECASE,
)


# ============================================================================
# CATEGORY BASED SKILLS
# ============================================================================

_SKILL_CATEGORY_NAMES = (
    r"programming\s+languages?"
    r"|languages?"
    r"|frameworks?\s*(?:&|and)\s*libraries?"
    r"|frameworks?"
    r"|libraries?"
    r"|databases?"
    r"|tools?"
    r"|ui\s+development"
    r"|web\s+technologies?"
    r"|computer\s+science\s+fundamentals?"
    r"|data\s+analytics"
    r"|ai\s*(?:&|and)\s*machine\s+learning"
    r"|machine\s+learning"
    r"|deep\s+learning"
    r"|operating\s+systems?"
    r"|version\s+control"
    r"|cloud"
    r"|platforms?"
    r"|soft\s+skills?"
    r"|technical\s+skills?"
)

_CATEGORY_BOUNDARY_PATTERN = re.compile(
    r"\b(?:" + _SKILL_CATEGORY_NAMES + r")\s*[:\-]\s*",
    re.IGNORECASE,
)

_STANDALONE_SKILL_CATEGORY_PATTERN = re.compile(
    r"^\s*(?:" + _SKILL_CATEGORY_NAMES + r")\s*:?\s*$",
    re.IGNORECASE,
)

_INLINE_STOP_PATTERN = re.compile(
    r"(?:"
    r"participation\s*(?:&|and)\s*certifications?"
    r"|certifications?"
    r"|experience\s*(?:&|and)\s*internships?"
    r"|professional\s+experience"
    r"|work\s+experience"
    r"|projects?"
    r"|education"
    r"|declaration"
    r"|references?"
    r"|career\s+objective"
    r"|objective"
    r"|summary"
    r"|profile"
    r")\b",
    re.IGNORECASE,
)


# ============================================================================
# EXPERIENCE / DATE
# ============================================================================

_DATE_RANGE_PATTERN = re.compile(
    r"(?P<start>"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)[\s,\-./]*\d{4}"
    r"|\b\d{4})"
    r"\s*(?:-|–|—|to|until)\s*"
    r"(?P<end>"
    r"(?:present|current|now)"
    r"|(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)[\s,\-./]*\d{4}"
    r"|\b\d{4})",
    re.IGNORECASE,
)

_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        "January February March April May June July August September October November December".split(),
        1,
    )
}
_MONTH_ALIASES = {**_MONTHS, **{m[:3]: n for m, n in _MONTHS.items()}}

_PROFESSIONAL_EXPERIENCE_HEADER = re.compile(
    r"^\s*(?:"
    r"professional\s+experience"
    r"|work\s+experience"
    r"|professional\s+work\s+experience"
    r"|employment\s+(?:history|experience)"
    r"|work\s+history"
    r")\s*:?\s*[\s-]*$",
    re.IGNORECASE,
)

_MAX_SECTION_LINES = 40


# ============================================================================
# SECTION HEADING DETECTION
# ============================================================================

_BULLET_ITEM_PATTERN = re.compile(
    r"^\s*[\u2022\u00b7\u25aa\u25ab\u25cf\u25cb\u25e6\u25a0\u25ba\u25b6"
    r"\u27a2\u27a4\u2713\u2714\u2023\uf0b7\uf0a7\uf076\uf0d8\uf0fc*\u2013\-]\s*\S"
)

_KNOWN_HEADINGS = frozenset({
    "profile", "summary", "objective", "career objective",
    "education", "academic details", "academic background", "qualifications",
    "experience", "work experience", "professional experience",
    "professional work experience", "employment history", "employment experience",
    "internship", "internships", "projects", "project",
    "certifications", "certification", "certificates",
    "achievements", "achievement", "participation", "references", "declaration",
    "skills", "skill", "skill set", "technical skills", "professional skills",
    "key skills", "core skills", "relevant skills",
    "personal details", "personal information", "extra curricular", "extracurricular",
    "publications", "awards", "hobbies", "interests", "languages", "soft skills",
    "technical competencies", "professional competencies",
    "technical proficiency", "professional proficiency", "areas of expertise",
})


def _is_known_skill_line(line: str) -> bool:
    """True when every word of the line is a known skill term
    (e.g. "PYTHON JAVA DSA HTML CSS")."""
    words = re.sub(r"\s+", " ", line).split()
    normalized_words = [word.strip("()[]{}|/:;-").lower() for word in words]
    return bool(normalized_words) and all(w in _KNOWN_SKILL_TERMS for w in normalized_words)


def _looks_like_section_heading(line: str) -> bool:
    """Detect genuine resume section headings. "PYTHON JAVA DSA HTML CSS" is
    NOT a heading; "PROFESSIONAL EXPERIENCE" or "PROJECTS" is."""
    # A line that starts with a bullet is a list item, never a heading.
    if _BULLET_ITEM_PATTERN.match(line):
        return False

    line = line.strip().strip(" :|-")
    if not line or len(line) > 70:
        return False
    if any(ch in line for ch in ".,;!?"):
        return False

    normalized = re.sub(r"\s+", " ", line).strip()
    lower = normalized.lower()

    if lower in _KNOWN_HEADINGS:
        return True

    words = normalized.split()
    letters = [c for c in normalized if c.isalpha()]

    if words and letters and all(len(word) == 1 or word.isupper() for word in words):
        # Every token is a known skill -> skill content, not a heading.
        if _is_known_skill_line(normalized):
            return False
        # Multi-word ALL-CAPS phrases are normally section headings.
        if len(words) >= 2:
            return True
        # Single ALL-CAPS word: PROJECTS/EDUCATION -> heading, PYTHON -> skill
        if len(letters) >= 6 and lower not in _KNOWN_SKILL_TERMS:
            return True

    return False


def _ends_skill_block(line: str) -> bool:
    """True when `line` starts a different (non-skills) section."""
    if _SECTION_STOP_PATTERN.match(line) or _NON_SKILL_SECTION_LINE_PATTERN.match(line):
        return True
    return _looks_like_section_heading(line) and not _is_known_skill_line(line)


# ============================================================================
# SKILL TEXT AFTER INLINE HEADER
# ============================================================================

def _capture_skill_text_after_match(text: str, match_end: int) -> str:
    """Captures the skills text that follows an inline skills header, up to
    the next section heading, an email address, or the line cap."""
    remainder = text[match_end:]
    captured = []

    for raw_line in remainder.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if _ends_skill_block(line):
            break

        # On a single collapsed line, cut at the first unrelated section name.
        stop = _INLINE_STOP_PATTERN.search(line)
        if stop:
            line = line[:stop.start()].strip()
            if line:
                captured.append(line)
            break

        email_match = _EMAIL_PATTERN.search(line)
        if email_match:
            line = line[:email_match.start()].strip()
            if line:
                captured.append(line)
            break

        captured.append(line)
        if len(captured) >= _MAX_SECTION_LINES:
            break

    return "\n".join(captured).strip()


def _find_skills_sections_by_header(full_text: str) -> list:
    """Find Technical Skills independently from the NER model. Education
    stops at TECHNICAL SKILLS; skills stop at PROJECTS (or any other
    section heading)."""
    sections = []
    lines = full_text.split("\n")

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        match = _SKILLS_HEADER_PATTERN.match(stripped)
        if not match:
            i += 1
            continue

        captured = []

        # Same-line skills: "TECHNICAL SKILLS: Python, Java, SQL"
        trailing = stripped[match.end():].strip()
        if trailing:
            captured.append(trailing)

        j = i + 1
        while j < len(lines) and len(captured) < _MAX_SECTION_LINES:
            line = lines[j].strip()
            if not line:
                j += 1
                continue
            if _ends_skill_block(line):
                break
            # Never consume contact details
            if _EMAIL_PATTERN.search(line):
                break
            captured.append(line)
            j += 1

        if captured:
            sections.append("\n".join(captured))

        i = max(j, i + 1)

    # ---- INLINE FALLBACK ----------------------------------------------------
    if not sections:
        for match in _INLINE_SKILLS_HEADER_PATTERN.finditer(full_text):
            block = _capture_skill_text_after_match(full_text, match.end())
            if block:
                sections.append(block)

    # ---- CATEGORY-BASED FALLBACK -------------------------------------------
    if not sections:
        category_indexes = [
            index for index, line in enumerate(lines)
            if _STANDALONE_SKILL_CATEGORY_PATTERN.match(line.strip())
        ]

        for position, start_index in enumerate(category_indexes):
            end_index = category_indexes[position + 1] if position + 1 < len(category_indexes) else len(lines)
            captured = []
            for line in lines[start_index + 1:end_index]:
                stripped = line.strip()
                if not stripped:
                    continue
                if _ends_skill_block(stripped):
                    break
                if _EMAIL_PATTERN.search(stripped):
                    break
                captured.append(stripped)
            if captured:
                sections.append("\n".join(captured))

    # ---- INLINE CATEGORY FALLBACK ------------------------------------------
    if not sections:
        matches = list(_CATEGORY_BOUNDARY_PATTERN.finditer(full_text))
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(full_text)
            chunk = full_text[start:end]

            stop = _INLINE_STOP_PATTERN.search(chunk)
            if stop:
                chunk = chunk[:stop.start()]

            email_match = _EMAIL_PATTERN.search(chunk)
            if email_match:
                chunk = chunk[:email_match.start()]

            if chunk.strip():
                sections.append(chunk.strip())

    # ---- REMOVE DUPLICATES -------------------------------------------------
    unique_sections = []
    seen = set()
    for section in sections:
        key = re.sub(r"\s+", " ", section).strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique_sections.append(section)

    return unique_sections


# ---------------------------------------------------------------------------
# Regex/heuristic fallbacks for other fields the NER model frequently misses
# (email, college names, graduation years, degree, name).
# ---------------------------------------------------------------------------

_EDUCATION_HEADER_PATTERN = re.compile(
    r"^\s*(?:"
    r"academic\s+(?:details|background|qualifications?)\b"
    r"|education\b"
    r"|qualifications?\b"
    r")\s*:?\s*",
    re.IGNORECASE,
)

_COLLEGE_LINE_PATTERN = re.compile(
    r"[^,\n]*\b(?:university|college|institute|polytechnic|vidyalaya|school)\b[^,\n]*",
    re.IGNORECASE,
)

_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")

# Words that mark an institution as pre-university when they appear on the
# same line as the institution name or on the line right after it.
_PRE_UNIVERSITY_INDICATORS = (
    "intermediate", "secondary school certificate", "matriculation",
    "higher secondary", "10th", "12th", "ssc", "hsc",
)

# Degree/qualification keywords, scoped to the Education section -- used
# INSTEAD OF the raw NER "Degree" entity, whose span is unreliable.
_DEGREE_PATTERN = re.compile(
    r"[^,\n]*\b(?:bachelor|master|b\.?\s?tech|m\.?\s?tech|b\.?\s?e\b|m\.?\s?e\b|"
    r"b\.?\s?sc\b|m\.?\s?sc\b|b\.?\s?a\b|m\.?\s?a\b|b\.?\s?com\b|m\.?\s?com\b|"
    r"mba|mca|bca|ph\.?d|diploma|intermediate|secondary school certificate|"
    r"higher secondary|matriculation)\b[^,\n]*",
    re.IGNORECASE,
)

# Title-like first lines that must NOT be mistaken for a person's name.
_NON_NAME_FIRST_LINES = frozenset({
    "resume", "curriculum vitae", "cv", "bio-data", "biodata", "profile",
})


def _find_education_section(full_text: str) -> str:
    """Header-scan scoped to an EDUCATION section -- bounds where the
    college-name and graduation-year regexes look."""
    lines = full_text.split("\n")

    for i, line in enumerate(lines):
        stripped_header_line = line.strip()
        m = _EDUCATION_HEADER_PATTERN.match(stripped_header_line)
        if m:
            captured = []
            trailing = stripped_header_line[m.end():].strip()
            if trailing:
                captured.append(trailing)

            j = i + 1
            while j < len(lines) and len(captured) < _MAX_SECTION_LINES:
                stripped = lines[j].strip()
                if _SECTION_STOP_PATTERN.match(stripped) or _SKILLS_HEADER_PATTERN.match(stripped):
                    break
                if stripped:
                    captured.append(stripped)
                j += 1
            return "\n".join(captured)

    return ""


_PROJECTS_HEADER_PATTERN = re.compile(r"^\s*projects?\s*:?\s*$", re.IGNORECASE)


def _find_projects_section(full_text: str) -> str:
    """Header-scan scoped to a PROJECTS section -- used to catch project
    titles the NER model mistook for a job "Designation"."""
    lines = full_text.split("\n")

    for i, line in enumerate(lines):
        if _PROJECTS_HEADER_PATTERN.match(line.strip()):
            captured = []
            j = i + 1
            while j < len(lines) and len(captured) < 60:
                stripped = lines[j].strip()
                if _SECTION_STOP_PATTERN.match(stripped) and not _PROJECTS_HEADER_PATTERN.match(stripped):
                    break
                if stripped:
                    captured.append(stripped)
                j += 1
            return "\n".join(captured)

    return ""


def extract_email_fallback(full_text: str) -> list:
    """A plain regex is strictly more reliable than a trained model here."""
    return sorted(set(_EMAIL_PATTERN.findall(full_text)))


def extract_college_names_fallback(full_text: str) -> list:
    education_text = _find_education_section(full_text)
    if not education_text:
        return []

    lines = education_text.split("\n")
    college_names = []
    exclude_phrases = ("certificate", "junior college", "high school", "secondary school")

    for idx, line in enumerate(lines):
        matches = _COLLEGE_LINE_PATTERN.findall(line)
        if not matches:
            continue

        context = line.lower()
        if idx + 1 < len(lines):
            context += " " + lines[idx + 1].lower()

        if any(p in context for p in exclude_phrases) or any(p in context for p in _PRE_UNIVERSITY_INDICATORS):
            continue

        for m in matches:
            cleaned = m.strip(" .-")
            # Strip a trailing year or year-range.
            year_match = _YEAR_PATTERN.search(cleaned)
            if year_match:
                cleaned = cleaned[:year_match.start()].strip(" ,.-\u2013\u2014")
            if cleaned:
                college_names.append(cleaned)

    return sorted(set(college_names))


def extract_graduation_years_fallback(full_text: str) -> list:
    education_text = _find_education_section(full_text)
    if not education_text:
        return []
    return sorted(set(m.group(0) for m in _YEAR_PATTERN.finditer(education_text)))


def extract_degree_fallback(full_text: str) -> list:
    """Extracts degree/qualification lines by keyword, bounded to the
    Education section."""
    education_text = _find_education_section(full_text)
    if not education_text:
        return []

    matches = [match.strip(" .-") for match in _DEGREE_PATTERN.findall(education_text)]
    return sorted(set(match for match in matches if match))


def filter_out_project_titles(designations: list, full_text: str) -> list:
    """Removes any "Designation" that is actually a project title."""
    projects_text = _find_projects_section(full_text)
    if not projects_text:
        return designations
    projects_lower = projects_text.lower()
    return [d for d in designations if d.lower() not in projects_lower]


def filter_out_language_lists(designations: list) -> list:
    """Removes any "Designation" that is actually a spoken-language list
    (e.g. "Telugu English Hindi Marathi")."""
    result = []
    for d in designations:
        words = d.split()
        if words and all(w.lower() in _SPOKEN_LANGUAGE_WORDS for w in words):
            continue
        result.append(d)
    return result


def extract_name_fallback(full_text: str):
    """If the NER model found no Name entity, fall back to the first
    non-empty line, with guards against titles and contact lines."""
    for line in full_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.lower() in _NON_NAME_FIRST_LINES:
            return None
        if "@" in line or any(ch.isdigit() for ch in line):
            return None  # looks like a contact-info line, not a name
        words = line.split()
        if 1 <= len(words) <= 5 and len(line) < 60:
            return line
        return None  # first non-empty line doesn't look name-shaped -- don't guess
    return None


# ============================================================================
# DATE HELPERS
# ============================================================================

def _parse_date_token(token: str, is_end: bool = False):
    token = token.strip().lower()

    if token in {"present", "current", "now"}:
        today = date.today()
        return (today.year, today.month)

    match = re.search(r"(19|20)\d{2}", token)
    if not match:
        return None

    year = int(match.group(0))
    month = 12 if is_end else 1

    for name, number in _MONTH_ALIASES.items():
        if re.search(rf"\b{name}\b", token):
            month = number
            break

    return (year, month)


def _months_between(start, end) -> int:
    if not start or not end:
        return 0
    return max(0, (end[0] - start[0]) * 12 + (end[1] - start[1]) + 1)


def _format_years(years: float) -> str:
    return f"{years:.1f} years" if years % 1 else f"{int(years)} years"


# ============================================================================
# EXPERIENCE FALLBACK
# ============================================================================

_EXPERIENCE_CONTEXT_YEARS = re.compile(
    r"\b(?:professional|work|employment)\s+(?:experience|history)\b|"
    r"\b(?:years?|yrs?)\s+of\s+professional\s+experience\b",
    re.IGNORECASE,
)
_EXPERIENCE_CONTEXT_MONTHS = re.compile(
    r"\b(?:professional|work|employment)\s+(?:experience|history)\b|"
    r"\b(?:months?|mos?)\s+of\s+professional\s+experience\b",
    re.IGNORECASE,
)


def extract_experience_fallback(full_text: str) -> list:
    lines = full_text.split("\n")
    explicit = []

    for match in re.finditer(r"(?<!\w)(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b", full_text, re.I):
        window = full_text[max(0, match.start() - 80):min(len(full_text), match.end() + 80)]
        if _EXPERIENCE_CONTEXT_YEARS.search(window):
            explicit.append(float(match.group(1)))

    for match in re.finditer(r"(?<!\w)(\d+(?:\.\d+)?)\s*(?:months?|mos?)\b", full_text, re.I):
        window = full_text[max(0, match.start() - 80):min(len(full_text), match.end() + 80)]
        if _EXPERIENCE_CONTEXT_MONTHS.search(window):
            explicit.append(float(match.group(1)) / 12)

    if explicit:
        return [_format_years(max(explicit))]

    blocks = []
    for i, line in enumerate(lines):
        if not _PROFESSIONAL_EXPERIENCE_HEADER.match(line.strip()):
            continue

        block_lines = []
        j = i + 1
        while j < len(lines) and len(block_lines) < 80:
            current = lines[j].strip()
            if _looks_like_section_heading(current) or _SECTION_STOP_PATTERN.match(current):
                break
            if current:
                block_lines.append(current)
            j += 1

        if block_lines:
            blocks.append("\n".join(block_lines))

    if not blocks:
        return []

    intervals = []
    for block in blocks:
        for match in _DATE_RANGE_PATTERN.finditer(block):
            start = _parse_date_token(match.group("start"))
            end = _parse_date_token(match.group("end"), True)
            if start and end and end >= start:
                intervals.append((start, end))

    if not intervals:
        return []

    intervals.sort()
    merged = []
    for start, end in intervals:
        if not merged:
            merged.append([start, end])
            continue
        previous_end = merged[-1][1]
        previous_end_month = previous_end[0] * 12 + previous_end[1]
        current_start_month = start[0] * 12 + start[1]
        if current_start_month <= previous_end_month + 1:
            if end > previous_end:
                merged[-1][1] = end
        else:
            merged.append([start, end])

    months = sum(_months_between(start, end) for start, end in merged)
    return [_format_years(months / 12)]


# ============================================================================
# SKILL NORMALIZATION
# ============================================================================

_CANONICAL_SKILL_NAMES = {
    "bootstrap": "Bootstrap", "cnn": "CNN", "css": "CSS", "css3": "CSS3",
    "dsa": "DSA", "django": "Django", "express": "Express.js",
    "express.js": "Express.js", "expressjs": "Express.js", "figma": "Figma",
    "flask": "Flask", "fastapi": "FastAPI", "github": "GitHub",
    "google colab": "Google Colab", "html": "HTML", "html5": "HTML5",
    "java": "Java", "javascript": "JavaScript", "java script": "JavaScript",
    "jupyter": "Jupyter", "jupyter notebooks": "Jupyter Notebooks",
    "keras": "Keras", "linux": "Linux", "matplotlib": "Matplotlib",
    "mongodb": "MongoDB", "mysql": "MySQL", "node": "Node.js",
    "node.js": "Node.js", "nodejs": "Node.js", "numpy": "NumPy",
    "opencv": "OpenCV", "operating systems": "Operating Systems",
    "pandas": "Pandas", "python": "Python", "react": "React",
    "react.js": "React.js", "reactjs": "React.js", "scikit-learn": "scikit-learn",
    "sql": "SQL", "sublime": "Sublime", "tensorflow": "TensorFlow",
    "vs code": "VS Code", "visual studio code": "VS Code", "windows": "Windows",
    "machine learning": "Machine Learning", "deep learning": "Deep Learning",
    "natural language processing": "Natural Language Processing", "nlp": "NLP",
    "aws": "AWS", "azure": "Azure", "gcp": "GCP", "git": "Git",
    "docker": "Docker", "kubernetes": "Kubernetes",
    "communication": "Communication", "teamwork": "Teamwork",
    "leadership": "Leadership", "adaptability": "Adaptability",
    "collaboration": "Collaboration", "debugging": "Debugging",
    "testing": "Testing", "test automation": "Test Automation",
    "rest api": "REST API", "rest apis": "REST APIs", "restful api": "REST API",
    "problem solving": "problem-solving",
    "object-oriented programming": "Object-Oriented Programming",
    "object oriented programming": "Object-Oriented Programming",
}


def _normalize_skill_token(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value)).strip(" .:-–—")
    value = value.replace("problem -solving", "problem-solving")
    return _CANONICAL_SKILL_NAMES.get(value.lower(), value)


# ============================================================================
# RECOVER KNOWN SKILLS
# ============================================================================

_RECOVERY_ALIASES = {
    "java script": "JavaScript", "reactjs": "React.js", "react": "React",
    "nodejs": "Node.js", "express": "Express.js", "expressjs": "Express.js",
    "html5": "HTML5", "css3": "CSS3", "sklearn": "scikit-learn",
    "jupyter": "Jupyter", "visual studio code": "VS Code",
    "rest api": "REST API", "rest apis": "REST APIs", "restful api": "REST API",
    "problem solving": "problem-solving",
    "object-oriented programming": "Object-Oriented Programming",
    "object oriented programming": "Object-Oriented Programming",
}

_RECOVERY_CASING = {
    "python": "Python", "java": "Java", "javascript": "JavaScript", "c": "C",
    "c++": "C++", "c#": "C#", "html": "HTML", "css": "CSS", "sql": "SQL",
    "git": "Git", "github": "GitHub", "mysql": "MySQL", "mongodb": "MongoDB",
    "pandas": "Pandas", "numpy": "NumPy", "matplotlib": "Matplotlib",
    "tensorflow": "TensorFlow", "keras": "Keras", "pytorch": "PyTorch",
    "flask": "Flask", "fastapi": "FastAPI", "django": "Django",
    "angular": "Angular", "typescript": "TypeScript", "power bi": "Power BI",
    "microsoft office": "Microsoft Office", "linux": "Linux",
    "windows": "Windows", "docker": "Docker", "kubernetes": "Kubernetes",
    "communication": "Communication", "communicator": "Communicator",
    "teamwork": "Teamwork", "leadership": "Leadership",
    "adaptability": "Adaptability", "collaboration": "Collaboration",
    "debugging": "Debugging", "testing": "Testing",
    "test automation": "Test Automation", "dsa": "DSA", "figma": "Figma",
    "vs code": "VS Code",
}

# Pre-compiled matchers for every known term, longest first. The trailing
# check allows a full stop that ENDS a sentence ("...using MySQL and
# Python.") but blocks names that continue after a dot ("node" must not
# match inside "node.js").
_KNOWN_SKILL_MATCHERS = [
    (
        term,
        re.compile(
            rf"(?<![A-Za-z0-9+#.]){re.escape(term)}(?![A-Za-z0-9+#]|\.[A-Za-z0-9])",
            re.IGNORECASE,
        ),
    )
    for term in sorted(_KNOWN_SKILL_TERMS, key=len, reverse=True)
]

_STRONG_UPPERCASE_TOKENS = frozenset({
    "DBMS", "SQL", "OOPS", "HTML", "CSS", "API", "AWS", "GCP", "DSA",
})


def _recover_known_skills_from_text(text: str) -> set:
    found = set()
    normalized = re.sub(r"\s+", " ", text).strip()
    lower = normalized.lower()

    for term, pattern in _KNOWN_SKILL_MATCHERS:
        if pattern.search(lower):
            # Longest match wins: blank out the matched text so a shorter
            # term cannot match again inside it ("spring" inside "spring boot").
            lower = pattern.sub(lambda m: " " * len(m.group(0)), lower)
            canonical = _RECOVERY_ALIASES.get(term, term)
            found.add(_RECOVERY_CASING.get(canonical.lower(), canonical))

    # .js / .py / .net technical names
    for match in re.finditer(r"\b[A-Za-z][A-Za-z0-9]*(?:\.js|\.py|\.net)\b", normalized):
        token = match.group(0)
        if len(token) <= 35:
            found.add(token)

    # Strong uppercase technical tokens
    for match in re.finditer(r"\b[A-Z]{2,}(?:\d+)?\b", normalized):
        token = match.group(0)
        if token in _STRONG_UPPERCASE_TOKENS:
            found.add(token)

    return found


# ============================================================================
# INDIVIDUAL SKILLS
# ============================================================================

_SENTENCE_WORD_PATTERN = re.compile(
    r"\b(?:"
    r"and|the|with|using|to|for|from|"
    r"that|which|this|achieved|developed|"
    r"deployed|implemented|passionate|"
    r"experience|responsibilities|responsibility|"
    r"candidate|applicants?|tasks?|project|"
    r"screen(?:ing)?|matching|collaboration|"
    r"integration|completion|fostering|"
    r"impactful|efficiency|results?"
    r")\b",
    re.IGNORECASE,
)


def extract_individual_skills(raw_skill_blocks: list, known_locations: list = None,
                              from_skills_section: bool = False) -> list:
    """Splits raw skill blocks into a clean, deduplicated list of individual
    skills. Text from a real skills HEADING (from_skills_section=True) is
    trusted: any item that survives the junk filters is kept even if it is
    not in _KNOWN_SKILL_TERMS. Text only the NER model guessed is not."""
    individual_skills = set()

    location_words = set(_STATE_WORDS)
    for location in (known_locations or []):
        location_words.update(word.lower() for word in re.findall(r"[A-Za-z]+", location))

    for block in raw_skill_blocks:
        if not block:
            continue

        truncated_lines = []
        for line in block.split("\n"):
            stripped = line.strip()
            if _SECTION_STOP_PATTERN.match(stripped) or _NON_SKILL_SECTION_LINE_PATTERN.match(stripped):
                break
            if _EMAIL_PATTERN.search(stripped):
                stripped = _EMAIL_PATTERN.sub(" ", stripped)
            truncated_lines.append(stripped)
        block = "\n".join(truncated_lines)

        # Recover known skills from the entire block
        individual_skills.update(_recover_known_skills_from_text(block))

        # Process list-shaped content
        block = _CATEGORY_BOUNDARY_PATTERN.sub(lambda match: "\n" + match.group(0), block)

        for piece in _SPLIT_PATTERN.split(block):
            piece = _normalize_skill_token(piece)
            piece = _BULLET_PATTERN.sub("", piece)
            piece = _COLON_LABEL_PATTERN.sub("", piece)
            piece = _HYPHEN_LABEL_PATTERN.sub("", piece)
            piece = _PAREN_SUFFIX_PATTERN.sub("", piece)
            piece = _normalize_skill_token(piece)

            if not piece or len(piece) >= 40:
                continue
            if _EMAIL_PATTERN.search(piece):
                continue

            words = piece.split()

            # PDF artifact: "G i t" -> "Git"
            if len(words) >= 2 and all(len(word) == 1 for word in words):
                piece = "".join(words)
                words = [piece]

            if len(words) > _MAX_WORDS:
                continue
            if words and words[0].lower() in _LEADING_REJECT_WORDS:
                continue

            piece_lower = piece.lower()
            if (piece_lower in _NONSKILL_WHOLE_WORDS
                    or piece_lower in _BARE_CATEGORY_LABELS
                    or piece_lower in _ALL_SECTION_NAMES):
                continue
            if words and all(word.lower() in _SPOKEN_LANGUAGE_WORDS for word in words):
                continue
            if words and any(word.lower() in location_words for word in words):
                continue
            if _SENTENCE_WORD_PATTERN.search(piece):
                continue
            if re.search(r"\b(?:by|in|on)\s+(?:20\d{2}|19\d{2})\b", piece, re.I) or re.search(r"\d+%", piece):
                continue

            if re.search(r"(?:\.js|\.py|\.net)$", piece, re.I) or re.fullmatch(r"[A-Z]{2,}(?:\d+)?", piece):
                individual_skills.add(piece)
            elif from_skills_section:
                individual_skills.add(piece)

    return sorted(individual_skills)


# ============================================================================
# WHITESPACE
# ============================================================================

def _normalize_whitespace(text: str) -> str:
    """Collapses runs of horizontal whitespace while preserving newlines."""
    return "\n".join(re.sub(r"[ \t]+", " ", line) for line in text.split("\n"))


# ============================================================================
# NAME / DESIGNATION FILTERS
# ============================================================================

_BAD_NAME_VALUES = frozenset(_BARE_CATEGORY_LABELS) | {
    "academic coursework", "coursework", "salesforce certified", "certifications",
    "experience", "projects", "education", "languages", "soft skills", "skill set",
    "technical skills", "professional skills", "key skills", "core skills",
    "technical competencies", "professional competencies",
    "technical proficiency", "professional proficiency", "areas of expertise",
}


def filter_false_name_values(names: list) -> list:
    result = []
    for name in names:
        normalized = re.sub(r"\s+", " ", str(name)).strip()
        if not normalized:
            continue
        if normalized.lower() in _BAD_NAME_VALUES:
            continue
        if _EMAIL_PATTERN.search(normalized):
            continue
        if len(normalized) > 60 or len(normalized.split()) > 6:
            continue
        result.append(normalized)
    return sorted(set(result))


_BAD_DESIGNATION_VALUES = _BARE_CATEGORY_LABELS | {
    "academic coursework", "coursework", "salesforce certified",
}


def filter_false_designations(designations: list) -> list:
    result = []
    for value in designations:
        designation = re.sub(r"\s+", " ", str(value)).strip()
        designation_lower = designation.lower()

        if not designation:
            continue
        if _EMAIL_PATTERN.search(designation):
            continue
        if designation_lower in _BAD_DESIGNATION_VALUES:
            continue
        if len(designation.split()) > 8:
            continue
        if any(word in designation_lower for word in (
                "technical skills", "soft skills", "programming languages", "academic coursework")):
            continue

        # Drop designations made up entirely of skill names.
        residue = designation_lower
        for term in sorted(_KNOWN_SKILL_TERMS, key=len, reverse=True):
            residue = re.sub(
                rf"(?<![a-z0-9+#.]){re.escape(term)}(?![a-z0-9+#.])",
                " ",
                residue,
            )
        residue = re.sub(r"[ ,/|&+.-]+", " ", residue).strip()
        if not residue:
            continue

        result.append(designation)

    return sorted(set(result))


# ============================================================================
# SANITIZE RAW SKILLS
# ============================================================================

def sanitize_skill_raw_blocks(raw_blocks: list, known_locations: list = None) -> list:
    result = []
    seen = set()
    for block in (raw_blocks or []):
        skills = extract_individual_skills([block], known_locations=known_locations)
        if skills:
            joined = "\n".join(skills)
            key = joined.lower()
            if key not in seen:
                seen.add(key)
                result.append(joined)
    return result


# ============================================================================
# MAIN RESUME PARSER
# ============================================================================

# Terms skipped by the "no skills heading" whole-text scan: soft skills and
# vocabulary words that are also everyday English or initials.
_NO_HEADING_SCAN_EXCLUDED = frozenset(
    {p.lower() for p in _SOFT_SKILL_PHRASES}
    | {
        "c", "r", "go", "spring", "express", "swift", "react", "node",
        "oracle", "excel", "windows", "algorithms", "reporting", "testing",
        "forecasting", "communicator", "sublime", "vercel", "ruby",
        "interpersonal skills", "presentation skills", "organizational skills",
        "web development",
    }
)


def parse_resume(text: str) -> dict:
    # Normalize PDF whitespace while preserving line boundaries.
    text = _normalize_whitespace(text or "")

    # Run trained spaCy model on the text exactly as before.
    doc = nlp(text)

    # Put section headings back on their own lines for the rule-based
    # scanners below (runs AFTER the NER model on purpose).
    text = _restore_section_breaks(text)

    raw_profile = {field: [] for field in _LABEL_TO_FIELD.values()}
    for label, field in _LABEL_TO_FIELD.items():
        raw_profile[field] = sorted({ent.text.strip() for ent in doc.ents if ent.label_ == label})

    # =========================================================================
    # SKILLS
    # =========================================================================
    header_scanned_sections = _find_skills_sections_by_header(text)

    if header_scanned_sections:
        final_skills = extract_individual_skills(
            header_scanned_sections,
            known_locations=raw_profile["location"],
            from_skills_section=True,
        )
    else:
        final_skills = extract_individual_skills(
            raw_profile["skills_raw"],
            known_locations=raw_profile["location"],
        )

    # Canonicalize and deduplicate
    skill_map = {}
    for skill in final_skills:
        skill = _normalize_skill_token(skill)
        if skill:
            skill_map.setdefault(skill.lower(), skill)

    # Explicit CNN recovery
    if re.search(r"(?<![A-Za-z0-9])cnn(?![A-Za-z0-9])", text, re.I):
        skill_map.setdefault("cnn", "CNN")

    if header_scanned_sections:
        # Recover explicit known skills from the Technical Skills section
        # again, so a line such as "Python Java DSA HTML CSS" is never lost.
        for skill in _recover_known_skills_from_text("\n".join(header_scanned_sections)):
            normalized_skill = _normalize_skill_token(skill)
            if normalized_skill:
                skill_map.setdefault(normalized_skill.lower(), normalized_skill)
    else:
        # No skills heading anywhere: scan the whole text for known technical
        # skills, excluding soft skills and ambiguous everyday words.
        for skill in _recover_known_skills_from_text(text):
            if skill.lower() in _NO_HEADING_SCAN_EXCLUDED:
                continue
            normalized_skill = _normalize_skill_token(skill)
            if normalized_skill:
                skill_map.setdefault(normalized_skill.lower(), normalized_skill)

    final_skills = sorted(skill_map.values(), key=str.lower)
    raw_profile["skills_raw"] = final_skills
    raw_profile["skills_cleaned"] = final_skills

    # =========================================================================
    # EXPERIENCE
    # =========================================================================
    experience_fallback = extract_experience_fallback(text)
    if experience_fallback:
        raw_profile["years_of_experience"] = experience_fallback

    # =========================================================================
    # EMAIL / COLLEGE / GRADUATION YEAR / NAME fallbacks
    # =========================================================================
    if not raw_profile["email_address"]:
        raw_profile["email_address"] = extract_email_fallback(text)

    college_fallback = extract_college_names_fallback(text)
    raw_profile["college_name"] = sorted(set(raw_profile["college_name"]) | set(college_fallback))

    year_fallback = extract_graduation_years_fallback(text)
    raw_profile["graduation_year"] = sorted(set(raw_profile["graduation_year"]) | set(year_fallback))

    if not raw_profile["name"]:
        name_fallback = extract_name_fallback(text)
        if name_fallback:
            raw_profile["name"] = [name_fallback]

    # =========================================================================
    # DEGREE -- replace the raw NER span entirely (it can swallow the whole
    # Technical Skills section) with the education-bounded extractor.
    # =========================================================================
    raw_profile["degree"] = extract_degree_fallback(text)

    # =========================================================================
    # DESIGNATION / NAME CLEANUP
    # =========================================================================
    raw_profile["designation"] = filter_out_project_titles(raw_profile["designation"], text)
    raw_profile["designation"] = filter_out_language_lists(raw_profile["designation"])
    raw_profile["designation"] = filter_false_designations(raw_profile["designation"])

    raw_profile["name"] = filter_false_name_values(raw_profile["name"])

    return raw_profile


def build_resume_dict_from_text(resume_text: str, resume_id: str = None,
                                seniority: str = None, industry: str = None) -> dict:
    """Bridges raw resume text into the shape /match-score, /skill-gap and
    /rank-candidates expect. seniority/industry can't be extracted by the
    current NER model, so they stay None unless supplied by the caller."""
    profile = parse_resume(resume_text)

    skills = profile["skills_cleaned"]
    skills_str = "[" + " ".join(f"'{skill}'" for skill in skills) + "]"
    role = profile["designation"][0] if profile["designation"] else ""

    years_experience = None
    if profile.get("years_of_experience"):
        match = re.search(r"(\d+(?:\.\d+)?)", profile["years_of_experience"][0])
        if match:
            years_experience = float(match.group(1))
            if years_experience.is_integer():
                years_experience = int(years_experience)

    return {
        "resume_id": resume_id or "PARSED_FROM_TEXT",
        "role": role,
        "seniority": seniority,
        "industry": industry,
        "skills": skills_str,
        # Raw resume text keeps the matching engine's text-similarity
        # feature meaningful.
        "summary": resume_text,
        "experience_bullets": "",
        "years_experience": years_experience,
    }
