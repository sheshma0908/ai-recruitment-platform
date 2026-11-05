"""
Job search for the candidate chatbot.

Answers questions like:
    "jobs for 2-3 years experience"      -> experience-range match
    "I have 3 years of experience"       -> jobs whose range includes 3
    "fresher jobs" / "experienced jobs"  -> 0-year jobs / jobs needing 1+ years
    "python jobs", "react"               -> skill match (required + preferred skills)
    "jobs in Hyderabad", "pune"          -> city match
    "internship", "full time jobs"       -> employment type match
    "python jobs in hyderabad for freshers" -> all filters combined
    "details of java full stack developer"  -> full details of one job

Everything comes ONLY from the jobs the backend sends (job_context text or the
structured `jobs` list). Nothing is invented. No ML model is needed here, so job
answers work even if the career model is not trained.

Accepted job data (any of these):
  1. Structured list (recommended) - ChatRequest.jobs, e.g.
     [{"title": "Java Full Stack Developer", "department": "Engineering",
       "location": "Hyderabad", "experience": "2 - 5 Years", "employment_type": "Full-Time",
       "required_skills": ["Java", "Spring Boot"], "preferred_skills": ["Docker"], ...}]
     camelCase keys from Java (jobTitle, requiredSkills, minExperience, ...) also work.
  2. job_context as a JSON string of that same list.
  3. job_context as text, one job per line, segments separated by "|":
       Java Full Stack Developer | Engineering | Hyderabad | 2 - 5 Years | Full-Time |
       Skills: Java, Spring Boot, React, PostgreSQL | Preferred: Docker, Git
     Labelled segments ("Skills: ...", "Location: ...") are read exactly; unlabelled
     ones are recognised automatically (experience, city, job type, title, skills).
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ============================================================
# VOCABULARY
# ============================================================

# Canonical city -> spellings people use. Cities found in the job data are added automatically.
CITY_ALIASES = {
    "hyderabad": ["hyderabad", "hyd", "secunderabad", "hitech city", "gachibowli"],
    "bangalore": ["bangalore", "bengaluru", "blr", "banglore"],
    "pune": ["pune"],
    "mumbai": ["mumbai", "bombay", "navi mumbai"],
    "chennai": ["chennai", "madras"],
    "delhi": ["delhi", "new delhi", "ncr", "delhi ncr"],
    "gurgaon": ["gurgaon", "gurugram"],
    "noida": ["noida", "greater noida"],
    "kolkata": ["kolkata", "calcutta"],
    "ahmedabad": ["ahmedabad"],
    "kochi": ["kochi", "cochin"],
    "coimbatore": ["coimbatore"],
    "jaipur": ["jaipur"],
    "chandigarh": ["chandigarh", "mohali"],
    "indore": ["indore"],
    "visakhapatnam": ["visakhapatnam", "vizag"],
    "vijayawada": ["vijayawada"],
    "warangal": ["warangal"],
    "trivandrum": ["trivandrum", "thiruvananthapuram"],
    "mysore": ["mysore", "mysuru"],
    "nagpur": ["nagpur"],
    "lucknow": ["lucknow"],
    "bhubaneswar": ["bhubaneswar"],
    "remote": ["remote", "work from home", "wfh"],
}

# Employment type canonical -> patterns
JOB_TYPES = {
    "full-time": r"full[\s-]?time|permanent",
    "part-time": r"part[\s-]?time",
    "internship": r"intern(ship)?s?",
    "contract": r"contract(ual)?|freelance",
}

# Rewrites applied to both job data and questions so spellings line up.
SKILL_REWRITES = [
    (r"\bmachine learning\b", "ml"), (r"\bartificial intelligence\b", "ai"),
    (r"\bai\s*/\s*ml\b|\baiml\b|\bai ml\b|\bai-ml\b", "ai ml"),
    (r"\bdeep learning\b", "dl"), (r"\bnatural language processing\b", "nlp"),
    (r"\bspring ?boot\b", "springboot"), (r"\breact\.?js\b", "react"), (r"\bnode\.?js\b", "nodejs"),
    (r"\bnext\.?js\b", "nextjs"), (r"\bvue\.?js\b", "vue"), (r"\bangular\.?js\b", "angular"),
    (r"\bpostgres(ql)?\b", "postgresql"), (r"\bmongo ?db\b", "mongodb"), (r"\bmy ?sql\b", "mysql"),
    (r"\bjs\b", "javascript"), (r"\bts\b", "typescript"), (r"\bgolang\b", "go"), (r"\bk8s\b", "kubernetes"),
    (r"\bfull[\s-]?stack\b", "fullstack"), (r"\bfront[\s-]?end\b", "frontend"), (r"\bback[\s-]?end\b", "backend"),
    (r"\brest ?apis?\b|\brestful\b", "restapi"), (r"\bdev ?ops\b", "devops"), (r"\bpower ?bi\b", "powerbi"),
    (r"\bui\s*/\s*ux\b|\bux\s*/\s*ui\b", "ui ux"), (r"\bc\+\+", "cpp"), (r"\bc#", "csharp"), (r"\.net\b", "dotnet"),
    (r"\bdata scientists?\b", "data science"), (r"\bdata analytics\b", "data analyst"),
]

# Words that carry no search meaning in a job question.
STOPWORDS = set("""
a an the any some all me my i im i'm we you your us our is are am was be been being do does did
can could would will shall should may might please pls plz kindly want wanted need needed looking look
find search show list give tell get see display share provide suggest recommend let know there here
for in at on of to from with by about as or and also only just like related relevant based matching match
what which who where when how why whether job jobs opening openings vacancy vacancies position positions
role roles post posts opportunity opportunities career careers work hiring hire available current currently
open new latest now today company companies have has having got experience experienced exp yrs yr year years
yoe fresher freshers fresh lead lateral graduate graduates entry level beginner beginners senior junior mid skill skills
city cities location locations located place area near around based skilled knowledge know knowing good best
top other more less than above below under over least most minimum maximum min max atleast upto up range
between same type types mode thing things stuff something anything everything one ones that this these those
it its them they then than so too very really ok okay hi hello hey thanks thank pl dear sir madam
details detail info information describe description requirements requirement responsibilities
many much count number total overall altogether currently right
apply applying applied eligible eligibility suitable suit suits fit fits qualify qualified so
interested interest familiar expert expertise proficient background worked working learned learnt
basic strong done based having got which other others another else
""".split())

NUMBER_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
                "eleven": "11", "twelve": "12", "fifteen": "15", "twenty": "20"}

EXPLICIT_JOB_WORDS = re.compile(
    r"\b(jobs?|openings?|vacanc(y|ies)|positions?|posts?|hiring|opportunit(y|ies)|"
    r"internships?|openings|roles? (available|open)|any (role|roles))\b", re.I)

# Career-guidance phrasing -> leave it to the career model unless a filter makes it a search.
CAREER_CUES = re.compile(
    r"\b(salary|salaries|earn|earning|pay|package|become|becoming|career path|career growth|growth|"
    r"roadmap|learn|learning|study|course|courses|certification|future|scope|what does|"
    r"what is a|what is an|day to day|how (do|can) i (start|get into)|get started|tips|prepare)\b", re.I)

DETAIL_CUES = re.compile(
    r"\b(details?|more (info|information|about)|describe|description|tell me (more )?about|"
    r"responsibilit(y|ies)|requirements?|eligibility|qualification|education|openings count|"
    r"closing date|last date|deadline|what (are|is) the (skills|requirements))\b", re.I)

COUNT_CUES = re.compile(r"\b(how many|number of|no\.? of|count of|total|how much jobs)\b", re.I)

# "other than X", "except X", "not in X", "outside X", "excluding X", "non-X"
NEGATION = re.compile(
    r"\b(other than|except(?: for)?|excluding|exclude|apart from|besides|not in|not at|not from|"
    r"outside(?: of)?|but not|without|instead of|leaving|non)[\s-]+", re.I)
OTHER_CITIES = re.compile(
    r"\b(other|another|different|remaining|rest of the|any other)\s+(cities|city|locations?|places?|areas?)\b|"
    r"\belsewhere\b|\boutside\s*(?:\?|$)", re.I)

FOLLOW_UP_CUES = re.compile(r"^(and|what about|how about|any|only|also|in|at|with|for|then|ok|okay)\b", re.I)


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class Job:
    title: str = ""
    department: str = ""
    location: str = ""
    experience_text: str = ""
    exp_min: Optional[float] = None
    exp_max: Optional[float] = None     # None = open ended ("3+ years")
    employment_type: str = ""
    work_mode: str = ""
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    education: str = ""
    openings: str = ""
    salary: str = ""
    closing_date: str = ""
    description: str = ""
    responsibilities: str = ""
    job_id: str = ""
    status: str = ""
    raw: str = ""

    # filled by _index()
    cities: set = field(default_factory=set)
    tokens: set = field(default_factory=set)
    title_tokens: set = field(default_factory=set)
    type_key: str = ""


@dataclass
class JobQuery:
    exp_min: Optional[float] = None
    exp_max: Optional[float] = None
    exp_exact: Optional[float] = None
    exp_label: str = ""          # human readable, e.g. "2-3 years"
    fresher: bool = False
    experienced: bool = False
    cities: set = field(default_factory=set)
    job_type: str = ""
    keywords: List[str] = field(default_factory=list)
    explicit_job_word: bool = False
    detail_request: bool = False
    exclude_cities: set = field(default_factory=set)      # "other than Hyderabad"
    exclude_keywords: List[str] = field(default_factory=list)   # "jobs except java"
    other_cities: bool = False                             # "jobs in other cities" (no city named)

    def has_filters(self) -> bool:
        return bool(self.has_experience() or self.cities or self.job_type or self.keywords
                    or self.exclude_cities or self.exclude_keywords or self.other_cities)

    def has_experience(self) -> bool:
        return any(v is not None for v in (self.exp_min, self.exp_max, self.exp_exact)) \
            or self.fresher or self.experienced


# ============================================================
# TEXT HELPERS
# ============================================================

def _norm(text: str) -> str:
    t = str(text or "").lower()
    t = t.replace("_", " ").replace("–", "-").replace("—", "-").replace("’", "'")
    for pat, rep in SKILL_REWRITES:
        t = re.sub(pat, rep, t)
    return t


def _tokens(text: str) -> List[str]:
    # node.js / .net / react.js are already rewritten by SKILL_REWRITES, so a dot can end a word
    return re.findall(r"[a-z0-9+#]+", _norm(text))


def _split_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = []
        for v in value:
            if isinstance(v, dict):
                v = v.get("name") or v.get("skill") or v.get("skillName") or ""
            items.extend(_split_list(v))
        return items
    text = str(value).strip().strip("[]")
    parts = re.split(r"[,;/\n]|'\s+'|\"\s+\"", text)
    return [p.strip(" '\"") for p in parts if p.strip(" '\"")]


def _fmt_num(n: float) -> str:
    return str(int(n)) if float(n).is_integer() else str(n)


def _words_to_numbers(text: str) -> str:
    return re.sub(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", lambda m: NUMBER_WORDS[m.group(1)], text)


# ============================================================
# PARSING JOB DATA
# ============================================================

def parse_experience(text: str) -> Tuple[Optional[float], Optional[float]]:
    """'2 - 5 Years' -> (2, 5); '3+ yrs' -> (3, None); 'Fresher' -> (0, 1); '4 years' -> (4, 4)."""
    t = _words_to_numbers(_norm(text))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)", t)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        return min(a, b), max(a, b)
    m = re.search(r"(\d+(?:\.\d+)?)\s*\+", t) or re.search(r"(?:min(?:imum)?|at ?least|above|over)\s*(\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(1)), None
    m = re.search(r"(\d+(?:\.\d+)?)", t)
    if m:
        n = float(m.group(1))
        return n, n
    if re.search(r"fresher|entry|no experience|graduate", t):
        return 0.0, 1.0
    return None, None


_EXP_SEGMENT = re.compile(r"\d+(\.\d+)?\s*(-|to|\+)?\s*(\d+(\.\d+)?)?\s*\+?\s*(yrs?|years?|yoe)\b|\bfresher", re.I)

# Map normalised key -> Job attribute
_KEY_MAP = {
    "title": "title", "jobtitle": "title", "job": "title", "role": "title", "position": "title",
    "designation": "title", "name": "title", "jobname": "title",
    "department": "department", "dept": "department", "category": "department", "team": "department",
    "location": "location", "city": "location", "joblocation": "location", "place": "location",
    "experience": "experience_text", "exp": "experience_text", "experiencerange": "experience_text",
    "experiencerequired": "experience_text", "yearsofexperience": "experience_text",
    "employmenttype": "employment_type", "jobtype": "employment_type", "type": "employment_type",
    "workmode": "work_mode", "worktype": "work_mode", "mode": "work_mode",
    "skills": "required_skills", "requiredskills": "required_skills", "musthaveskills": "required_skills",
    "mandatoryskills": "required_skills", "keyskills": "required_skills", "skillsrequired": "required_skills",
    "preferred": "preferred_skills", "preferredskills": "preferred_skills", "nicetohave": "preferred_skills",
    "nicetohaveskills": "preferred_skills", "goodtohave": "preferred_skills",
    "education": "education", "qualification": "education", "educationrequired": "education",
    "openings": "openings", "vacancies": "openings", "noofopenings": "openings", "numberofopenings": "openings",
    "salary": "salary", "salaryrange": "salary", "ctc": "salary", "package": "salary",
    "closingdate": "closing_date", "deadline": "closing_date", "lastdate": "closing_date",
    "applicationdeadline": "closing_date",
    "description": "description", "about": "description", "abouttherole": "description",
    "jobdescription": "description", "summary": "description",
    "responsibilities": "responsibilities", "responsibility": "responsibilities", "duties": "responsibilities",
    "id": "job_id", "jobid": "job_id",
    "status": "status", "jobstatus": "status",
}


def _keynorm(k: str) -> str:
    return re.sub(r"[^a-z]", "", str(k).lower())


def _flat_value(v):
    """{"name": "Engineering"} -> "Engineering"; {"city": "Pune", "state": ..} -> "Pune"."""
    if isinstance(v, dict):
        for key in ("name", "title", "city", "label", "value", "displayName", "skillName"):
            if v.get(key):
                return v[key]
        return ", ".join(str(x) for x in v.values() if isinstance(x, (str, int, float)))
    return v


def _num(v) -> Optional[float]:
    m = re.search(r"\d+(\.\d+)?", str(v))
    return float(m.group()) if m else None


def _job_from_dict(d: dict) -> Job:
    if isinstance(d.get("job"), dict):           # {"job": {...}} wrapper
        d = {**d, **d["job"]}
    job = Job(raw=json.dumps(d, default=str)[:500])
    exp_min = exp_max = None
    for k, v in d.items():
        nk = _keynorm(k)
        if not isinstance(v, (list, tuple)):
            v = _flat_value(v)
        if v is None or v == "":
            continue
        if nk in ("minexperience", "experiencemin", "minexp", "expmin", "minimumexperience", "experiencefrom"):
            exp_min = _num(v); continue
        if nk in ("maxexperience", "experiencemax", "maxexp", "expmax", "maximumexperience", "experienceto"):
            exp_max = _num(v); continue
        if nk in ("salarymin", "minsalary") or nk in ("salarymax", "maxsalary"):
            job.salary = (job.salary + " - " if job.salary else "") + str(v); continue
        attr = _KEY_MAP.get(nk)
        if not attr:
            continue
        if attr in ("required_skills", "preferred_skills"):
            setattr(job, attr, _split_list(v))
        elif attr == "responsibilities" and isinstance(v, (list, tuple)):
            job.responsibilities = "; ".join(str(x) for x in v)
        else:
            setattr(job, attr, str(v).strip())
    if (exp_min is not None or exp_max is not None) and not job.experience_text:
        job.exp_min = exp_min if exp_min is not None else 0.0
        job.exp_max = exp_max
        if not job.experience_text:
            job.experience_text = (f"{_fmt_num(job.exp_min)} - {_fmt_num(exp_max)} Years"
                                   if exp_max is not None else f"{_fmt_num(job.exp_min)}+ Years")
    return job


def _job_from_line(line: str, known_cities: dict) -> Job:
    job = Job(raw=line)
    segments = [s.strip() for s in re.split(r"\s*\|\s*", line) if s.strip()]
    comma_mode = len(segments) == 1 and line.count(",") >= 2 and ":" not in line
    if comma_mode:
        segments = [s.strip() for s in line.split(",") if s.strip()]
    unlabelled = []
    for seg in segments:
        m = re.match(r"^([A-Za-z][A-Za-z _/-]{1,30})\s*[:=]\s*(.+)$", seg)
        attr = _KEY_MAP.get(_keynorm(m.group(1))) if m else None
        if attr:
            value = m.group(2).strip()
            if attr in ("required_skills", "preferred_skills"):
                setattr(job, attr, _split_list(value))
            else:
                setattr(job, attr, value)
        else:
            unlabelled.append(seg)

    leftovers = []
    for seg in unlabelled:
        low = _norm(seg)
        if not job.experience_text and _EXP_SEGMENT.search(low):
            job.experience_text = seg
        elif not job.employment_type and any(re.fullmatch(p, low.strip()) for p in JOB_TYPES.values()):
            job.employment_type = seg
        elif not job.location and _find_cities(low, known_cities) and len(low.split()) <= 4:
            job.location = seg
        else:
            leftovers.append(seg)
    if comma_mode and leftovers:
        job.title, leftovers = leftovers[0], []
        job.required_skills = job.required_skills or [x for x in unlabelled[1:]
                                                      if x not in (job.experience_text, job.employment_type, job.location)]
    for seg in leftovers:
        if not job.title:
            job.title = seg
        elif "," in seg and not job.required_skills:
            job.required_skills = _split_list(seg)
        elif not job.department and len(seg.split()) <= 3:
            job.department = seg
        elif not job.required_skills:
            job.required_skills = _split_list(seg)
        else:
            job.description = (job.description + " " + seg).strip()
    return job


def _city_lookup(extra_locations: List[str]) -> dict:
    """alias -> canonical city, including any city that appears in the job data."""
    lookup = {}
    for canon, aliases in CITY_ALIASES.items():
        for a in aliases:
            lookup[a] = canon
    for loc in extra_locations:
        for part in re.split(r"[,/;&]| and ", _norm(loc)):
            part = part.strip()
            if part and part not in lookup and not re.search(r"\d", part) and len(part) > 2:
                lookup[part] = part
    return lookup


def _find_cities(text_norm: str, lookup: dict) -> set:
    found = set()
    for alias in sorted(lookup, key=len, reverse=True):
        if re.search(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", text_norm):
            found.add(lookup[alias])
    return found


def _index(job: Job, lookup: dict) -> Job:
    if "_" in job.employment_type or job.employment_type.isupper():
        job.employment_type = job.employment_type.replace("_", "-").title()   # FULL_TIME -> Full-Time
    if job.exp_min is None and job.experience_text:
        job.exp_min, job.exp_max = parse_experience(job.experience_text)
    job.cities = _find_cities(_norm(job.location + " " + job.work_mode), lookup)
    tl = _norm(job.employment_type + " " + job.work_mode)
    job.type_key = next((k for k, p in JOB_TYPES.items() if re.search(p, tl)), "")
    job.title_tokens = set(_tokens(job.title))
    job.tokens = job.title_tokens | set(_tokens(" ".join(job.required_skills + job.preferred_skills))) \
        | set(_tokens(job.department))
    return job


# "Label: value" blocks, e.g. the backend's
#   ===== JOB INFORMATION =====
#   Job ID: 1
#   Title: Java Full Stack Developer
#   ...
# Works whether each field is on its own line or all fields are on one line.
_FIELD_LABELS = [
    "job id", "id", "job title", "title", "role", "position", "designation", "job name",
    "department", "dept", "category", "team", "location", "job location", "city",
    "employment type", "job type", "work mode", "work type", "experience required",
    "experience range", "years of experience", "experience", "min experience", "max experience",
    "minimum experience", "maximum experience", "required skills", "must have skills",
    "mandatory skills", "key skills", "skills required", "skills", "preferred skills",
    "nice to have skills", "nice to have", "good to have", "education required", "education",
    "qualification", "number of openings", "no of openings", "openings", "vacancies",
    "salary range", "salary", "ctc", "closing date", "application deadline", "deadline",
    "last date", "job description", "description", "about the role", "about",
    "responsibilities", "status", "job status", "company", "posted date", "posted on",
]
_LABEL_RE = re.compile(
    r"(?<![A-Za-z])(" + "|".join(re.escape(l) for l in sorted(_FIELD_LABELS, key=len, reverse=True))
    + r")\s*:", re.I)
_SECTION_RE = re.compile(r"={3,}\s*([^=\n]+?)\s*={3,}")
# Lines meant for an LLM / recruiters, never shown to candidates.
_INSTRUCTION_LINE = re.compile(
    r"^\s*(use the .* data above|do not invent|don't invent|if the requested information|"
    r"answer (only|recruiter)|you are an? |instructions?:)", re.I)


def _job_sections(text: str) -> str:
    """Keep only the job part of the context; drop instructions and other sections."""
    if not _SECTION_RE.search(text):
        return text
    parts = _SECTION_RE.split(text)            # [before, name1, body1, name2, body2, ...]
    keep = [parts[i + 1] for i in range(1, len(parts) - 1, 2)
            if "JOB" in parts[i].upper() and "INSTRUCTION" not in parts[i].upper()]
    return "\n".join(keep)


def _is_labelled_blocks(text: str) -> bool:
    return " | " not in text and len(re.findall(r"(?<![A-Za-z])(job )?title\s*:", text, re.I)) >= 1 \
        and len(_LABEL_RE.findall(text)) >= 2


def _parse_labelled_blocks(text: str) -> List[dict]:
    text = "\n".join(l for l in text.splitlines() if not _INSTRUCTION_LINE.search(l))
    matches = list(_LABEL_RE.finditer(text))
    records, current = [], {}
    for i, m in enumerate(matches):
        label = m.group(1).lower()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        value = re.sub(r"\s+", " ", text[m.end():end]).strip(" ,;|")
        starts_new = label in ("job id", "id") or (label in ("title", "job title") and
                                                  any(k in current for k in ("title", "job title")))
        if starts_new and current:
            records.append(current)
            current = {}
        if value:
            current[label] = value
    if current:
        records.append(current)
    closed = re.compile(r"closed|inactive|expired|filled|draft", re.I)
    return [r for r in records if (r.get("title") or r.get("job title"))
            and not closed.search(r.get("status", "") + r.get("job status", ""))]


def parse_jobs(job_context=None, jobs=None) -> List[Job]:
    """Build Job objects from the structured list and/or the job_context string."""
    raw_dicts, lines = [], []
    for item in jobs or []:
        if hasattr(item, "dict"):
            item = item.dict()
        if isinstance(item, dict):
            raw_dicts.append(item)
        elif item:
            lines.append(str(item))

    # The structured list wins: if the backend sends both (during migration), job_context
    # is ignored so jobs are not listed twice.
    if job_context and str(job_context).strip() and not raw_dicts and not lines:
        text = str(job_context).strip()
        parsed = None
        if text[0] in "[{":
            try:
                parsed = json.loads(text)
            except ValueError:
                parsed = None
        if isinstance(parsed, dict):
            parsed = parsed.get("jobs") or parsed.get("content") or [parsed]
        if isinstance(parsed, list):
            raw_dicts.extend(p for p in parsed if isinstance(p, dict))
        elif _is_labelled_blocks(_job_sections(text)):
            raw_dicts.extend(_parse_labelled_blocks(_job_sections(text)))
        else:
            text = "\n".join(l for l in _job_sections(text).splitlines() if not _INSTRUCTION_LINE.search(l))
            lines.extend(l.strip(" -*•\t") for l in text.splitlines() if l.strip(" -*•\t"))

    result = [_job_from_dict(d) for d in raw_dicts]
    pre_lookup = _city_lookup([j.location for j in result])
    result += [_job_from_line(l, pre_lookup) for l in lines]
    lookup = _city_lookup([j.location for j in result])

    # Drop closed / expired jobs and duplicates (same id, or same title + location)
    closed = re.compile(r"\b(closed|inactive|expired|filled|draft|on hold)\b", re.I)
    unique, seen = [], set()
    for j in result:
        if not j.title or closed.search(j.status):
            continue
        key = ("id", str(j.job_id)) if j.job_id else ("tl", j.title.lower().strip(), j.location.lower().strip())
        if key in seen:
            continue
        seen.add(key)
        unique.append(_index(j, lookup))
    return unique


# ============================================================
# PARSING THE QUESTION
# ============================================================

def _strip_experience(text: str, q: JobQuery) -> str:
    """Read the experience part of the question into q and remove it from the text."""
    t = _words_to_numbers(text)
    has_unit = bool(re.search(r"\b(yrs?|years?|yoe|exp|experience|experienced)\b", t))
    # A bare range on its own ("for 2-3", "2 to 4") is an experience range too.
    if not has_unit and re.fullmatch(r"\W*(?:for|with|of|around|about|having|i have)?\s*"
                                     r"\d+(?:\.\d+)?\s*(?:-|to)\s*\d+(?:\.\d+)?\W*", t.strip()):
        has_unit = True
    unit = r"\s*(?:\+\s*)?(?:yrs?|years?|yoe)?"
    rules = [
        (r"(?:between\s*)?(\d+(?:\.\d+)?)\s*(?:-|to|and)\s*(\d+(?:\.\d+)?)" + unit, "range"),
        (r"(?:more than|over|above|at ?least|minimum|min|greater than)\s*(\d+(?:\.\d+)?)" + unit, "min"),
        (r"(\d+(?:\.\d+)?)\s*\+\s*(?:yrs?|years?|yoe)?", "min"),
        (r"(?:less than|under|below|upto|up to|maximum|max|within)\s*(\d+(?:\.\d+)?)" + unit, "max"),
        (r"(\d+(?:\.\d+)?)\s*(?:yrs?|years?|yoe)\b", "exact"),
    ]
    for pat, kind in rules:
        m = re.search(pat, t)
        if not m or (kind != "exact" and not has_unit):
            continue
        if kind == "range":
            a, b = sorted((float(m.group(1)), float(m.group(2))))
            q.exp_min, q.exp_max, q.exp_label = a, b, f"{_fmt_num(a)}-{_fmt_num(b)} years"
        elif kind == "min":
            q.exp_min, q.exp_label = float(m.group(1)), f"{_fmt_num(float(m.group(1)))}+ years"
        elif kind == "max":
            q.exp_max, q.exp_label = float(m.group(1)), f"up to {_fmt_num(float(m.group(1)))} years"
        else:
            n = float(m.group(1))
            q.exp_exact, q.exp_label = n, f"{_fmt_num(n)} year{'s' if n != 1 else ''}"
        t = t[:m.start()] + " " + t[m.end():]
        break

    if not q.has_experience():
        if re.search(r"\b(fresher|freshers|entry[\s-]?level|no experience|0 experience|zero experience|"
                     r"just graduated|recent graduate|beginner|beginners)\b", t):
            q.fresher, q.exp_label = True, "freshers"
        elif re.search(r"\b(senior|lead)\b", t):
            q.exp_min, q.exp_label = 5.0, "senior (5+ years)"
        elif re.search(r"\b(experienced|with experience|experience jobs|exp jobs|lateral)\b", t):
            q.experienced, q.exp_label = True, "experienced candidates"
    return t


def parse_query(message: str, jobs: List[Job]) -> JobQuery:
    q = JobQuery()
    text = _norm(message)
    q.explicit_job_word = bool(EXPLICIT_JOB_WORDS.search(text))
    q.detail_request = bool(DETAIL_CUES.search(text))

    text = _strip_experience(text, q)
    lookup = _city_lookup([j.location for j in jobs])

    # "jobs in other cities" / "elsewhere"
    if OTHER_CITIES.search(text):
        q.other_cities = True
        text = OTHER_CITIES.sub(" ", text)

    # "other than hyderabad", "except java and python", "non-java jobs"
    for m in list(NEGATION.finditer(text))[::-1]:
        tail = text[m.end():]
        stop = re.search(r"\b(for|with|jobs?|openings?|roles?|positions?|that|which|having|where|"
                         r"in which|please)\b|[?.!]", tail)
        segment = tail[:stop.start()] if stop else tail
        segment = " ".join(segment.split()[:5])
        if not segment:
            continue
        cities = _find_cities(segment, lookup)
        q.exclude_cities |= cities
        rest = segment
        for alias in sorted(lookup, key=len, reverse=True):
            if lookup[alias] in cities:
                rest = re.sub(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", " ", rest)
        q.exclude_keywords += [w for w in _tokens(rest) if w not in STOPWORDS and not w.isdigit()
                               and w not in ("and", "or", "in", "the") and len(w) > 1]
        text = text[:m.start()] + " " + tail[len(tail[:stop.start()]) if stop else len(tail):]

    q.cities = _find_cities(text, lookup) - q.exclude_cities
    for alias in sorted(lookup, key=len, reverse=True):
        if lookup[alias] in q.cities:
            text = re.sub(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", " ", text)

    for key, pat in JOB_TYPES.items():
        if re.search(r"\b(" + pat + r")\b", text):
            q.job_type = key
            text = re.sub(r"\b(" + pat + r")\b", " ", text)
            break

    q.keywords = [w for w in _tokens(text) if w not in STOPWORDS and not w.isdigit()
                  and (len(w) > 1 or w in ("c", "r"))]
    return q


# ============================================================
# MATCHING
# ============================================================

def _kw_match(kw: str, tokens: set) -> bool:
    if kw in tokens:
        return True
    if kw.endswith("s") and kw[:-1] in tokens:
        return True
    for t in tokens:
        # data scientist ~ science, developer ~ development, analyst ~ analytics
        if len(kw) >= 5 and len(t) >= 5 and kw[:5] == t[:5]:
            return True
    return False


def _exp_ok(job: Job, q: JobQuery) -> bool:
    if not q.has_experience():
        return True
    if job.exp_min is None:            # job has no experience info -> don't hide it
        return True
    lo, hi = job.exp_min, job.exp_max if job.exp_max is not None else float("inf")
    if q.fresher:
        return lo == 0
    if q.experienced:
        return lo >= 1 or hi > 1
    if q.exp_exact is not None:
        return lo <= q.exp_exact <= hi
    qlo = q.exp_min if q.exp_min is not None else 0
    qhi = q.exp_max if q.exp_max is not None else float("inf")
    return lo <= qhi and qlo <= hi     # ranges overlap


def _city_ok(job: Job, q: JobQuery) -> bool:
    if q.exclude_cities and job.cities & q.exclude_cities:
        return False
    return not q.cities or bool(job.cities & q.cities)


def _not_excluded(job: Job, q: JobQuery) -> bool:
    return not any(_kw_match(k, job.tokens) for k in q.exclude_keywords)


def _type_ok(job: Job, q: JobQuery) -> bool:
    return not q.job_type or job.type_key == q.job_type


def _kw_hits(job: Job, keywords: List[str]) -> int:
    return sum(1 for k in keywords if _kw_match(k, job.tokens))


# ============================================================
# FORMATTING
# ============================================================

def _skills_str(skills: List[str]) -> str:
    return ", ".join(skills)


def _openings(job: Job) -> Optional[int]:
    """Number of positions for one job listing (the 'openings' field), if the backend sent it."""
    n = _num(job.openings) if job.openings not in (None, "") else None
    return int(n) if n is not None else None


def _count_sentence(jobs: List[Job], what: str = "job listings") -> str:
    """'There are 3 openings across 2 job listings right now' (falls back to listings only)."""
    n = len(jobs)
    seats = [_openings(j) for j in jobs]
    listings = what if n != 1 else what.replace("listings", "listing").replace("jobs", "job", 1)
    if any(s is not None for s in seats):
        total = sum(s if s is not None else 1 for s in seats)
        if total != n or n == 1:
            return (f"There {'is' if total == 1 else 'are'} {total} opening{'' if total == 1 else 's'} "
                    f"across {n} {listings} right now:" if n > 1 else
                    f"There {'is' if total == 1 else 'are'} {total} opening{'' if total == 1 else 's'} "
                    f"for 1 {listings} right now:")
    return f"There {'is' if n == 1 else 'are'} {n} {listings} right now:"


def format_job_short(job: Job, n: int) -> str:
    head = f"**{n}) {job.title}**" + (f" ({job.department})" if job.department else "")
    seats = _openings(job)
    meta = " | ".join(x for x in (job.location, job.experience_text, job.employment_type, job.work_mode,
                                  f"{seats} opening{'s' if seats != 1 else ''}" if seats else "") if x)
    lines = [head]
    if meta:
        lines.append(f"- {meta}")
    if job.required_skills:
        lines.append(f"- Skills: {_skills_str(job.required_skills)}")
    if job.preferred_skills:
        lines.append(f"- Preferred: {_skills_str(job.preferred_skills)}")
    return "\n".join(lines)


def _inr(n: float) -> str:
    """Indian digit grouping: 200000 -> 2,00,000."""
    n = int(round(n))
    s = str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = ",".join([head[max(i - 2, 0):i] for i in range(len(head), 0, -2)][::-1])
        s = head + "," + tail
    return ("-" if n < 0 else "") + s


def _fmt_salary(text: str) -> str:
    if not text:
        return ""
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(text).replace(",", ""))]
    if not nums or re.search(r"[a-z]{3,}", str(text), re.I) and not re.search(r"lpa|inr|rs|per", str(text), re.I):
        return str(text)                      # "Not specified", "Negotiable", ...
    if len(nums) >= 2 and nums[0] != nums[1]:
        return f"{_inr(nums[0])} - {_inr(nums[1])}"
    return _inr(nums[0]) if nums[0] >= 1000 else str(text)


def format_job_full(job: Job) -> str:
    rows = [
        ("Department", job.department), ("Location", job.location),
        ("Experience", job.experience_text), ("Employment type", job.employment_type),
        ("Work mode", job.work_mode), ("Required skills", _skills_str(job.required_skills)),
        ("Preferred skills", _skills_str(job.preferred_skills)), ("Education", job.education),
        ("Openings", job.openings), ("Salary range", _fmt_salary(job.salary)), ("Closing date", job.closing_date),
        ("About the role", job.description), ("Responsibilities", job.responsibilities
                             if _norm(job.responsibilities).strip() != _norm(job.description).strip() else ""),
    ]
    body = "\n".join(f"- {k}: {v}" for k, v in rows if v)
    return (f"**{job.title}**\n{body}\n\nTo apply, open this job on the Jobs page and click Apply Now.")


def _list(jobs: List[Job], header: str) -> str:
    if len(jobs) == 1:
        return header + "\n\n" + format_job_full(jobs[0])
    return header + "\n\n" + "\n\n".join(format_job_short(j, i + 1) for i, j in enumerate(jobs)) + \
        "\n\nAsk me for the details of any of these jobs, e.g. \"details of " + jobs[0].title + "\"."


_DISPLAY = {"springboot": "Spring Boot", "fullstack": "full stack", "restapi": "REST API", "nodejs": "Node.js",
            "nextjs": "Next.js", "cpp": "C++", "csharp": "C#", "dotnet": ".NET", "powerbi": "Power BI",
            "frontend": "frontend", "backend": "backend", "devops": "DevOps"}


def _pretty(keywords: List[str]) -> str:
    return " ".join(_DISPLAY.get(k, k) for k in keywords)


def _describe(q: JobQuery, keywords: List[str]) -> str:
    parts = []
    if keywords:
        parts.append(_pretty(keywords))
    if q.job_type:
        parts.append(q.job_type)
    parts.append("jobs")
    if q.cities:
        parts.append("in " + " / ".join(sorted(c.title() for c in q.cities)))
    if q.exclude_cities:
        parts.append("outside " + " / ".join(sorted(c.title() for c in q.exclude_cities)))
    if q.exclude_keywords:
        parts.append("without " + _pretty(q.exclude_keywords))
    if q.exp_label:
        parts.append("for " + q.exp_label)
    return " ".join(parts)


# ============================================================
# PUBLIC API
# ============================================================

def is_job_search(message: str, jobs: List[Job], previous_message: Optional[str] = None) -> bool:
    """Decide whether the message is a job search (vs. a career-guidance question)."""
    q = parse_query(message, jobs)
    if q.explicit_job_word or q.has_experience() or q.cities or q.job_type \
            or q.exclude_cities or q.other_cities:
        # "salary of a data analyst with 3 years experience" is still a career question
        if CAREER_CUES.search(message) and not (q.explicit_job_word or q.cities or q.exclude_cities):
            return False
        return True
    if CAREER_CUES.search(message):
        return False
    # Short message that names a skill / title we have, e.g. "python", "java developer", "react"
    if q.keywords and len(message.split()) <= 5 and any(_kw_hits(j, q.keywords) for j in jobs):
        return True
    # Detail request for a job title we have ("details of java full stack developer")
    if q.detail_request and _best_title_match(q.keywords, jobs):
        return True
    # Follow-up of a previous job search ("and python?")
    if previous_message and FOLLOW_UP_CUES.search(message.strip()) and q.keywords \
            and is_job_search(previous_message, jobs):
        return any(_kw_hits(j, q.keywords) for j in jobs)
    return False


def _best_title_match(keywords: List[str], jobs: List[Job]) -> Optional[Job]:
    """One job whose title clearly matches the keywords (for 'details of X')."""
    if not keywords:
        return None
    scored = []
    for j in jobs:
        hits = sum(1 for k in keywords if _kw_match(k, j.title_tokens))
        if hits:
            scored.append((hits / max(len(j.title_tokens), 1), hits, j))
    if not scored:
        return None
    scored.sort(key=lambda x: (x[1], x[0]), reverse=True)
    if len(scored) == 1 or scored[0][1] > scored[1][1] or scored[0][0] > scored[1][0]:
        return scored[0][2]
    return None


def _merge_follow_up(q: JobQuery, prev: JobQuery) -> JobQuery:
    """'python jobs' -> 'and in hyderabad?' keeps python and adds Hyderabad."""
    if not q.has_experience():
        q.exp_min, q.exp_max, q.exp_exact = prev.exp_min, prev.exp_max, prev.exp_exact
        q.fresher, q.experienced, q.exp_label = prev.fresher, prev.experienced, prev.exp_label
    if not q.cities and not q.exclude_cities and not q.other_cities:
        q.cities = prev.cities
    if not q.job_type:
        q.job_type = prev.job_type
    if not q.keywords:
        q.keywords = prev.keywords
    return q


def search_jobs(message: str, job_context=None, jobs=None,
                previous_message: Optional[str] = None,
                no_jobs_message: str = "No job listings are available right now.") -> str:
    all_jobs = parse_jobs(job_context, jobs)
    if not all_jobs:
        return no_jobs_message

    q = parse_query(message, all_jobs)
    if previous_message and FOLLOW_UP_CUES.search(message.strip()) and is_job_search(previous_message, all_jobs):
        q = _merge_follow_up(q, parse_query(previous_message, all_jobs))

    # "jobs in other cities" -> other than the city asked about before, or the only city we have
    all_cities = set().union(*(j.cities for j in all_jobs)) if all_jobs else set()
    if q.other_cities and not q.exclude_cities:
        prev_cities = parse_query(previous_message, all_jobs).cities if previous_message else set()
        if prev_cities:
            q.exclude_cities = set(prev_cities)
        elif len(all_cities) == 1:
            q.exclude_cities = set(all_cities)
        elif all_cities:
            return _list(all_jobs, "Current openings are in "
                         + ", ".join(sorted(c.title() for c in all_cities))
                         + ". Here they are, or ask e.g. \"jobs in " + sorted(all_cities)[0].title() + "\":")
    q.cities -= q.exclude_cities

    # Keywords that exist in our jobs vs. ones that don't (typos, skills nobody asks for)
    known_kw = [k for k in q.keywords if any(_kw_match(k, j.tokens) for j in all_jobs)]
    unknown_kw = [k for k in q.keywords if k not in known_kw]

    # "details of java full stack developer"
    if q.detail_request and known_kw:
        job = _best_title_match(known_kw, all_jobs)
        if job:
            return format_job_full(job)

    counting = bool(COUNT_CUES.search(message))

    def header(found: List[Job], desc: str) -> str:
        n = len(found)
        if counting:
            return _count_sentence(found, desc)
        return f"I found {n} {desc}:" if n > 1 else f"I found 1 match for {desc}:"

    if not q.has_filters():
        if counting:
            return _list(all_jobs, _count_sentence(all_jobs))
        return _list(all_jobs, f"Here are the {len(all_jobs)} current job listings:")

    if unknown_kw and not known_kw and not (q.has_experience() or q.cities or q.job_type
                                            or q.exclude_cities or q.exclude_keywords):
        return (f"I couldn't find any jobs for '{_pretty(unknown_kw)}'.\n\n"
                + _list(all_jobs, f"Here are the {len(all_jobs)} current job listings:"))

    def candidates(use_exp=True, use_city=True, use_type=True):
        out = [j for j in all_jobs
               if (not use_exp or _exp_ok(j, q)) and (not use_city or _city_ok(j, q))
               and (not use_type or _type_ok(j, q)) and (not known_kw or _kw_hits(j, known_kw))
               and _not_excluded(j, q)]
        out.sort(key=lambda j: _kw_hits(j, known_kw), reverse=True)
        return out

    note = ""
    matched = candidates()
    if unknown_kw and known_kw:
        asked = _pretty([k for k in q.keywords])
        if matched:
            return _list(matched, f"There's no {asked} job open right now. "
                                  f"{'This related job is' if len(matched) == 1 else 'These related jobs are'} open:")
        note = f"There's no {asked} job open right now.\n\n"
    if matched:
        full = [j for j in matched if _kw_hits(j, known_kw) == len(known_kw)]
        if full:
            return note + _list(full, header(full, _describe(q, known_kw)))
        return note + _list(matched, f"No job matches all of '{_pretty(known_kw)}', "
                                     f"but these are related:")

    wanted = _describe(q, known_kw)

    # Nothing left after "other than X" -> say so plainly instead of showing X again
    if q.exclude_cities or q.exclude_keywords:
        where = ""
        if q.exclude_cities and all_cities and all_cities <= q.exclude_cities:
            where = " All current openings are in " + ", ".join(sorted(c.title() for c in all_cities)) + "."
        return note + f"Sorry, there are no {wanted} right now.{where}\n\n" + \
            _list(all_jobs, f"Here are the {len(all_jobs)} current job listings:")

    # Nothing matched -> relax one filter at a time and say what was relaxed
    for label, kwargs in (("in other locations", dict(use_city=False)),
                          ("with a different experience range", dict(use_exp=False)),
                          ("with a different job type", dict(use_type=False)),
                          ("", dict(use_city=False, use_exp=False, use_type=False))):
        if label == "in other locations" and not q.cities:
            continue
        if label.startswith("with a different experience") and not q.has_experience():
            continue
        if label.startswith("with a different job type") and not q.job_type:
            continue
        relaxed = candidates(**kwargs)
        if relaxed:
            suffix = f" Here are related jobs {label}:" if label else " Here are related jobs:"
            return note + _list(relaxed, f"Sorry, there are no {wanted} right now.{suffix}")

    return (note + f"Sorry, there are no {wanted} right now.\n\n"
            + _list(all_jobs, f"Here are the {len(all_jobs)} current job listings:"))
