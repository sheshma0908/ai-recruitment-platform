"""
AI Interview Question Generation (Module 19) -- LLM prompting, not a trained
model. Supports Anthropic or OpenAI; falls back to a template-based generator
if no API key is configured, so the endpoint always returns something usable.
"""
import os
import json
import logging
import random

# Optional: load ANTHROPIC_API_KEY / OPENAI_API_KEY from a .env file in the
# project root, if python-dotenv is installed. Without this, the keys MUST be
# exported in the same terminal that starts uvicorn.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("uvicorn")

from app.skill_gap import skill_gap_analysis
from app.ml_utils import parse_skills, normalize_skill

INTERVIEW_PROMPT_TEMPLATE = """You are an experienced technical interviewer preparing questions \
for a candidate interview.

JOB TITLE: {job_title}
JOB DESCRIPTION: {job_description}
REQUIRED SKILLS (must-have): {must_have_skills}
REQUIRED SKILLS (nice-to-have): {nice_to_have_skills}

CANDIDATE BACKGROUND:
Current role: {candidate_role}
Years of experience: {candidate_experience}
Candidate's skills: {candidate_skills}

SKILL GAP ANALYSIS:
Skills the candidate already has that match the job: {matched_skills}
Required skills the candidate's resume does NOT show: {missing_skills}

TASK:
Generate exactly {num_questions} interview questions for this specific candidate and job. Follow these rules:
1. Ask 2-3 questions that probe the candidate's DEPTH in their matched skills (don't just ask them to
   define the skill -- ask about a real scenario, trade-off, or decision they'd have made using it).
2. Ask 1-2 questions that directly investigate the missing/gap skills -- to check whether the resume
   simply didn't mention it, or whether it's a genuine gap.
3. Include at least 1 behavioral/situational question relevant to the seniority level of this role.
4. Do NOT ask generic questions like "tell me about yourself" or "what are your strengths."
5. Return ONLY a JSON array, no other text, in this exact format:
[
  {{"question": "...", "category": "technical_depth | skill_gap | behavioral", "targets_skill": "..."}}
]
"""

TECHNICAL_DEPTH_TEMPLATES = [
    "Tell me about a time you used {skill} to solve a difficult problem. What trade-offs did you consider?",
    "How would you decide between two different approaches when working with {skill}?",
    "What's a mistake you've made while working with {skill}, and what did you learn from it?",
]
SKILL_GAP_TEMPLATES = [
    "Your resume doesn't mention {skill} -- have you worked with it, even informally?",
    "This role requires {skill}. Walk me through how quickly you could get up to speed on it.",
]
BEHAVIORAL_TEMPLATES = [
    "Describe a situation where you disagreed with a teammate's technical decision. How did you handle it?",
    "Tell me about a project that didn't go as planned. What would you do differently?",
]


def call_llm(prompt: str, provider: str = "anthropic", model: str = None) -> str:
    """Calls the chosen LLM provider. Raises if no API key is configured --
    caller should catch this and use the fallback."""
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model or "claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    else:
        raise ValueError(f"Unknown provider: {provider}")


def generate_questions_fallback(job, candidate_skills, matched_skills, missing_skills, num_questions=5):
    rng = random.Random(hash(job["job_id"]) % (2**32))
    questions = []

    depth_skills = matched_skills[:3] if matched_skills else candidate_skills[:3]
    for skill in depth_skills[:3]:
        template = rng.choice(TECHNICAL_DEPTH_TEMPLATES)
        questions.append({"question": template.format(skill=skill), "category": "technical_depth", "targets_skill": skill})

    for skill in missing_skills[:2]:
        template = rng.choice(SKILL_GAP_TEMPLATES)
        questions.append({"question": template.format(skill=skill), "category": "skill_gap", "targets_skill": skill})

    questions.append({"question": rng.choice(BEHAVIORAL_TEMPLATES), "category": "behavioral", "targets_skill": None})

    return questions[:num_questions]


def generate_interview_questions(job: dict, resume: dict, num_questions: int = 5, provider: str = "anthropic") -> dict:
    gap = skill_gap_analysis(job, resume)
    candidate_skills = parse_skills(resume["skills"])

    job_skill_set = {
        normalize_skill(s)
        for s in parse_skills(job["must_have_skills"]) + parse_skills(job["nice_to_have_skills"])
    }
    matched_skills = [s for s in candidate_skills if normalize_skill(s) in job_skill_set]

    prompt = INTERVIEW_PROMPT_TEMPLATE.format(
        job_title=job["job_title"],
        job_description=job.get("description", ""),
        must_have_skills=parse_skills(job["must_have_skills"]),
        nice_to_have_skills=parse_skills(job["nice_to_have_skills"]),
        candidate_role=resume["role"],
        candidate_experience=resume.get("years_experience", "unknown"),
        candidate_skills=candidate_skills,
        matched_skills=matched_skills,
        missing_skills=gap["missing_must_have"] + gap["missing_nice_to_have"],
        num_questions=num_questions,
    )

    provider = (provider or "anthropic").strip().lower()
    fallback_reason = None

    try:
        raw_response = call_llm(prompt, provider=provider)
        cleaned = raw_response.strip()
        for fence in ("```json", "```"):
            if cleaned.startswith(fence):
                cleaned = cleaned[len(fence):]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        questions = json.loads(cleaned.strip())
        source = f"llm:{provider}"
    except Exception as exc:
        # Previously this swallowed the error silently, so there was nothing
        # in the logs explaining WHY the fallback was used.
        fallback_reason = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "Interview questions: LLM provider '%s' failed, using offline_fallback. Reason: %s",
            provider, fallback_reason,
        )
        questions = generate_questions_fallback(
            job, candidate_skills, matched_skills or candidate_skills,
            gap["missing_must_have"] + gap["missing_nice_to_have"],
            num_questions=num_questions,
        )
        source = "offline_fallback"

    result = {"questions": questions, "source": source}
    if fallback_reason:
        result["fallback_reason"] = fallback_reason
    return result
