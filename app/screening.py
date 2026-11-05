"""
AI Resume Screening (Module 6) -- standalone from Module 7's matching engine.

Extracts structured info from both sides (job requirements, candidate profile) and
compares them dimension by dimension, BEFORE any single weighted score is produced.
Module 7 (matching.py) consumes the same underlying signals to produce that score.
"""
import re

from app.ml_utils import parse_skills, normalize_skill, get_tokens
from app.matching import vectorizer as _match_vectorizer  # reuse Module 7's TF-IDF vectorizer


def extract_job_requirements(job: dict) -> dict:
    """The 'job description' side of screening."""
    return {
        "job_id": job["job_id"],
        "job_title": job["job_title"],
        "seniority_required": job["seniority"],
        "industry": job["industry"],
        "must_have_skills": [normalize_skill(s) for s in parse_skills(job["must_have_skills"])],
        "nice_to_have_skills": [normalize_skill(s) for s in parse_skills(job["nice_to_have_skills"])],
    }


def extract_resume_profile(resume: dict) -> dict:
    """The 'resume' side of screening, from a structured resume record."""
    return {
        "resume_id": resume["resume_id"],
        "role": resume["role"],
        "seniority": resume["seniority"],
        "years_experience": resume.get("years_experience"),
        "industry": resume["industry"],
        "education": resume.get("education", ""),
        "skills": [normalize_skill(s) for s in parse_skills(resume["skills"])],
    }


def extract_resume_profile_from_text(raw_text: str) -> dict:
    """The 'resume' side of screening, from raw resume text -- bridges Module 5
    (NER parsing) into Module 6 (screening) for real, unstructured resume uploads.
    Reuses parse_resume() directly rather than duplicating its extraction logic,
    so any future fix to skill extraction only needs to happen in one place."""
    from app.resume_parsing import parse_resume

    profile = parse_resume(raw_text)

    return {
        "resume_id": None,
        "role": profile["designation"][0] if profile["designation"] else None,
        "seniority": None,  # not in this NER model's label set -- correctly left unavailable, not guessed
        "years_experience": profile["years_of_experience"],
        "industry": None,   # same as above
        "education": profile["degree"],
        "skills": [normalize_skill(s) for s in profile["skills_cleaned"]],
        "companies_worked_at": profile["companies_worked_at"],
    }


def screen_resume_against_job(resume_profile: dict, job_requirements: dict,
                               resume_text: str = None, job_text: str = None) -> dict:
    """Compares resume_profile against job_requirements dimension by dimension.
    Pass resume_text/job_text to also include the TF-IDF content-relevance dimension
    (the same signal Module 7 uses as one of its 8 features)."""
    dimensions = []

    resume_skills = set(resume_profile.get("skills") or [])
    must = set(job_requirements.get("must_have_skills") or [])
    nice = set(job_requirements.get("nice_to_have_skills") or [])

    matched_must = sorted(must & resume_skills)
    missing_must = sorted(must - resume_skills)
    must_pct = len(matched_must) / len(must) * 100 if must else 100.0
    dimensions.append({
        "dimension": "Must-Have Skills", "match_pct": round(must_pct, 1),
        "matched": matched_must, "missing": missing_must,
    })

    matched_nice = sorted(nice & resume_skills)
    missing_nice = sorted(nice - resume_skills)
    nice_pct = len(matched_nice) / len(nice) * 100 if nice else 100.0
    dimensions.append({
        "dimension": "Nice-to-Have Skills", "match_pct": round(nice_pct, 1),
        "matched": matched_nice, "missing": missing_nice,
    })

    resume_sen, job_sen = resume_profile.get("seniority"), job_requirements.get("seniority_required")
    dimensions.append({
        "dimension": "Seniority Level", "resume_value": resume_sen, "job_requirement": job_sen,
        "match": (resume_sen == job_sen) if resume_sen and job_sen else None,
    })

    resume_ind, job_ind = resume_profile.get("industry"), job_requirements.get("industry")
    dimensions.append({
        "dimension": "Industry Background", "resume_value": resume_ind, "job_requirement": job_ind,
        "match": (resume_ind == job_ind) if resume_ind and job_ind else None,
    })

    resume_role = resume_profile.get("role") or ""
    job_title = job_requirements.get("job_title") or ""
    role_words, job_words = get_tokens(resume_role), get_tokens(job_title)
    role_sim = len(role_words & job_words) / len(role_words | job_words) * 100 if (role_words | job_words) else 0.0
    dimensions.append({
        "dimension": "Role / Title Alignment", "resume_value": resume_role,
        "job_requirement": job_title, "similarity_pct": round(role_sim, 1),
    })

    if resume_text is not None and job_text is not None:
        job_vec = _match_vectorizer.transform([job_text])
        res_vec = _match_vectorizer.transform([resume_text])
        text_sim = float(job_vec.multiply(res_vec).sum())
        dimensions.append({
            "dimension": "Overall Content Relevance", "similarity_pct": round(text_sim * 100, 1),
            "note": "TF-IDF cosine similarity between resume text and job description text -- "
                    "the same signal Module 7 uses as one of its 8 model features.",
        })

    # Overall match row first: same match % and Match / No Match rule as
    # /match-score. Uses its own keys (overall_match_pct / result) so the
    # backend's per-dimension screening average is not changed by it.
    from app.matching import MATCH_THRESHOLD_PCT, match_result_from_pct
    if must:
        overall_pct = round(must_pct, 1)
    elif nice:
        overall_pct = round(nice_pct, 1)
    else:
        overall_pct = 0.0
    dimensions.insert(0, {
        "dimension": "Overall Match",
        "overall_match_pct": overall_pct,
        "result": match_result_from_pct(overall_pct),
        "note": f"Must-have skills matched. {MATCH_THRESHOLD_PCT:g}% or more is a Match.",
    })

    return {
        "job_id": job_requirements.get("job_id"),
        "resume_id": resume_profile.get("resume_id"),
        "extracted_resume_profile": resume_profile,
        "extracted_job_requirements": job_requirements,
        "screening_dimensions": dimensions,
    }
