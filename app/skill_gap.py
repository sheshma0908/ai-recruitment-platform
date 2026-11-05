"""
Skill Gap Analysis (Module 17) and Candidate Ranking (Module 18).

Both reuse the matching engine directly -- no separate model or dataset needed.
"""
from app.ml_utils import parse_skills, normalize_skill
from app.matching import score_candidate


def skill_gap_analysis(job: dict, resume: dict) -> dict:
    """Returns missing must-have / nice-to-have skills and coverage percentages
    for one job-resume pair."""
    job_must = {normalize_skill(s) for s in parse_skills(job["must_have_skills"])}
    job_nice = {normalize_skill(s) for s in parse_skills(job["nice_to_have_skills"])}
    resume_skills = {normalize_skill(s) for s in parse_skills(resume["skills"])}

    missing_must = job_must - resume_skills
    missing_nice = job_nice - resume_skills
    matched_must = job_must & resume_skills

    must_coverage = len(matched_must) / len(job_must) * 100 if job_must else 100.0
    nice_coverage = len(job_nice & resume_skills) / len(job_nice) * 100 if job_nice else 100.0

    return {
        "job_id": job["job_id"],
        "resume_id": resume["resume_id"],
        "must_have_coverage_pct": round(must_coverage, 1),
        "nice_to_have_coverage_pct": round(nice_coverage, 1),
        "missing_must_have": sorted(missing_must),
        "missing_nice_to_have": sorted(missing_nice),
    }


def rank_candidates_for_job(job: dict, candidates: list, top_n: int = 10) -> list:
    """Scores every candidate against one job, returns the top N ranked by
    match % (ties broken by the AI model), each annotated with their skill gap."""
    rows = []
    for resume in candidates:
        match = score_candidate(job, resume)
        gap = skill_gap_analysis(job, resume)

        rows.append({
            "resume_id": resume["resume_id"],
            "candidate_role": resume["role"],
            "match_probability_pct": match["match_probability_pct"],
            "must_have_coverage_pct": gap["must_have_coverage_pct"],
            "missing_must_have": gap["missing_must_have"],
        })

        rows[-1]["_ai_model_pct"] = match["ai_model_probability_pct"]

    # Highest match % first; equal match % -> higher AI model probability first
    ranked = sorted(rows, key=lambda r: (r["match_probability_pct"], r["_ai_model_pct"]), reverse=True)
    for r in ranked:
        r.pop("_ai_model_pct", None)
    for i, r in enumerate(ranked[:top_n], 1):
        r["rank"] = i

    return ranked[:top_n]
