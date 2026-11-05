"""
Candidate-Job Matching (Modules 6/7) -- wraps the XGBoost model trained in
model_training_fixed.ipynb.

The XGBoost model itself is loaded from its own native format (.ubj), not via
pickle -- pickled XGBoost boosters can fail to deserialize when the loading
machine's xgboost build differs from the training machine's, even when the
reported version number is identical (this bit us once already -- see the
"input stream corrupted" issue). XGBoost's native save_model/load_model format
is explicitly designed to be portable across platforms and versions, so we use
that for the model and keep pickle only for the plain Python/sklearn objects
around it (which don't have this problem).
"""
import os
import joblib
import pandas as pd
import xgboost as xgb

from app.ml_utils import parse_skills, normalize_skill, get_tokens

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
_META_PATH = os.path.join(_MODELS_DIR, "match_engine_meta.pkl")
_MODEL_PATH = os.path.join(_MODELS_DIR, "match_model.ubj")

_meta = joblib.load(_META_PATH)
vectorizer = _meta["vectorizer"]
feature_columns = _meta["feature_columns"]
seniority_map = _meta["seniority_map"]
threshold = _meta["threshold"]

model = xgb.XGBClassifier()
model.load_model(_MODEL_PATH)

# ---------------------------------------------------------------------------
# MATCH PERCENTAGE + MATCH RULE (used by every endpoint)
# ---------------------------------------------------------------------------
# match % = must-have skills matched (same number the screening shows under
# "Must-Have Skills"). match % >= MATCH_THRESHOLD_PCT  ->  "Match".
#
# The XGBoost model is still run: its probability is returned inside
# `features` and breaks ties when ranking. It is not the deciding number,
# because it was trained on resumes that always had a role, seniority and
# industry; resumes parsed from text often have these empty, which pushes the
# model's probability close to 0 even with good skill overlap.
#
# Change the limit with the MATCH_THRESHOLD_PCT env var, e.g. 50.
MATCH_THRESHOLD_PCT = float(os.getenv("MATCH_THRESHOLD_PCT", "40"))


def match_result_from_pct(match_pct: float) -> str:
    return "Match" if match_pct >= MATCH_THRESHOLD_PCT else "No Match"


def compute_match_pct(must_coverage: float, nice_coverage: float,
                      has_must: bool, has_nice: bool, model_probability: float) -> float:
    """Overall match % (0-100) for one job-resume pair."""
    if has_must:
        value = must_coverage
    elif has_nice:          # job lists no must-have skills
        value = nice_coverage
    else:                   # job lists no skills at all -> use the AI model
        value = model_probability
    return round(value * 100, 2)


FEATURE_LABELS = {
    "must_skill_coverage": "Must-have skills match",
    "nice_skill_coverage": "Nice-to-have skills match",
    "skill_jaccard": "Overall skill overlap",
    "seniority_diff": "Seniority level gap",
    "seniority_match": "Seniority match",
    "industry_match": "Industry match",
    "role_similarity": "Role / title similarity",
    "text_similarity": "Description similarity",
}


def _pct(value: float) -> float:
    return round(float(value) * 100, 2)


def build_features_from_raw(job: dict, resume: dict) -> dict:
    """Computes the 8 model features from raw job/resume dicts. Must match the
    feature engineering used at training time exactly."""
    job_must = {normalize_skill(s) for s in parse_skills(job["must_have_skills"])}
    job_nice = {normalize_skill(s) for s in parse_skills(job["nice_to_have_skills"])}
    res_skills = {normalize_skill(s) for s in parse_skills(resume["skills"])}

    must_coverage = len(job_must & res_skills) / len(job_must) if job_must else 0
    nice_coverage = len(job_nice & res_skills) / len(job_nice) if job_nice else 0

    union = (job_must | job_nice) | res_skills
    jaccard = len((job_must | job_nice) & res_skills) / len(union) if union else 0

    job_sen = seniority_map.get(job["seniority"])
    res_sen = seniority_map.get(resume["seniority"])
    sen_diff = abs(job_sen - res_sen) if job_sen is not None and res_sen is not None else 2
    sen_match = int(job["seniority"] == resume["seniority"])

    industry_match = int(job["industry"] == resume["industry"])

    job_words = get_tokens(job["job_title"])
    res_words = get_tokens(resume["role"])
    role_sim = len(job_words & res_words) / len(job_words | res_words) if (job_words | res_words) else 0

    job_text = f"{job.get('description', '')} {job.get('responsibilities', '')} {job.get('requirements', '')}"
    resume_text = f"{resume.get('summary', '')} {resume.get('experience_bullets', '')}"
    job_vec = vectorizer.transform([job_text])
    res_vec = vectorizer.transform([resume_text])
    text_sim = float(job_vec.multiply(res_vec).sum())

    return {
        "must_skill_coverage": must_coverage,
        "nice_skill_coverage": nice_coverage,
        "skill_jaccard": jaccard,
        "seniority_diff": sen_diff,
        "seniority_match": sen_match,
        "industry_match": industry_match,
        "role_similarity": role_sim,
        "text_similarity": text_sim,
    }


def score_candidate(job: dict, resume: dict, explain: bool = False) -> dict:
    """Full pipeline: raw job + raw resume -> match %, Match / No Match, (optional) explanation.

    Match / No Match comes from the match % (must-have skills matched):
    >= MATCH_THRESHOLD_PCT (default 40%) is a Match."""
    feats = build_features_from_raw(job, resume)
    X_new = pd.DataFrame([feats], columns=feature_columns)

    probability = float(model.predict_proba(X_new)[0][1])

    has_must = bool(parse_skills(job["must_have_skills"]))
    has_nice = bool(parse_skills(job["nice_to_have_skills"]))
    match_pct = compute_match_pct(feats["must_skill_coverage"], feats["nice_skill_coverage"],
                                  has_must, has_nice, probability)

    output = {
        "result": match_result_from_pct(match_pct),
        # Same field name the backend already reads; it now holds the match %.
        "match_probability_pct": match_pct,
        # Raw XGBoost probability (internal use, e.g. tie-break in ranking)
        "ai_model_probability_pct": _pct(probability),
        # Every value shown as a percentage (0-100) instead of 0-1
        "features": {
            "must_have_skills_match_pct": _pct(feats["must_skill_coverage"]),
            "nice_to_have_skills_match_pct": _pct(feats["nice_skill_coverage"]),
            "overall_skill_overlap_pct": _pct(feats["skill_jaccard"]),
            "role_title_similarity_pct": _pct(feats["role_similarity"]),
            "description_similarity_pct": _pct(feats["text_similarity"]),
            "seniority_match_pct": 100.0 if feats["seniority_match"] else 0.0,
            "industry_match_pct": 100.0 if feats["industry_match"] else 0.0,
            "seniority_level_gap": feats["seniority_diff"],   # number of levels apart (0-2)
            "ai_model_probability_pct": _pct(probability),
            "match_threshold_pct": MATCH_THRESHOLD_PCT,
        },
    }

    if explain:
        import shap
        explainer = shap.TreeExplainer(model)
        contributions = explainer.shap_values(X_new)[0]
        # Shown as % instead of raw SHAP points: how much of the AI model's
        # decision each factor accounts for, and whether it pushed the score up or down.
        total = sum(abs(float(c)) for c in contributions) or 1.0
        output["explanation"] = [
            {
                "feature": FEATURE_LABELS.get(f, f),
                "impact_pct": round(abs(float(c)) / total * 100, 1),
                "effect": "increases match" if c > 0 else "decreases match",
            }
            for f, c in sorted(zip(feature_columns, contributions), key=lambda x: abs(x[1]), reverse=True)
        ]

    return output
