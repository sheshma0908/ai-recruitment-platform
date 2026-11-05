"""
Candidate Decision Recommendation -- supports the Hiring Manager's
"Approve/Reject Candidates" permission (Module 23 area of the report).

IMPORTANT: this produces a SUGGESTED recommendation only, combining the AI match
score with the interviewer's structured feedback. It does NOT store or make the
final decision -- the Hiring Manager approves/rejects, and the backend team's
database is what persists that decision. This module is intentionally stateless,
same as every other endpoint in this service.
"""

# Weights for blending AI match score with human interview feedback.
# Interview feedback is weighted higher (60%) since it's a direct human judgment
# of the candidate after actually meeting them, versus the match score (40%)
# which only looks at resume/JD text before any interview happened.
MATCH_WEIGHT = 0.40
INTERVIEW_WEIGHT = 0.60

RECOMMEND_THRESHOLD = 75.0   # combined score >= this -> "Recommended"
HOLD_THRESHOLD = 50.0        # combined score >= this (but below RECOMMEND) -> "Hold"
# below HOLD_THRESHOLD -> "Not Recommended"


def compute_interview_average(feedback: dict) -> float:
    """feedback values are expected on a 0-10 scale, matching the report's
    example (Technical Skills 9/10, Communication 8/10, etc.)."""
    scores = [v for v in feedback.values() if v is not None]
    return sum(scores) / len(scores) if scores else 0.0


def recommend_candidate_decision(match_probability_pct: float, interview_feedback: dict) -> dict:
    interview_avg = compute_interview_average(interview_feedback)
    interview_avg_pct = interview_avg * 10  # convert 0-10 scale to 0-100 scale

    combined_score = (match_probability_pct * MATCH_WEIGHT) + (interview_avg_pct * INTERVIEW_WEIGHT)

    if combined_score >= RECOMMEND_THRESHOLD:
        recommendation = "Recommended"
    elif combined_score >= HOLD_THRESHOLD:
        recommendation = "Hold"
    else:
        recommendation = "Not Recommended"

    # Flag the weakest interview dimension, if any, so the HM sees the specific
    # concern rather than just a number.
    weakest = None
    if interview_feedback:
        valid = {k: v for k, v in interview_feedback.items() if v is not None}
        if valid:
            weakest_key = min(valid, key=valid.get)
            if valid[weakest_key] <= 5:
                weakest = weakest_key

    rationale_parts = [
        f"AI match score: {match_probability_pct:.1f}%.",
        f"Average interview feedback: {interview_avg:.1f}/10.",
    ]
    if weakest:
        rationale_parts.append(f"Lowest-rated area: {weakest} ({interview_feedback[weakest]}/10).")

    return {
        "match_probability_pct": round(match_probability_pct, 2),
        "interview_average_score": round(interview_avg, 2),
        "combined_score_pct": round(combined_score, 2),
        "recommendation": recommendation,
        "rationale": " ".join(rationale_parts),
        "note": "This is a SUGGESTED recommendation for the Hiring Manager's review, "
                "not an automatic decision. The Hiring Manager makes the final "
                "approve/reject call.",
    }
