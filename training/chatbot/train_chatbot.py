"""
Train the candidate chatbot on an ORIGINAL public dataset.

Dataset: Career Guidance QA Dataset (Pradeep, 2025)
         https://huggingface.co/datasets/Pradeep016/career-guidance-qa-dataset
         License: CC BY 4.0  |  ~1,620 rows  |  columns: role, question, answer

Model:   two steps (see career_model.py)
         1. Topic classifier - TF-IDF (word + char n-grams) + Logistic Regression,
            trained on the dataset questions with the job role masked -> 10 topics
         2. Role detector    - finds which of the 54 dataset job roles is asked about
         Answer = dataset answer for that role and topic

Usage (from the EMS project root):
    python training/chatbot/train_chatbot.py
    python training/chatbot/train_chatbot.py --test training/chatbot/tests/holdout_questions.csv

Output:
    app/models/candidate_chatbot.joblib       (loaded by app/chatbot.py)
    app/models/chatbot_training_report.json

Train with the scikit-learn / pandas versions pinned in requirements.txt,
otherwise the saved model may fail to load in the service.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict

BASE_DIR = Path(__file__).resolve().parent          # training/chatbot/
PROJECT_ROOT = BASE_DIR.parent.parent                # EMS/
sys.path.insert(0, str(PROJECT_ROOT))                # so "app." imports work

from app.career_model import TOPIC_BY_POSITION, CareerModel   # noqa: E402
from app.chatbot_guardrails import is_restricted_topic          # noqa: E402
DEFAULT_CSV = BASE_DIR / "dataset" / "career_guidance_qa.csv"
HF_DATASET = "Pradeep016/career-guidance-qa-dataset"
SEED = 42


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

def download_dataset(target: Path) -> None:
    """Download the original dataset from Hugging Face (needs internet once)."""
    from datasets import load_dataset

    print(f"Downloading {HF_DATASET} ...")
    ds = load_dataset(HF_DATASET, split="train")
    target.parent.mkdir(parents=True, exist_ok=True)
    ds.to_pandas().to_csv(target, index=False)
    print(f"Saved to {target}")


def load_and_clean(path: Path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = {"role", "question", "answer"} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}. Found: {list(df.columns)}")

    stats = {"raw_rows": int(len(df))}

    df = df[["role", "question", "answer"]].dropna()
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
    df = df[(df.question.str.len() > 3) & (df.answer.str.len() > 3)]
    df = df.drop_duplicates(subset=["role", "question"])
    stats["after_cleaning"] = int(len(df))

    # Candidate-only: drop rows about the platform's internal roles
    # (recruiter, HR, hiring manager, admin, interviewer, employee).
    restricted = df.role.apply(is_restricted_topic) | df.question.apply(is_restricted_topic)
    stats["removed_restricted_roles"] = int(restricted.sum())
    stats["removed_role_names"] = sorted(df[restricted].role.unique().tolist())
    df = df[~restricted].reset_index(drop=True)
    stats["final_rows"] = int(len(df))
    stats["job_roles"] = int(df.role.nunique())

    # Topic label from the row position within each role (dataset uses the
    # same 30-question order for every role: 10 topics x 3 phrasings).
    df["position"] = df.groupby("role").cumcount()
    bad = df.groupby("role").size()
    bad = bad[bad != len(TOPIC_BY_POSITION)]
    if len(bad):
        raise ValueError(f"Expected {len(TOPIC_BY_POSITION)} rows per role, got: {bad.to_dict()}")
    df["topic"] = df.position.map(lambda p: TOPIC_BY_POSITION[p])

    return df, stats


def evaluate(df, test_csv: Path):
    model_for_masking = CareerModel()

    # 1. Topic classifier on UNSEEN job roles: 6-fold, grouped by role, so the
    #    roles in each test fold were never seen in training.
    X = [model_for_masking.topic_features(CareerModel.mask_role(q, r))
         for r, q in zip(df.role, df.question)]
    pred = cross_val_predict(CareerModel().topic_clf, X, df.topic, groups=df.role,
                             cv=GroupKFold(n_splits=6))
    topic_acc = float(np.mean(pred == df.topic.values))

    full = CareerModel().fit(df.role, df.question, df.answer, df.topic)

    # 2. Role detection on every dataset question
    role_acc = float(np.mean([full.detect_role(q)[0] == r for r, q in zip(df.role, df.question)]))

    # 3. Hand-written candidate-style questions (evaluation only, never trained on)
    user = {}
    if test_csv.exists():
        t = pd.read_csv(test_csv)
        rows = []
        for q, er, et in zip(t.question, t.expected_role, t.expected_topic):
            r, name = full.detect_role(q)
            tp, conf = full.predict_topic(q, name)
            rows.append((r == er, tp == et))
        rows = np.array(rows)
        user = {"user_questions": int(len(rows)),
                "user_role_accuracy": round(float(rows[:, 0].mean()), 4),
                "user_topic_accuracy": round(float(rows[:, 1].mean()), 4),
                "user_both_correct": round(float(rows.all(axis=1).mean()), 4)}

    return {"topic_accuracy_unseen_roles": round(topic_acc, 4),
            "role_detection_accuracy": round(role_acc, 4), **user}, full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_CSV))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "app" / "models" / "candidate_chatbot.joblib"))
    parser.add_argument("--min_topic_confidence", type=float, default=0.35,
                        help="below this the bot asks the user to rephrase")
    parser.add_argument("--test", default=str(BASE_DIR / "tests" / "user_test_questions.csv"))
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        download_dataset(data_path)

    df, stats = load_and_clean(data_path)
    print(json.dumps(stats, indent=2))

    scores, model = evaluate(df, Path(args.test))
    print(json.dumps(scores, indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "min_topic_confidence": args.min_topic_confidence,
        "roles": model.roles,
        "dataset": HF_DATASET,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": __import__("sklearn").__version__,
    }, out)

    report = dict(stats, **scores, dataset=HF_DATASET, license="CC BY 4.0")
    (out.parent / "chatbot_training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Model saved to {out}")


if __name__ == "__main__":
    main()
