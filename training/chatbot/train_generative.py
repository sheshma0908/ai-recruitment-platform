"""
Optional: fine-tune google/flan-t5-small on the Career Guidance QA Dataset
so answers are generated instead of returned word for word.

Chatbot.py only uses a generated answer when it stays close to the real
dataset answer, so facts (skills, salary ranges) are not invented.

Usage (from the EMS project root):
    pip install -r training/chatbot/requirements-generative.txt
    python training/chatbot/train_generative.py     # ~15-30 min on CPU, 2-4 min on GPU

Output:
    app/models/flan_t5_candidate/            model + tokenizer (picked up by app/chatbot.py)
    app/models/flan_t5_candidate/eval.json   validation score and sample answers
"""

import argparse
import json
import random
import time
from difflib import SequenceMatcher
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from train_chatbot import DEFAULT_CSV, PROJECT_ROOT, load_and_clean

BASE_DIR = Path(__file__).resolve().parent
SEED = 42


def format_input(role: str, question: str) -> str:
    """Must match _generate() in Chatbot.py."""
    return f"Answer this career question. job role: {role} | question: {question.strip()}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DEFAULT_CSV))
    p.add_argument("--base_model", default="google/flan-t5-small")
    p.add_argument("--out", default=str(PROJECT_ROOT / "app" / "models" / "flan_t5_candidate"))
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    args = p.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df, _ = load_and_clean(Path(args.data))
    rows = df.to_dict("records")
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * 0.1))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    print(f"Train {len(train_rows)}  Validation {len(val_rows)}  Device {device}")

    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model).to(device)

    def collate(batch):
        enc = tok([format_input(r["role"], r["question"]) for r in batch],
                  max_length=96, truncation=True, padding=True, return_tensors="pt")
        labels = tok([r["answer"] for r in batch], max_length=200,
                     truncation=True, padding=True, return_tensors="pt").input_ids
        labels[labels == tok.pad_token_id] = -100
        enc["labels"] = labels
        return enc

    loader = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total = len(loader) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / max(total, 1)))

    @torch.no_grad()
    def evaluate():
        model.eval()
        scores, examples = [], []
        for i in range(0, min(len(val_rows), 120), 16):
            chunk = val_rows[i:i + 16]
            enc = tok([format_input(r["role"], r["question"]) for r in chunk],
                      max_length=96, truncation=True, padding=True, return_tensors="pt").to(device)
            preds = tok.batch_decode(model.generate(**enc, max_new_tokens=200, num_beams=2),
                                     skip_special_tokens=True)
            for r, pr in zip(chunk, preds):
                scores.append(SequenceMatcher(None, pr.split(), r["answer"].split()).ratio())
                if len(examples) < 6:
                    examples.append({"role": r["role"], "question": r["question"],
                                     "expected": r["answer"], "generated": pr})
        model.train()
        return sum(scores) / max(len(scores), 1), examples

    best, out = -1.0, Path(args.out)
    model.train()
    for epoch in range(1, args.epochs + 1):
        t0, loss_sum = time.time(), 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            loss_sum += loss.item()
        score, examples = evaluate()
        print(f"Epoch {epoch}/{args.epochs} loss={loss_sum / len(loader):.4f} "
              f"val_similarity={score:.3f} ({time.time() - t0:.0f}s)")
        if score > best:
            best = score
            out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out)
            tok.save_pretrained(out)
            (out / "eval.json").write_text(json.dumps(
                {"base_model": args.base_model, "best_epoch": epoch,
                 "val_similarity": round(score, 4), "examples": examples}, indent=2), encoding="utf-8")

    print(f"Best validation similarity {best:.3f}. Saved to {out}")


if __name__ == "__main__":
    main()
