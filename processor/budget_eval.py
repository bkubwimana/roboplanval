#!/usr/bin/env python3
"""Aggregate Natural-Plan *budget-limited* evaluation 

"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple


import subprocess, tempfile

EVAL_DATA_MAP = {
    "meeting": "eval/data/meeting_planning.json",
    "calendar": "eval/data/calendar_scheduling.json",
    "trip": "eval/data/trip_planning.json",
}


import re
NOISE_PATTERNS = [
    # Conciseness instructions
    re.compile(r"^Be concise and direct\.", re.I),
    re.compile(r"Okay, so I'm trying to figure out the best way to meet as many friends as possible in San Francisco today", re.I),
    re.compile(r"the best way to meet as many friends as possible in San Francisco today", re.I),
    re.compile(r"Each solution must be concise", re.I),
    re.compile(r"Each solution should be concise", re.I),
    re.compile(r"So, each solution should be concise", re.I),
    re.compile(r"Each solution must be concise and fit within the token limit", re.I),
    re.compile(r"Each step is a separate line, so each line is a token", re.I),
    re.compile(r"followed by the steps, each step being a concise sentence, and the entire thing under 512 tokens", re.I),
    re.compile(r"Keep your response concise", re.I),
    re.compile(r"Be direct and concise", re.I),
    re.compile(r"Please be concise", re.I),
    re.compile(r"You must be concise and direct", re.I),
    re.compile(r"You must be concise and clear", re.I),
    re.compile(r"You must limit your answer to exactly", re.I),
    re.compile(r"You must limit your answer to", re.I),
    re.compile(r"You can use any combination", re.I),
    re.compile(r"You can use any abbreviations", re.I),
    re.compile(r"Use only standard abbreviations", re.I),
    re.compile(r"You must be careful with your", re.I),
    
    # Solution format instructions
    re.compile(r"So, the solution should be", re.I),
    re.compile(r"Each solution is a separate", re.I),
    re.compile(r"Each problem is", re.I),
    re.compile(r"Each solution is a", re.I),
    re.compile(r"and follow the same solution format", re.I),
    re.compile(r"follow the same format", re.I),
    
    # Thinking/planning noise
    re.compile(r"Alright, so I'm trying to figure out", re.I),
    re.compile(r"Alright, I need to figure out", re.I),
    re.compile(r"Alright, let's tackle this problem", re.I),
    re.compile(r"Okay, so I'm trying to figure out", re.I),
    re.compile(r"First, I need to", re.I),
    re.compile(r"First, I'll", re.I),
    re.compile(r"First, I should", re.I),
    re.compile(r"Now, let's", re.I),
    re.compile(r"Now, I need to", re.I),
    re.compile(r"Wait, no, the user", re.I),
    re.compile(r"Wait, but", re.I),
    re.compile(r"But wait, the user's instruction says", re.I),
    re.compile(r"I start at", re.I),
    re.compile(r"I arrive at", re.I),
    re.compile(r"I also have", re.I),
    re.compile(r"I should probably", re.I),
    
    # Format markers
    re.compile(r"```", re.I),
    re.compile(r"^\"$", re.I),  # standalone quote marks
    
    # Template fragments
    re.compile(r"', followed by", re.I),
    re.compile(r"', then the", re.I),
    re.compile(r"', and", re.I),
    re.compile(r"and then the plan", re.I),
    re.compile(r"Start at Downtown", re.I),
    re.compile(r"followed by the schedule", re.I),
    re.compile(r"followed by the plan", re.I),
    re.compile(r"followed by the steps", re.I),
    re.compile(r"ollowed by the steps", re.I),  # catches "followed" with missing "f"
    re.compile(r"then the steps", re.I),
    re.compile(r"and following the same format", re.I),
    re.compile(r"</think>", re.I),
    re.compile(r"SOLUTION:", re.I),
    re.compile(r"Now, the problem you need to solve is:", re.I),
    re.compile(r"You are visiting San Francisco for the day", re.I),
    re.compile(r"' and end with", re.I),
    re.compile(r"Wait, the constraints are:", re.I),
    re.compile(r"Travel distances \\(in minutes\\):", re.I),
    re.compile(r"You start at", re.I),
    re.compile(r"You arrive at", re.I),
    re.compile(r"Use the", re.I),
    re.compile(r"You'll be able to", re.I),
]

def _clean_prediction(txt: str, task: str) -> str:
    """Remove instruction noise from generated text, keeping everything else intact."""
    # First, do character-level cleaning within lines
    cleaned_text = txt
    for pattern in NOISE_PATTERNS:
        cleaned_text = pattern.sub('', cleaned_text)
    
    # Then clean up extra whitespace and empty lines
    lines = [line.strip() for line in cleaned_text.splitlines()]
    lines = [l for l in lines if l]  # Remove empty lines
    
    return "\n".join(lines)

def _prepare_results_json(orig: Path, clean: bool) -> Path:
    if not clean:
        return orig
    data = json.loads(orig.read_text())
    for q in data.get("question_results", []):
        q["generated_text"] = _clean_prediction(q.get("generated_text", ""), data.get("subject", ""))
    tmp = Path(tempfile.mkstemp(suffix="_clean.json")[1])
    tmp.write_text(json.dumps(data))
    return tmp

def _run_google_eval(results_json: Path, task: str, clean: bool) -> float:
    """Run Google's eval script via planner_predictions.py and return accuracy."""
    dataset_path = Path(EVAL_DATA_MAP[task])
    with tempfile.NamedTemporaryFile(suffix="_eval.json", delete=False) as tmp:
        out_dataset = Path(tmp.name)
    cmd = [
        "python",
        "processor/planner_predictions.py",
        "--results", str(results_json),
        "--dataset", str(dataset_path),
        "--out", str(out_dataset),
        "--run-eval",
        "--task", task,
    ]
    prepared_json = _prepare_results_json(results_json, clean)
    cmd[cmd.index("--results")+1] = str(prepared_json)
    completed = subprocess.run(cmd, capture_output=True, text=True)
    # Echo Google's evaluation output so the user can inspect errors / warnings
    print(completed.stdout)
    acc = None
    for line in completed.stdout.splitlines():
        if "Overall solve rate" in line or "Accuracy for all" in line or line.startswith("EM_Accuracy"):
            try:
                acc = float(line.split(":" if ":" in line else " ")[-1].strip())
            except ValueError:
                continue
    return acc if acc is not None else 0.0


def _scan_single_json(path: Path, task: str, rerun: bool, clean: bool) -> Tuple[float, float]:
    """Return (accuracy, avg_tokens_per_q) from results JSON."""
    with path.open() as f:
        data = json.load(f)
    acc = float(data.get("accuracy", 0.0))  # will be overwritten if rerun
    avg_tok = float(data.get("avg_tokens_per_q", 0.0))
    # Optionally re-run Google script for sanity check
    if rerun:
        try:
            acc = _run_google_eval(path, task, clean)
        except Exception:
            pass
    return acc, avg_tok


def scan_budget_results(root: Path, out_csv: Path, args_rerun: bool, clean: bool) -> None:
    rows: List[List] = []

    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        model = model_dir.name  # e.g. 1.5b / 8b / 14b

        for budget_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            try:
                token_budget = int(budget_dir.name)  # 128 / 256 / 512 …
            except ValueError:
                continue  # ignore unexpected folders

            for task_run in sorted(p for p in budget_dir.iterdir() if p.is_dir()):
                task = task_run.name.split("_")[0]  # calendar_YYYY → calendar

                json_files = list(task_run.glob("results_*.json"))
                if not json_files:
                    continue
                # assume only one per dir; take first if multiple
                acc, avg_tok = _scan_single_json(json_files[0], task, args_rerun, clean)
                # Fallback: compute avg tokens if missing or zero
                if avg_tok == 0.0:
                    try:
                        with json_files[0].open() as f:
                            j = json.load(f)
                        qrs = j.get("question_results", [])
                        if qrs:
                            tot = sum(q.get("output_tokens", 0) for q in qrs)
                            avg_tok = tot / len(qrs)
                    except Exception:
                        pass
                avg_tok_int = int(round(avg_tok))
                rows.append([model, token_budget, task, acc, avg_tok_int])

    # sort nicely: model → budget → task
    rows.sort(key=lambda r: (r[0], r[1], r[2]))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "token_budget", "task", "accuracy", "avg_tokens_per_q"])
        writer.writerows(rows)

    print(f"✅ Budget summary written → {out_csv}  (rows: {len(rows)})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate budget-limited evaluation results into a CSV summary")
    parser.add_argument("--results-dir", default="results/budget", help="Directory containing budget run folders")
    parser.add_argument("--out", default="processor/budget_eval_summary.csv", help="CSV file to write")
    parser.add_argument("--rerun", action="store_true", help="Rerun Google evaluation scripts for each budget result (sanity check)")
    parser.add_argument("--clean", action="store_true", help="Clean prediction noise before re-running evaluator")
    args = parser.parse_args()

    root = Path(args.results_dir)
    if not root.exists():
        parser.error(f"results-dir not found: {root}")
    scan_budget_results(root, Path(args.out), args.rerun, args.clean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
