#!/usr/bin/env python3
"""Merge Natural-Plan model predictions into the original dataset JSON.

This utility replaces the pred_5shot_pro field in the original dataset
with your model's predictions, creating datasets compatible with the
Google evaluation scripts.

Example
-------
# Basic usage - replace predictions
python processor/planner_predictions.py \
        --results results/base/14b/meeting_20240731_120000/results_*.json \
        --dataset eval/data/meeting_planning.json \
        --out meeting_14b_eval.json

# For static evaluation workflow  
python processor/planner_predictions.py \
        --results results/base/14b/meeting_20240731_120000/results_*.json \
        --dataset eval/data/meeting_planning.json \
        --out eval_datasets/meeting_14b_eval.json \
        --run-eval
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, Any


def _args():
    p = argparse.ArgumentParser(description="Merge Natural-Plan predictions into dataset JSON")
    p.add_argument("--results", required=True, help="results_<run>.json produced by evaluator")
    p.add_argument("--dataset", required=True, help="Original Natural-Plan dataset json file")
    p.add_argument("--out", required=True, help="Destination json with predictions inserted")
    p.add_argument("--field", default="pred_5shot_pro", help="JSON key to store predictions (use pred_5shot_pro for Google eval scripts)")
    p.add_argument("--run-eval", action="store_true", help="Also run the corresponding Google evaluation script")
    p.add_argument("--task", choices=["meeting", "calendar", "trip"], help="Task name (required if --run-eval is used)")
    return p.parse_args()


def _load_predictions(results_path: pathlib.Path) -> Dict[str, str]:
    with results_path.open() as f:
        data = json.load(f)
    preds: Dict[str, str] = {}
    for item in data.get("question_results", []):
        qid = str(item.get("question_id"))
        if not qid:
            continue
        preds[qid] = item.get("generated_text", "")
    if not preds:
        raise ValueError("No predictions found in results file")
    return preds


def _run_evaluation(dataset_path: pathlib.Path, task: str) -> None:
    """Run the Google evaluation script on the merged dataset"""
    import subprocess
    
    eval_script_map = {
        "meeting": "eval/evaluate_meeting_planning.py",
        "calendar": "eval/evaluate_calendar_scheduling.py",
        "trip": "eval/evaluate_trip_planning.py"
    }
    
    eval_script = eval_script_map.get(task)
    if not eval_script or not pathlib.Path(eval_script).exists():
        print(f"[ERROR] Evaluation script not found: {eval_script}", file=sys.stderr)
        return
    
    print(f"🚀 Running evaluation: {eval_script}")
    
    try:
        result = subprocess.run([
            sys.executable, eval_script,
            f"--data_path={dataset_path}"
        ], check=True)
        
        print("✅ Evaluation completed successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed: {e}", file=sys.stderr)


def main() -> None:
    args = _args()
    preds = _load_predictions(pathlib.Path(args.results))

    with pathlib.Path(args.dataset).open() as f:
        dataset: Dict[str, Any] = json.load(f)

    missing = []
    for qid, record in dataset.items():
        if qid in preds:
            record[args.field] = preds[qid]
        else:
            missing.append(qid)

    if missing:
        print(f"[WARN] {len(missing)} dataset items had no prediction", file=sys.stderr)

    # Ensure output directory exists
    output_path = pathlib.Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Merged dataset written → {args.out}")

    # Run evaluation if requested
    if args.run_eval:
        if not args.task:
            print("[ERROR] --task required when using --run-eval", file=sys.stderr)
            return
        _run_evaluation(output_path, args.task)


if __name__ == "__main__":
    main() 