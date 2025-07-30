#!/usr/bin/env python
"""Merge Natural-Plan model predictions into the original dataset JSON.

Renamed utility (was ``np_merge_predictions.py``) so that the command is
short and clearly associated with the *planner* benchmark.

Example
-------
python bench/tools/planner_predictions.py \
        --results results_20240729_153000.json \
        --dataset eval/data/trip_planning.json \
        --out trip_planning_with_preds.json
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
    p.add_argument("--field", default="pred_5shot_local", help="JSON key to store predictions")
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

    with pathlib.Path(args.out).open("w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Merged dataset written → {args.out}")


if __name__ == "__main__":
    main() 