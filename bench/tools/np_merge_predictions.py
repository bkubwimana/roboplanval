#!/usr/bin/env python
"""Merge Natural-Plan model predictions into the original dataset JSON.

This is a shorter alias for the earlier dump script.  Run on a host machine
(after collecting `results_<run>.json` from the edge device) to inject the
`generated_text` back into the dataset so that the official evaluation scripts
can be used unchanged.

Example
-------
python bench/tools/np_merge_predictions.py \
        --results results_20240729_153000.json \
        --dataset eval/data/trip_planning.json \
        --out trip_planning_with_preds.json
"""
from __future__ import annotations
import argparse, json, pathlib, sys
from typing import Dict, Any

def _args():
    p = argparse.ArgumentParser(description="Merge Natural-Plan predictions into dataset JSON")
    p.add_argument("--results", required=True, help="results_<run>.json produced by evaluator")
    p.add_argument("--dataset", required=True, help="Original Natural-Plan dataset json file")
    p.add_argument("--out", required=True, help="Destination json with predictions inserted")
    p.add_argument("--field", default="pred_5shot_local", help="JSON key to store predictions")
    return p.parse_args()

def _load_predictions(results_path: pathlib.Path) -> Dict[str, str]:
    with results_path.open() as f: data = json.load(f)
    preds: Dict[str, str] = {}
    for item in data.get("question_results", []):
        qid = str(item.get("question_id"))
        if qid is None: continue
        preds[qid] = item.get("generated_text", "")
    if not preds: raise ValueError("No predictions found in results file")
    return preds

def main():
    a = _args()
    preds = _load_predictions(pathlib.Path(a.results))
    with pathlib.Path(a.dataset).open() as f:
        dataset: Dict[str, Any] = json.load(f)
    missing=[]
    for qid, rec in dataset.items():
        if qid in preds: rec[a.field] = preds[qid]
        else: missing.append(qid)
    if missing:
        print(f"[WARN] {len(missing)} dataset items had no prediction", file=sys.stderr)
    with pathlib.Path(a.out).open("w") as f: json.dump(dataset, f, indent=2)
    print(f"Merged dataset written → {a.out}")
if __name__ == "__main__": main() 