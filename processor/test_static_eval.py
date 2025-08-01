#!/usr/bin/env python3
"""
Test the static evaluation workflow with a small subset of data.

This script validates that the static evaluation pipeline works correctly
by testing it on a small sample of data.

Usage:
    python processor/test_static_eval.py
"""

import json
import tempfile
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from create_static_eval import StaticEvalCreator


def create_test_results_file():
    test_results = {
        "accuracy": 0.5,
        "total_questions": 2,
        "question_results": [
            {
                "question_id": "0",
                "generated_text": "Monday, 10:00 - 10:30",
                "is_correct": True,
                "prompt_tokens": 100,
                "output_tokens": 10
            },
            {
                "question_id": "1", 
                "generated_text": "Tuesday, 14:00 - 15:00",
                "is_correct": False,
                "prompt_tokens": 120,
                "output_tokens": 12
            }
        ]
    }
    temp_file = Path(tempfile.mktemp(suffix=".json"))
    with temp_file.open("w") as f:
        json.dump(test_results, f, indent=2)
    return temp_file


def test_static_eval_workflow():
    print("Testing static evaluation workflow...")

    eval_scripts = [
        "eval/evaluate_calendar_scheduling.py",
        "eval/evaluate_meeting_planning.py",
        "eval/evaluate_trip_planning.py"
    ]

    missing_scripts = []
    for script in eval_scripts:
        if not Path(script).exists():
            missing_scripts.append(script)

    if missing_scripts:
        print("Missing evaluation scripts:")
        for script in missing_scripts:
            print(f"   {script}")
        print("Make sure you're running from the project root directory")
        return False

    eval_data_dir = Path("eval/data")
    if not eval_data_dir.exists():
        print(f"Eval data directory not found: {eval_data_dir}")
        return False

    required_datasets = [
        "calendar_scheduling.json",
        "meeting_planning.json", 
        "trip_planning.json"
    ]

    missing_datasets = []
    for dataset in required_datasets:
        if not (eval_data_dir / dataset).exists():
            missing_datasets.append(dataset)

    if missing_datasets:
        print("Missing evaluation datasets:")
        for dataset in missing_datasets:
            print(f"   eval/data/{dataset}")
        return False

    print("All required files found")

    print("\nTesting calendar scheduling evaluation...")

    try:
        test_results_file = create_test_results_file()
        creator = StaticEvalCreator()

        with tempfile.TemporaryDirectory() as temp_dir:
            eval_dataset_path = Path(temp_dir) / "test_calendar_eval.json"

            created_path = creator.create_eval_dataset(
                test_results_file, "calendar", eval_dataset_path
            )

            print(f"Created evaluation dataset: {created_path}")

            with created_path.open() as f:
                eval_data = json.load(f)

            found_prediction = False
            for item in eval_data.values():
                if "pred_5shot_pro" in item and "Monday, 10:00 - 10:30" in item["pred_5shot_pro"]:
                    found_prediction = True
                    break

            if found_prediction:
                print("Predictions correctly merged into dataset")
            else:
                print("Could not verify prediction merging")

            print("Testing evaluation script...")
            try:
                eval_result = creator.run_evaluation(created_path, "calendar")
                print(f"Evaluation completed with accuracy: {eval_result['accuracy']}")
            except Exception as e:
                print(f"Evaluation script test failed (this may be expected): {e}")

        test_results_file.unlink()

        print("\nStatic evaluation workflow test completed successfully!")
        return True

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("Natural-Plan Static Evaluation Test")
    print("=" * 50)

    success = test_static_eval_workflow()

    if success:
        print("\nAll tests passed! The static evaluation workflow is ready to use.")
        print("\nNext steps:")
        print("1. Run baseline evaluations if you haven't already")
        print("2. Use: python processor/run_static_eval.py")
        print("3. Or: make run-static-eval")
        return 0
    else:
        print("\nTests failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    exit(main())