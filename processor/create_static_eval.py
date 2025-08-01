#!/usr/bin/env python3
"""
Create static evaluation datasets from Natural-Plan evaluation results.

This script takes your model evaluation results and creates dataset files
compatible with the Google evaluation scripts (evaluate_*_planning.py).

Usage:
    python processor/create_static_eval.py --results-dir results/
    python processor/create_static_eval.py --results-file results/base/14b/meeting_20240731_120000/results_*.json
    python processor/create_static_eval.py --task meeting --model 14b
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import shutil
from analyze_results import ResultsAnalyzer


class StaticEvalCreator:
    """Creates static evaluation datasets from model results"""
    
    def __init__(self, eval_data_dir: Path = None):
        self.eval_data_dir = eval_data_dir or Path("eval/data")
        
    def create_eval_dataset(self, results_json_path: Path, task: str, 
                          output_path: Path = None) -> Path:
        """
        Create evaluation dataset by replacing pred_5shot_pro with model predictions
        
        Args:
            results_json_path: Path to results JSON from evaluation
            task: Task name (meeting, calendar, trip)
            output_path: Where to save the eval dataset (optional)
            
        Returns:
            Path to created evaluation dataset
        """
        # Load original dataset
        original_dataset_path = self.eval_data_dir / f"{task}_{'scheduling' if task == 'calendar' else 'planning'}.json"
        
        if not original_dataset_path.exists():
            raise FileNotFoundError(f"Original dataset not found: {original_dataset_path}")
        
        print(f"📖 Loading original dataset: {original_dataset_path}")
        with original_dataset_path.open() as f:
            original_data = json.load(f)
        
        # Load model results
        print(f"📊 Loading model results: {results_json_path}")
        with results_json_path.open() as f:
            results_data = json.load(f)
        
        # Create mapping of question_id -> generated_text
        predictions = {}
        for result in results_data.get("question_results", []):
            question_id = str(result.get("question_id", ""))
            generated_text = result.get("generated_text", "")
            if question_id and generated_text:
                predictions[question_id] = generated_text
        
        print(f"🔄 Found {len(predictions)} predictions to merge")
        
        # Clone dataset and replace predictions
        eval_dataset = {}
        matched_count = 0
        
        for qid, item in original_data.items():
            # Clone the original item
            eval_item = dict(item)
            
            # Replace pred_5shot_pro with our model's prediction
            if qid in predictions:
                eval_item["pred_5shot_pro"] = predictions[qid]
                matched_count += 1
            else:
                # Keep original prediction as fallback
                print(f"⚠️  No prediction found for question {qid}, keeping original")
        
            eval_dataset[qid] = eval_item
        
        print(f"✅ Replaced {matched_count}/{len(original_data)} predictions")
        
        # Save evaluation dataset
        if output_path is None:
            # Extract model info from results path for naming
            parts = results_json_path.parts
            model_info = "unknown"
            timestamp = "unknown"
            
            for part in parts:
                if part in ["14b", "8b", "1.5b"]:
                    model_info = part
                if "_" in part and len(part) == 15:  # YYYYMMDD_HHMMSS format
                    timestamp = part
            
            output_path = Path(f"processor/eval_datasets/{task}_{model_info}_{timestamp}.json")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with output_path.open("w") as f:
            json.dump(eval_dataset, f, indent=2)
        
        print(f"💾 Evaluation dataset saved: {output_path}")
        return output_path
    
    def run_evaluation(self, eval_dataset_path: Path, task: str) -> Dict[str, Any]:
        """
        Run the Google evaluation script on the created dataset
        
        Args:
            eval_dataset_path: Path to evaluation dataset JSON
            task: Task name (meeting, calendar, trip)
            
        Returns:
            Dictionary with evaluation results
        """
        import subprocess
        import sys
        
        # Map task to evaluation script
        eval_script_map = {
            "meeting": "eval/evaluate_meeting_planning.py",
            "calendar": "eval/evaluate_calendar_scheduling.py", 
            "trip": "eval/evaluate_trip_planning.py"
        }
        
        eval_script = eval_script_map.get(task)
        if not eval_script or not Path(eval_script).exists():
            raise FileNotFoundError(f"Evaluation script not found: {eval_script}")
        
        print(f"🚀 Running evaluation script: {eval_script}")
        
        # Run evaluation script
        try:
            result = subprocess.run([
                sys.executable, eval_script,
                f"--data_path={eval_dataset_path}"
            ], capture_output=True, text=True, check=True)
            
            output = result.stdout
            print("📊 Evaluation output:")
            print(output)
            
            # Parse accuracy from output
            accuracy = self._parse_accuracy_from_output(output, task)
            
            return {
                "accuracy": accuracy,
                "raw_output": output,
                "eval_script": eval_script,
                "dataset_path": str(eval_dataset_path)
            }
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Evaluation failed: {e}")
            print(f"Error output: {e.stderr}")
            raise
    
    def _parse_accuracy_from_output(self, output: str, task: str) -> float:
        """Parse accuracy from evaluation script output"""
        lines = output.strip().split('\n')
        
        if task == "meeting":
            # Look for "Accuracy for all: X.XXX"
            for line in lines:
                if "Accuracy for all:" in line:
                    try:
                        return float(line.split(":")[-1].strip())
                    except ValueError:
                        pass
        
        elif task == "calendar":
            # Look for "Overall solve rate of X samples: X.XXX"
            for line in lines:
                if "Overall solve rate" in line and "samples:" in line:
                    try:
                        return float(line.split(":")[-1].strip())
                    except ValueError:
                        pass
        
        elif task == "trip":
            # Look for "EM Accuracy of X samples: X.XXX"
            for line in lines:
                if "EM Accuracy" in line and "samples:" in line:
                    try:
                        return float(line.split(":")[-1].strip())
                    except ValueError:
                        pass
        
        print(f"⚠️  Could not parse accuracy from output, returning 0.0")
        return 0.0
    
    def batch_create_and_evaluate(self, results_dir: Path, tasks: List[str] = None, 
                                models: List[str] = None) -> List[Dict[str, Any]]:
        """
        Batch create evaluation datasets and run evaluations for all results
        
        Args:
            results_dir: Directory containing evaluation results
            tasks: List of tasks to process (default: all)
            models: List of models to process (default: all)
            
        Returns:
            List of evaluation results
        """
        if tasks is None:
            tasks = ["meeting", "calendar", "trip"]
        if models is None:
            models = ["14b", "8b", "1.5b"]
        
        # Use analyzer to find all baseline results
        analyzer = ResultsAnalyzer(results_dir)
        all_results = analyzer.scan_all_results()
        
        # Filter for baseline results only
        baseline_results = [r for r in all_results if r.eval_type == "baseline"]
        
        if not baseline_results:
            print("❌ No baseline results found for static evaluation")
            return []
        
        evaluation_results = []
        
        for result in baseline_results:
            if result.task not in tasks or result.model_size not in models:
                continue
            
            print(f"\n🔄 Processing {result.task} task with {result.model_size} model...")
            
            try:
                # Create evaluation dataset
                eval_dataset_path = self.create_eval_dataset(
                    result.result_path, result.task
                )
                
                # Run evaluation
                eval_result = self.run_evaluation(eval_dataset_path, result.task)
                
                # Add metadata
                eval_result.update({
                    "task": result.task,
                    "model_size": result.model_size,
                    "timestamp": result.timestamp,
                    "original_result_path": str(result.result_path)
                })
                
                evaluation_results.append(eval_result)
                
            except Exception as e:
                print(f"❌ Failed to evaluate {result.task} ({result.model_size}): {e}")
                continue
        
        return evaluation_results


def main():
    parser = argparse.ArgumentParser(description="Create static evaluation datasets from model results")
    parser.add_argument("--results-dir", default="results/",
                       help="Directory containing evaluation results")
    parser.add_argument("--results-file", 
                       help="Specific results JSON file to process")
    parser.add_argument("--task", choices=["meeting", "calendar", "trip"],
                       help="Specific task to process")
    parser.add_argument("--model", choices=["14b", "8b", "1.5b"],
                       help="Specific model to process")
    parser.add_argument("--eval-data-dir", default="eval/data",
                       help="Directory containing original evaluation datasets")
    parser.add_argument("--output-dir", default="processor/eval_datasets",
                       help="Directory to save evaluation datasets")
    parser.add_argument("--run-eval", action="store_true",
                       help="Also run the evaluation scripts")
    
    args = parser.parse_args()
    
    creator = StaticEvalCreator(Path(args.eval_data_dir))
    
    if args.results_file:
        # Process single results file
        results_path = Path(args.results_file)
        if not results_path.exists():
            print(f"❌ Results file not found: {results_path}")
            return
        
        if not args.task:
            print("❌ --task required when using --results-file")
            return
        
        eval_dataset_path = creator.create_eval_dataset(results_path, args.task)
        
        if args.run_eval:
            eval_result = creator.run_evaluation(eval_dataset_path, args.task)
            print(f"\n✅ Final accuracy: {eval_result['accuracy']:.4f}")
    
    else:
        # Batch process all results
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            print(f"❌ Results directory not found: {results_dir}")
            return
        
        tasks = [args.task] if args.task else None
        models = [args.model] if args.model else None
        
        evaluation_results = creator.batch_create_and_evaluate(
            results_dir, tasks, models
        )
        
        if args.run_eval and evaluation_results:
            # Save summary of all evaluations
            summary_path = Path(args.output_dir) / "static_eval_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            
            with summary_path.open("w") as f:
                json.dump(evaluation_results, f, indent=2)
            
            print(f"\n📊 STATIC EVALUATION SUMMARY")
            print("=" * 50)
            for result in evaluation_results:
                print(f"{result['task']:8} {result['model_size']:4} {result['accuracy']:.4f}")
            
            print(f"\n💾 Full results saved: {summary_path}")


if __name__ == "__main__":
    main()