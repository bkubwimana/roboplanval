#!/usr/bin/env python3
"""
Run static evaluations using Google's original evaluation scripts.

This script automates the complete workflow:
1. Find baseline evaluation results
2. Create evaluation datasets by replacing pred_5shot_pro with model predictions  
3. Run Google's evaluation scripts
4. Save results for analysis

Usage:
    python processor/run_static_eval.py
    python processor/run_static_eval.py --task meeting --model 14b
    python processor/run_static_eval.py --results-dir custom_results/
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Any
from create_static_eval import StaticEvalCreator
from analyze_results import ResultsAnalyzer


class StaticEvaluationRunner:
    """Runs complete static evaluation workflow"""
    
    def __init__(self, results_dir: Path, eval_data_dir: Path = None, output_dir: Path = None):
        self.results_dir = results_dir
        self.eval_data_dir = eval_data_dir or Path("eval/data")
        self.output_dir = output_dir or Path("processor/static_eval_results")
        self.creator = StaticEvalCreator(self.eval_data_dir)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def find_baseline_results(self, tasks: List[str] = None, models: List[str] = None):
        """Find all baseline evaluation results"""
        analyzer = ResultsAnalyzer(self.results_dir)
        all_results = analyzer.scan_all_results()
        
        # Filter for baseline results
        baseline_results = [r for r in all_results if r.eval_type == "baseline"]
        
        if tasks:
            baseline_results = [r for r in baseline_results if r.task in tasks]
        if models:
            baseline_results = [r for r in baseline_results if r.model_size in models]
        
        return baseline_results
    
    def run_static_evaluation(self, baseline_result, verbose: bool = True) -> Dict[str, Any]:
        """Run static evaluation for a single baseline result"""
        task = baseline_result.task
        model = baseline_result.model_size
        timestamp = baseline_result.timestamp
        
        if verbose:
            print(f"\n🚀 Running static evaluation: {task} task, {model} model")
            print(f"   Source: {baseline_result.result_path}")
        
        try:
            # Create evaluation dataset
            eval_dataset_name = f"{task}_{model}_{timestamp}_eval.json"
            eval_dataset_path = self.output_dir / "datasets" / eval_dataset_name
            eval_dataset_path.parent.mkdir(parents=True, exist_ok=True)
            
            if verbose:
                print(f"📝 Creating evaluation dataset...")
            
            self.creator.create_eval_dataset(
                baseline_result.result_path, 
                task, 
                eval_dataset_path
            )
            
            # Run evaluation
            if verbose:
                print(f"🧮 Running Google evaluation script...")
            
            eval_result = self.creator.run_evaluation(eval_dataset_path, task)
            
            # Add metadata
            result = {
                "task": task,
                "model_size": model,
                "timestamp": timestamp,
                "accuracy": eval_result["accuracy"],
                "eval_dataset_path": str(eval_dataset_path),
                "original_result_path": str(baseline_result.result_path),
                "evaluation_output": eval_result["raw_output"],
                "eval_script": eval_result["eval_script"]
            }
            
            if verbose:
                print(f"✅ Static evaluation complete: {result['accuracy']:.4f} accuracy")
            
            return result
            
        except Exception as e:
            error_result = {
                "task": task,
                "model_size": model,
                "timestamp": timestamp,
                "accuracy": 0.0,
                "error": str(e),
                "original_result_path": str(baseline_result.result_path)
            }
            
            if verbose:
                print(f"❌ Static evaluation failed: {e}")
            
            return error_result
    
    def run_all_static_evaluations(self, tasks: List[str] = None, models: List[str] = None,
                                 verbose: bool = True) -> List[Dict[str, Any]]:
        """Run static evaluations for all found baseline results"""
        
        if verbose:
            print("🔍 Finding baseline evaluation results...")
        
        baseline_results = self.find_baseline_results(tasks, models)
        
        if not baseline_results:
            print("❌ No baseline results found!")
            print("   Make sure you have run baseline evaluations first.")
            print("   Use: python processor/check_coverage.py")
            return []
        
        if verbose:
            print(f"📊 Found {len(baseline_results)} baseline results to evaluate")
        
        static_eval_results = []
        
        # --- Special Handling for Combined Meeting Results ---
        for model_size in ["8b", "14b"]:
            combined_json = Path(f"combined_results/{model_size}/meeting_combined/results_meeting_combined.json")
            if combined_json.exists():
                if verbose:
                    print(f"🧬 Found combined meeting result for {model_size}, evaluating separately.")
                
                # Manually create a result-like object for the evaluator
                combined_result_obj = argparse.Namespace(
                    task="meeting",
                    model_size=model_size,
                    timestamp="combined",
                    result_path=combined_json
                )
                result = self.run_static_evaluation(combined_result_obj, verbose)
                static_eval_results.append(result)

        baseline_results = [r for r in baseline_results if not (r.task == "meeting" and r.model_size in ["8b", "14b"])]
        
        for baseline_result in baseline_results:
            result = self.run_static_evaluation(baseline_result, verbose)
            static_eval_results.append(result)
        
        # Save all results
        summary_file = self.output_dir / "static_eval_summary.json"
        with summary_file.open("w") as f:
            json.dump(static_eval_results, f, indent=2)
        
        if verbose:
            print(f"\n📋 STATIC EVALUATION SUMMARY")
            print("=" * 50)
            print(f"{'Task':<10} {'Model':<6} {'Accuracy':<10} {'Status'}")
            print("-" * 50)
            
            for result in static_eval_results:
                status = "✅ OK" if "error" not in result else "❌ Error"
                print(f"{result['task']:<10} {result['model_size']:<6} {result['accuracy']:<10.4f} {status}")
            
            print(f"\n💾 Full results saved: {summary_file}")
        
        return static_eval_results
    
    def create_detailed_csv_report(self, static_results: List[Dict[str, Any]]) -> None:
        """Create a detailed CSV report from the static evaluation results."""
        
        report_file = self.output_dir / "static_eval_detailed_summary.csv"
        
        with report_file.open("w", newline="") as f:
            import csv
            writer = csv.writer(f)
            writer.writerow(["model", "task", "metric_name", "metric_value"])
            
            for result in static_results:
                if "error" in result:
                    writer.writerow([result["model_size"], result["task"], "error", 1.0])
                    continue
                
                model = result["model_size"]
                task = result["task"]
                raw_output = result.get("evaluation_output", "")
                
                # Parse detailed metrics from raw output
                detailed_metrics = self._parse_detailed_metrics(raw_output, task)
                
                # Always write overall accuracy from structured field
                writer.writerow([model, task, "accuracy_total", result["accuracy"]])
                # Additional detailed metrics from raw output
                for metric_name, metric_value in detailed_metrics.items():
                    # Skip accuracy_total duplicate if present
                    if metric_name == "accuracy_total":
                        continue
                    writer.writerow([model, task, metric_name, metric_value])

        print(f"📊 Detailed CSV report saved: {report_file}")

    def _parse_detailed_metrics(self, raw_output: str, task: str) -> Dict[str, float]:
        """Parse detailed metrics from the raw output of an evaluation script."""
        metrics = {}
        for line in raw_output.splitlines():
            line = line.strip()
            if not line:
                continue

            # General accuracy/solve rate
            if ":" in line:
                parts = line.split(":")
                key = parts[0].strip()
                try:
                    value = float(parts[1].strip())
                    # Skip distance-matrix or timing lines (values > 1.0)
                    if value > 1.0:
                        continue
                    # Sanitize key
                    key = key.replace(" of 1000 samples", "").replace(" of 100 samples", "")
                    key = key.replace("Overall solve rate", "accuracy_total")
                    key = key.replace("Accuracy for all", "accuracy_total")
                    key = key.replace("Solve rate of ", "solve_rate_")
                    key = key.replace(" ", "_").replace(",", "")
                    metrics[key] = value
                except (ValueError, IndexError):
                    continue
        
        return metrics

    def create_comparison_report(self, static_results: List[Dict[str, Any]]) -> None:
        """DEPRECATED: This method is no longer used in favor of the detailed CSV report."""
        print("ℹ️ Markdown comparison report is deprecated and will not be generated.")
        pass


def main():
    parser = argparse.ArgumentParser(description="Run static evaluations using Google's evaluation scripts")
    parser.add_argument("--results-dir", default="results/",
                       help="Directory containing evaluation results")
    parser.add_argument("--eval-data-dir", default="eval/data",
                       help="Directory containing original evaluation datasets")
    parser.add_argument("--output-dir", default="processor/static_eval_results",
                       help="Directory to save static evaluation results")
    parser.add_argument("--task", choices=["meeting", "calendar", "trip"],
                       help="Run evaluation for specific task only")
    parser.add_argument("--model", choices=["14b", "8b", "1.5b"],
                       help="Run evaluation for specific model only")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Validate directories
    results_dir = Path(args.results_dir)
    eval_data_dir = Path(args.eval_data_dir)
    output_dir = Path(args.output_dir)
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return 1
    
    if not eval_data_dir.exists():
        print(f"❌ Eval data directory not found: {eval_data_dir}")
        return 1
    
    # Check if evaluation scripts exist
    eval_scripts = [
        "eval/evaluate_meeting_planning.py",
        "eval/evaluate_calendar_scheduling.py", 
        "eval/evaluate_trip_planning.py"
    ]
    
    missing_scripts = [script for script in eval_scripts if not Path(script).exists()]
    if missing_scripts:
        print("❌ Missing evaluation scripts:")
        for script in missing_scripts:
            print(f"   {script}")
        print("   Make sure you're running from the project root directory")
        return 1
    
    # Parse filter arguments
    tasks = [args.task] if args.task else None
    models = [args.model] if args.model else None
    verbose = not args.quiet
    
    # Run static evaluations
    runner = StaticEvaluationRunner(results_dir, eval_data_dir, output_dir)
    static_results = runner.run_all_static_evaluations(tasks, models, verbose)
    
    if not static_results:
        return 1
    
    # Create detailed CSV report
    if verbose:
        print("\n📊 Creating detailed CSV report...")
    runner.create_detailed_csv_report(static_results)
    
    if verbose:
        print(f"\n✅ Static evaluation workflow complete!")
        print(f"📁 All outputs saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())