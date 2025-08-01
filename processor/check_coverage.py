#!/usr/bin/env python3
"""
Natural-Plan Evaluation Coverage Reporter

Read-only coverage checker - scans your results directory to show which 
evaluations are present across baseline, budget, and other evaluation types.

The script expects the following directory structure:
- results/base/{model}/{task}_{timestamp}/results_*.json  
- results/budget/{model}/{token_limit}/{task}_{timestamp}/results_*.json

Usage:
    python processor/check_coverage.py                     
    python processor/check_coverage.py --debug              
    python processor/check_coverage.py --task meeting       
"""

import argparse
from pathlib import Path
import re
from typing import List, Tuple


class EvaluationCoverageChecker:
    """Checks evaluation coverage across different evaluation types"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        
    def _scan_actual_results(self) -> List[Tuple[str, str, str]]:
        """Scan results directory and return (task, model, eval_type) tuples"""
        found_results = []
        
        # Scan base/ directory for baseline results
        base_dir = self.results_dir / "base"
        if base_dir.exists():
            for model_dir in base_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model = model_dir.name
                
                for task_dir in model_dir.iterdir():
                    if not task_dir.is_dir():
                        continue
                    
                    # Extract task from directory name like "meeting_20250730_162655"
                    match = re.match(r'([^_]+)_\d{8}_\d{6}', task_dir.name)
                    if match:
                        task = match.group(1)
                        # Check if results file exists
                        json_files = list(task_dir.glob("results_*.json"))
                        if json_files:
                            found_results.append((task, model, "baseline"))
        
        # Scan budget/ directory for budget results  
        budget_dir = self.results_dir / "budget"
        if budget_dir.exists():
            for model_dir in budget_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                model = model_dir.name
                
                # Handle nested structure (budget may have token limit subdirs)
                self._scan_budget_recursive(model_dir, model, found_results)
        
        return found_results
    
    def _scan_budget_recursive(self, search_dir: Path, model: str, found_results: List, token_budget: str = None):
        """Recursively scan budget directory structure"""
        for subdir in search_dir.iterdir():
            if not subdir.is_dir():
                continue
            
            # If subdir name is a number (token limit), recurse into it
            if subdir.name.isdigit():
                self._scan_budget_recursive(subdir, model, found_results, subdir.name)
            else:
                # This should be a task directory
                match = re.match(r'([^_]+)_\d{8}_\d{6}', subdir.name)
                if match:
                    task = match.group(1)
                    # Check if results file exists
                    json_files = list(subdir.glob("results_*.json"))
                    if json_files:
                        eval_type = f"budget-{token_budget}" if token_budget else "budget"
                        found_results.append((task, model, eval_type))
        
    def find_missing_baseline_evals(self, required_tasks: List[str] = None, 
                                  required_models: List[str] = None) -> List[Tuple[str, str]]:
        """Find missing baseline evaluations"""
        if required_tasks is None:
            required_tasks = ["meeting", "calendar", "trip"]
        if required_models is None:
            required_models = ["14b", "8b", "1.5b"]
            
        # Scan existing results using our direct scanning method
        existing_results = self._scan_actual_results()
        
        # Get existing baseline combinations
        existing_baseline = set()
        for task, model, eval_type in existing_results:
            if eval_type == "baseline":
                existing_baseline.add((task, model))
        
        # Find missing combinations
        missing = []
        for task in required_tasks:
            for model in required_models:
                if (task, model) not in existing_baseline:
                    missing.append((task, model))
        
        return missing
    

    def check_budget_coverage(self) -> dict:
        """Check if all baseline tasks have corresponding budget evaluations"""
        existing_results = self._scan_actual_results()
        
        baseline_combos = set()
        budget_combos = set()
        token_budgets = {}  # (task, model) -> set of token budgets
        
        for task, model, eval_type in existing_results:
            combo = (task, model)
            if eval_type == "baseline":
                baseline_combos.add(combo)
            elif eval_type.startswith("budget"):
                budget_combos.add(combo)
                if combo not in token_budgets:
                    token_budgets[combo] = set()
                if "-" in eval_type:
                    token_budget = eval_type.split("-")[1]
                    token_budgets[combo].add(token_budget)
        
        return {
            "baseline_only": baseline_combos - budget_combos,
            "budget_only": budget_combos - baseline_combos,
            "both": baseline_combos & budget_combos,
            "token_budgets": token_budgets
        }
    
    def get_found_evaluations(self) -> dict:
        """Get summary of found evaluations for debugging"""
        existing_results = self._scan_actual_results()
        
        summary = {}
        
        for task, model, eval_type in existing_results:
            if eval_type not in summary:
                summary[eval_type] = []
            summary[eval_type].append(f"{task} ({model})")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Check Natural-Plan evaluation coverage")
    parser.add_argument("--results-dir", default="results/",
                       help="Directory containing evaluation results")
    parser.add_argument("--task", help="Check specific task only")
    parser.add_argument("--models", help="Comma-separated list of models (14b,8b,1.5b)")
    parser.add_argument("--debug", action="store_true",
                       help="Show detailed info about found evaluations")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    required_tasks = [args.task] if args.task else ["meeting", "calendar", "trip"]
    required_models = args.models.split(",") if args.models else ["14b", "8b", "1.5b"]
    
    checker = EvaluationCoverageChecker(results_dir)
    
    if args.debug:
        found_evals = checker.get_found_evaluations()
        print("FOUND EVALUATIONS:")
        print("=" * 30)
        
        if "baseline" in found_evals:
            print(f"Baseline evaluations: {len(found_evals['baseline'])}")
            for eval_info in sorted(found_evals['baseline']):
                print(f"  {eval_info}")
            print()
        
        # Show budget evaluations by token budget
        budget_types = [k for k in found_evals.keys() if k.startswith("budget")]
        for budget_type in sorted(budget_types):
            token_budget = budget_type.split("-")[1] if "-" in budget_type else "unknown"
            print(f"Budget-{token_budget} evaluations: {len(found_evals[budget_type])}")
            for eval_info in sorted(found_evals[budget_type]):
                print(f"  {eval_info}")
            print()
    
    # Find missing baseline evaluations
    missing_baseline = checker.find_missing_baseline_evals(required_tasks, required_models)
    
    if missing_baseline:
        print("MISSING BASELINE EVALUATIONS:")
        print("=" * 50)
        for task, model in missing_baseline:
            print(f"  {task} task with {model} model")
        print()
    else:
        print("All required baseline evaluations found!")
    
    # Check budget coverage  
    coverage = checker.check_budget_coverage()
    
    print("EVALUATION COVERAGE SUMMARY:")
    print("=" * 35)
    total_baseline = len(coverage['both']) + len(coverage['baseline_only'])
    total_budget = len(coverage['both']) + len(coverage['budget_only']) 
    print(f"  Baseline evaluations found: {total_baseline}")
    print(f"  Budget evaluations found: {total_budget}")
    print(f"  Complete pairs (both types): {len(coverage['both'])}")
    
    # Show token budget details
    if coverage['token_budgets']:
        token_counts = {}
        for combo, tokens in coverage['token_budgets'].items():
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        
        print(f"\n  Token budget breakdown:")
        for token in sorted(token_counts.keys(), key=int):
            print(f"    {token} tokens: {token_counts[token]} evaluations")
    
    if coverage['baseline_only']:
        print(f"\nBaseline-only (missing budget): {len(coverage['baseline_only'])}")
        for task, model in sorted(coverage['baseline_only']):
            print(f"    {task} ({model})")
    
    if coverage['budget_only']:
        print(f"\nBudget-only (missing baseline): {len(coverage['budget_only'])}")
        for task, model in sorted(coverage['budget_only']):
            print(f"    {task} ({model})")
    
    if len(coverage['both']) > 0 and not coverage['baseline_only'] and not coverage['budget_only']:
        print("\nStatus: Complete coverage! All evaluations have both baseline and budget results.")


if __name__ == "__main__":
    main()