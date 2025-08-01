#!/usr/bin/env python3
"""
Combine split baseline evaluation results into unified files.

This handles cases where evaluations were run in parts (e.g., meeting task 
examples 1-500 and 501-1000 in separate runs) and combines them into 
complete evaluation results.

Usage:
    python processor/combine_split_results.py
    python processor/combine_split_results.py --task meeting --model 14b
    python processor/combine_split_results.py --output-dir combined_results/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import re
from collections import defaultdict


class SplitResultsCombiner:
    """Combines split evaluation results into unified files"""
    
    def __init__(self, results_dir: Path = None, output_dir: Path = None):
        self.results_dir = results_dir or Path("results")
        self.output_dir = output_dir or (self.results_dir / "combined")
        self.output_dir.mkdir(exist_ok=True)
        
    def find_split_evaluations(self) -> Dict[str, List[Path]]:
        """Find evaluations that were split across multiple runs"""
        # Scan baseline results
        base_dir = self.results_dir / "base"
        if not base_dir.exists():
            return {}
            
        # Group by (model, task)
        evaluations = defaultdict(list)
        
        for model_dir in base_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            
            for task_dir in model_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                
                # Extract task from directory name
                match = re.match(r'([^_]+)_\d{8}_\d{6}', task_dir.name)
                if match:
                    task = match.group(1)
                    key = f"{model}_{task}"
                    evaluations[key].append(task_dir)
        
        # Only return evaluations with multiple runs
        split_evaluations = {k: v for k, v in evaluations.items() if len(v) > 1}
        return split_evaluations
    
    def combine_results_files(self, result_dirs: List[Path]) -> Dict[str, Any]:
        """Combine multiple results files into one"""
        combined_data = {
            "accuracy": 0.0,
            "total_questions": 0,
            "question_results": []
        }
        
        # Use dict to deduplicate by question_id
        question_results_dict = {}
        
        for result_dir in sorted(result_dirs):  # Sort by timestamp to maintain order
            # Find the results JSON file
            json_files = list(result_dir.glob("results_*.json"))
            if not json_files:
                print(f"⚠️  No results file found in {result_dir}")
                continue
                
            results_file = json_files[0]  # Take first if multiple
            print(f"📂 Reading {results_file}")
            
            with results_file.open() as f:
                data = json.load(f)
            
            # Combine question results, replacing empty generated_text with real ones
            question_results = data.get("question_results", [])
            questions_added = 0
            questions_replaced = 0
            
            for q in question_results:
                q_id = str(q.get("question_id", ""))
                generated_text = q.get("generated_text", "").strip()
                
                if not q_id:
                    continue
                    
                if q_id not in question_results_dict:
                    # New question
                    question_results_dict[q_id] = q
                    questions_added += 1
                else:
                    # Question already exists - check if we should replace it
                    existing_text = question_results_dict[q_id].get("generated_text", "").strip()
                    
                    if not existing_text and generated_text:
                        # Replace empty with real prediction
                        question_results_dict[q_id] = q
                        questions_replaced += 1
                    elif existing_text and not generated_text:
                        # Keep existing real prediction, ignore empty one
                        pass
                    elif existing_text and generated_text:
                        # Both have text - this might be unexpected
                        print(f"   ⚠️  Both versions of question_id {q_id} have generated_text, keeping first")
            
            print(f"   ✅ Added {questions_added} new questions, replaced {questions_replaced} empty predictions")
        
        # Convert back to list and calculate stats
        all_question_results = list(question_results_dict.values())
        total_questions = len(all_question_results)
        total_correct = sum(1 for q in all_question_results if q.get("is_correct", False))
        
        # Update combined data
        combined_data["question_results"] = all_question_results
        combined_data["total_questions"] = total_questions
        combined_data["accuracy"] = total_correct / total_questions if total_questions > 0 else 0.0
        
        # Copy other fields from the first file if they exist
        if result_dirs:
            first_file = list(result_dirs[0].glob("results_*.json"))[0]
            with first_file.open() as f:
                first_data = json.load(f)
            
            for key, value in first_data.items():
                if key not in combined_data:
                    combined_data[key] = value
        
        return combined_data
    
    def save_combined_results(self, model: str, task: str, combined_data: Dict[str, Any]) -> Path:
        """Save combined results to output directory"""
        # Create output directory structure
        model_dir = self.output_dir / model
        model_dir.mkdir(exist_ok=True)
        
        # Create combined directory name
        timestamp = "combined"
        task_dir = model_dir / f"{task}_{timestamp}"
        task_dir.mkdir(exist_ok=True)
        
        # Save combined results
        results_file = task_dir / f"results_{task}_{timestamp}.json"
        with results_file.open("w") as f:
            json.dump(combined_data, f, indent=2)
        
        print(f"💾 Combined results saved: {results_file}")
        print(f"   📊 Total questions: {combined_data['total_questions']}")
        print(f"   🎯 Accuracy: {combined_data['accuracy']:.3f}")
        
        return results_file
    
    def process_all_splits(self) -> List[Path]:
        """Find and combine all split evaluations"""
        split_evals = self.find_split_evaluations()
        
        if not split_evals:
            print("✅ No split evaluations found")
            return []
        
        combined_files = []
        
        for key, result_dirs in split_evals.items():
            model, task = key.split("_", 1)
            print(f"\n🔧 Combining {len(result_dirs)} parts for {task} ({model}):")
            
            for i, dir_path in enumerate(sorted(result_dirs), 1):
                print(f"   Part {i}: {dir_path.name}")
            
            # Combine the results
            combined_data = self.combine_results_files(result_dirs)
            
            # Save combined results
            combined_file = self.save_combined_results(model, task, combined_data)
            combined_files.append(combined_file)
        
        return combined_files
    
    def process_specific(self, task: str, model: str) -> Path:
        """Combine split results for a specific task/model"""
        split_evals = self.find_split_evaluations()
        key = f"{model}_{task}"
        
        if key not in split_evals:
            print(f"❌ No split evaluations found for {task} ({model})")
            return None
        
        result_dirs = split_evals[key]
        print(f"🔧 Combining {len(result_dirs)} parts for {task} ({model})")
        
        combined_data = self.combine_results_files(result_dirs)
        combined_file = self.save_combined_results(model, task, combined_data)
        
        return combined_file


def main():
    parser = argparse.ArgumentParser(description="Combine split baseline evaluation results")
    parser.add_argument("--results-dir", default="results", 
                       help="Results directory to scan")
    parser.add_argument("--output-dir", default="results/combined",
                       help="Output directory for combined results")
    parser.add_argument("--task", help="Specific task to combine (e.g., meeting)")
    parser.add_argument("--model", help="Specific model to combine (e.g., 14b)")
    parser.add_argument("--list-only", action="store_true",
                       help="Only list split evaluations, don't combine")
    
    args = parser.parse_args()
    
    combiner = SplitResultsCombiner(
        results_dir=Path(args.results_dir),
        output_dir=Path(args.output_dir)
    )
    
    if args.list_only:
        split_evals = combiner.find_split_evaluations()
        if split_evals:
            print("SPLIT EVALUATIONS FOUND:")
            print("=" * 30)
            for key, dirs in split_evals.items():
                model, task = key.split("_", 1)
                print(f"{task} ({model}): {len(dirs)} parts")
                for d in sorted(dirs):
                    print(f"  - {d.name}")
        else:
            print("No split evaluations found")
        return
    
    if args.task and args.model:
        # Combine specific task/model
        combiner.process_specific(args.task, args.model)
    else:
        # Combine all split evaluations
        combined_files = combiner.process_all_splits()
        
        if combined_files:
            print(f"\n✅ Combined {len(combined_files)} split evaluations")
            print("\nNext steps:")
            print("1. Use combined results with planner_predictions.py:")
            for f in combined_files:
                # Just use the path as-is since it's already relative or handle absolute paths
                if f.is_absolute():
                    try:
                        rel_path = f.relative_to(Path.cwd())
                    except ValueError:
                        rel_path = f
                else:
                    rel_path = f
                print(f"   python processor/planner_predictions.py --results {rel_path} ...")
            print("2. Or use run_static_eval.py with --results-dir results/combined/")
        else:
            print("\n✅ No split evaluations to combine")


if __name__ == "__main__":
    main()