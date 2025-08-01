#!/usr/bin/env python3
"""
Comprehensive Natural-Plan Results Analyzer

Analyzes evaluation results across baseline, budget, and scaling experiments
to generate accuracy vs tokens/question plots and performance comparisons.

Usage:
    python processor/analyze_results.py --results-dir results/
    python processor/analyze_results.py --results-dir results/ --plot-type accuracy_vs_tokens
    python processor/analyze_results.py --results-dir results/ --task meeting --models 14b,8b,1.5b
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import re


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    task: str
    model_size: str
    eval_type: str  # baseline, budget, scaling, hightokens
    token_limit: Optional[int]  # For budget evaluations
    accuracy: float
    avg_tokens_per_question: float
    total_questions: int
    avg_time_ms: float
    tokens_per_second: float
    timestamp: str
    result_path: Path


class ResultsAnalyzer:
    """Analyzes Natural-Plan evaluation results across different experiment types"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results: List[EvaluationResult] = []
        
    def scan_all_results(self) -> List[EvaluationResult]:
        """Scan all result directories and extract evaluation metrics"""
        print(f"🔍 Scanning results in {self.results_dir}")
        
        # Scan different evaluation types
        self._scan_baseline_results()
        self._scan_budget_results()
        self._scan_hightokens_results()
        self._scan_scaling_results()
        
        print(f"📊 Found {len(self.results)} evaluation results")
        return self.results
    
    def _scan_baseline_results(self):
        """Scan base/ directory for baseline evaluations"""
        base_dir = self.results_dir / "base"
        if not base_dir.exists():
            return
            
        for model_dir in base_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_size = model_dir.name
            
            for task_dir in model_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                
                # Extract task and timestamp from directory name
                match = re.match(r'([^_]+)_(\d{8}_\d{6})', task_dir.name)
                if not match:
                    continue
                task, timestamp = match.groups()
                
                # Find results JSON file
                json_files = list(task_dir.glob("results_*.json"))
                if not json_files:
                    continue
                    
                result = self._parse_result_file(json_files[0], task, model_size, "baseline", None, timestamp)
                if result:
                    self.results.append(result)
    
    def _scan_budget_results(self):
        """Scan budget/ directory for budget-constrained evaluations"""
        budget_dir = self.results_dir / "budget"
        if not budget_dir.exists():
            return
            
        for model_dir in budget_dir.iterdir():
            if not model_dir.is_dir():
                continue
            model_size = model_dir.name
            
            # Handle both flat structure and token-limit subdirectories
            for subdir in model_dir.iterdir():
                if not subdir.is_dir():
                    continue
                    
                # Check if this is a token limit directory (e.g., "128")
                if subdir.name.isdigit():
                    token_limit = int(subdir.name)
                    search_dir = subdir
                else:
                    token_limit = 256  # Default budget from config
                    search_dir = subdir
                    
                # If search_dir contains task directories
                if any(d.is_dir() for d in search_dir.iterdir()):
                    for task_dir in search_dir.iterdir():
                        if not task_dir.is_dir():
                            continue
                        
                        match = re.match(r'([^_]+)_(\d{8}_\d{6})', task_dir.name)
                        if not match:
                            continue
                        task, timestamp = match.groups()
                        
                        json_files = list(task_dir.glob("results_*.json"))
                        if not json_files:
                            continue
                            
                        result = self._parse_result_file(json_files[0], task, model_size, "budget", token_limit, timestamp)
                        if result:
                            self.results.append(result)
                else:
                    # This is a task directory directly under model
                    match = re.match(r'([^_]+)_(\d{8}_\d{6})', search_dir.name)
                    if not match:
                        continue
                    task, timestamp = match.groups()
                    
                    json_files = list(search_dir.glob("results_*.json"))
                    if not json_files:
                        continue
                        
                    result = self._parse_result_file(json_files[0], task, model_size, "budget", token_limit, timestamp)
                    if result:
                        self.results.append(result)
    
    def _scan_hightokens_results(self):
        """Scan hightokens/ directory"""
        hightokens_dir = self.results_dir / "hightokens"
        if not hightokens_dir.exists():
            return
            
        for task_dir in hightokens_dir.iterdir():
            if not task_dir.is_dir():
                continue
                
            match = re.match(r'([^_]+)_(\d{8}_\d{6})', task_dir.name)
            if not match:
                continue
            task, timestamp = match.groups()
            
            json_files = list(task_dir.glob("results_*.json"))
            if not json_files:
                continue
                
            # Assume high-tokens uses 14b model (or extract from results)
            result = self._parse_result_file(json_files[0], task, "14b", "hightokens", None, timestamp)
            if result:
                self.results.append(result)
    
    def _scan_scaling_results(self):
        """Scan scaling/ directory for scaling evaluations"""
        scaling_dir = self.results_dir / "scaling"
        if not scaling_dir.exists():
            return
        # Implementation similar to baseline, but for scaling results
        pass
    
    def _parse_result_file(self, json_path: Path, task: str, model_size: str, 
                          eval_type: str, token_limit: Optional[int], timestamp: str) -> Optional[EvaluationResult]:
        """Parse a results JSON file and extract metrics"""
        try:
            with json_path.open() as f:
                data = json.load(f)
            
            # Calculate average tokens per question from question results
            question_results = data.get("question_results", [])
            if not question_results:
                return None
                
            total_tokens = sum(q.get("prompt_tokens", 0) + q.get("output_tokens", 0) 
                             for q in question_results)
            avg_tokens = total_tokens / len(question_results) if question_results else 0
            
            return EvaluationResult(
                task=task,
                model_size=model_size,
                eval_type=eval_type,
                token_limit=token_limit,
                accuracy=data.get("accuracy", 0.0),
                avg_tokens_per_question=avg_tokens,
                total_questions=data.get("total_questions", len(question_results)),
                avg_time_ms=data.get("avg_time_per_question", 0.0),
                tokens_per_second=data.get("avg_tokens_per_second", 0.0),
                timestamp=timestamp,
                result_path=json_path
            )
            
        except Exception as e:
            print(f"⚠️  Error parsing {json_path}: {e}")
            return None
    
    def create_accuracy_vs_tokens_plot(self, task: str = None, models: List[str] = None, 
                                     output_dir: Path = None):
        """Create accuracy vs tokens/question plots"""
        if not self.results:
            print("❌ No results found. Run scan_all_results() first.")
            return
            
        # Filter results
        filtered_results = self.results
        if task:
            filtered_results = [r for r in filtered_results if r.task == task]
        if models:
            filtered_results = [r for r in filtered_results if r.model_size in models]
        
        if not filtered_results:
            print(f"❌ No results found for task={task}, models={models}")
            return
        
        # Convert to DataFrame for easier plotting
        df_data = []
        for result in filtered_results:
            df_data.append({
                'Task': result.task,
                'Model': result.model_size,
                'Eval Type': result.eval_type,
                'Token Limit': result.token_limit,
                'Accuracy': result.accuracy,
                'Tokens per Question': result.avg_tokens_per_question,
                'Timestamp': result.timestamp
            })
        
        df = pd.DataFrame(df_data)
        
        if df.empty:
            print("❌ No data to plot")
            return
        
        # Create plots for each task
        tasks = df['Task'].unique()
        
        for task_name in tasks:
            task_df = df[df['Task'] == task_name]
            
            plt.figure(figsize=(12, 8))
            
            # Create scatter plot with different colors for eval types and shapes for models
            eval_types = task_df['Eval Type'].unique()
            models = task_df['Model'].unique()
            
            colors = plt.cm.Set1(np.linspace(0, 1, len(eval_types)))
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
            
            for i, eval_type in enumerate(eval_types):
                for j, model in enumerate(models):
                    subset = task_df[(task_df['Eval Type'] == eval_type) & (task_df['Model'] == model)]
                    if not subset.empty:
                        plt.scatter(subset['Tokens per Question'], subset['Accuracy'],
                                  c=[colors[i]], marker=markers[j % len(markers)], s=100,
                                  label=f"{model} ({eval_type})", alpha=0.7)
            
            plt.xlabel('Average Tokens per Question', fontsize=12)
            plt.ylabel('Accuracy', fontsize=12)
            plt.title(f'Accuracy vs Tokens per Question - {task_name.title()} Task', fontsize=14)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            if output_dir:
                output_dir.mkdir(exist_ok=True)
                plot_path = output_dir / f"accuracy_vs_tokens_{task_name}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"📊 Plot saved: {plot_path}")
            else:
                plt.show()
            
            plt.close()
    
    def generate_summary_report(self, output_dir: Path = None):
        """Generate a comprehensive summary report"""
        if not self.results:
            print("❌ No results found. Run scan_all_results() first.")
            return
        
        # Create summary DataFrame
        df_data = []
        for result in self.results:
            df_data.append({
                'Task': result.task,
                'Model': result.model_size,
                'Eval Type': result.eval_type,
                'Token Limit': result.token_limit or 'None',
                'Accuracy': f"{result.accuracy:.4f}",
                'Avg Tokens/Q': f"{result.avg_tokens_per_question:.1f}",
                'Questions': result.total_questions,
                'Avg Time (ms)': f"{result.avg_time_ms:.1f}",
                'Tokens/sec': f"{result.tokens_per_second:.1f}",
                'Timestamp': result.timestamp
            })
        
        df = pd.DataFrame(df_data)
        
        # Print summary
        print("\n📋 EVALUATION SUMMARY")
        print("=" * 80)
        print(df.to_string(index=False))
        
        # Save to CSV if output directory specified
        if output_dir:
            output_dir.mkdir(exist_ok=True)
            csv_path = output_dir / "evaluation_summary.csv"
            df.to_csv(csv_path, index=False)
            print(f"\n💾 Summary saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Natural-Plan evaluation results")
    parser.add_argument("--results-dir", default="results/", 
                       help="Directory containing evaluation results")
    parser.add_argument("--output-dir", default="processor/plots/",
                       help="Directory to save plots and reports")
    parser.add_argument("--plot-type", default="accuracy_vs_tokens",
                       choices=["accuracy_vs_tokens", "summary"],
                       help="Type of analysis to perform")
    parser.add_argument("--task", help="Filter by specific task (meeting, calendar, trip)")
    parser.add_argument("--models", help="Comma-separated list of models to include (14b,8b,1.5b)")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Parse models list
    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    
    # Initialize analyzer and scan results
    analyzer = ResultsAnalyzer(results_dir)
    analyzer.scan_all_results()
    
    # Generate requested analysis
    if args.plot_type == "accuracy_vs_tokens":
        analyzer.create_accuracy_vs_tokens_plot(
            task=args.task, 
            models=models, 
            output_dir=output_dir
        )
    elif args.plot_type == "summary":
        analyzer.generate_summary_report(output_dir)
    
    # Always generate summary
    analyzer.generate_summary_report(output_dir)


if __name__ == "__main__":
    main()