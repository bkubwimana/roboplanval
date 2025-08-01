#!/usr/bin/env python3
"""
Batch analysis script for comprehensive Natural-Plan evaluation analysis.

This script runs all analysis types and generates plots for all tasks and models.

Usage:
    python processor/batch_analysis.py
    python processor/batch_analysis.py --results-dir custom_results/
    python processor/batch_analysis.py --skip-plots
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from analyze_results import ResultsAnalyzer
from check_coverage import EvaluationCoverageChecker


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive batch analysis")
    parser.add_argument("--results-dir", default="results/",
                       help="Directory containing evaluation results")
    parser.add_argument("--output-dir", default="processor/analysis_output/",
                       help="Directory to save all outputs")
    parser.add_argument("--skip-plots", action="store_true",
                       help="Skip generating plots")
    parser.add_argument("--skip-missing-check", action="store_true",
                       help="Skip checking for missing evaluations")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    print("🚀 Starting comprehensive Natural-Plan analysis...")
    print("=" * 60)
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Check for missing evaluations
    if not args.skip_missing_check:
        print("\n📋 Step 1: Checking for missing evaluations...")
        checker = EvaluationCoverageChecker(results_dir)
        missing_baseline = checker.find_missing_baseline_evals()
        
        if missing_baseline:
            print(f"⚠️  Found {len(missing_baseline)} missing baseline evaluations:")
            for task, model in missing_baseline:
                print(f"    {task} task with {model} model")
            print("    Run baseline evaluations before proceeding with analysis.")
        else:
            print("✅ All baseline evaluations found!")
    
    # 2. Initialize analyzer and scan results
    print("\n🔍 Step 2: Scanning all evaluation results...")
    analyzer = ResultsAnalyzer(results_dir)
    results = analyzer.scan_all_results()
    
    if not results:
        print("❌ No results found to analyze!")
        return
    
    print(f"Found {len(results)} evaluation results")
    
    # 3. Generate summary report
    print("\n📊 Step 3: Generating summary report...")
    analyzer.generate_summary_report(output_dir)
    
    # 4. Generate plots if not skipped
    if not args.skip_plots:
        print("\n📈 Step 4: Generating accuracy vs tokens plots...")
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Generate plots for each task
        tasks = ["meeting", "calendar", "trip"]
        models = ["14b", "8b", "1.5b"]
        
        for task in tasks:
            print(f"  📊 Creating plot for {task} task...")
            try:
                analyzer.create_accuracy_vs_tokens_plot(
                    task=task,
                    models=models,
                    output_dir=plots_dir
                )
            except Exception as e:
                print(f"    ⚠️  Error creating plot for {task}: {e}")
        
        # Generate combined plot (all tasks)
        print("  📊 Creating combined plot for all tasks...")
        try:
            analyzer.create_accuracy_vs_tokens_plot(
                task=None,  # All tasks
                models=models,
                output_dir=plots_dir
            )
        except Exception as e:
            print(f"    ⚠️  Error creating combined plot: {e}")
    
    # 5. Generate analysis insights
    print("\n🧠 Step 5: Generating analysis insights...")
    insights_file = output_dir / "analysis_insights.md"
    
    # Analyze results for insights
    task_accuracies = {}
    model_performance = {}
    eval_type_comparison = {}
    
    for result in results:
        # Task accuracies
        if result.task not in task_accuracies:
            task_accuracies[result.task] = []
        task_accuracies[result.task].append(result.accuracy)
        
        # Model performance
        if result.model_size not in model_performance:
            model_performance[result.model_size] = []
        model_performance[result.model_size].append(result.accuracy)
        
        # Eval type comparison
        if result.eval_type not in eval_type_comparison:
            eval_type_comparison[result.eval_type] = []
        eval_type_comparison[result.eval_type].append(result.accuracy)
    
    # Write insights
    with insights_file.open("w") as f:
        f.write("# Natural-Plan Evaluation Analysis Insights\n\n")
        f.write(f"Generated from {len(results)} evaluation results\n\n")
        
        f.write("## Task Performance Summary\n\n")
        for task, accuracies in task_accuracies.items():
            avg_acc = sum(accuracies) / len(accuracies)
            f.write(f"- **{task.title()}**: {avg_acc:.3f} average accuracy ({len(accuracies)} evaluations)\n")
        
        f.write("\n## Model Size Performance\n\n")
        for model, accuracies in model_performance.items():
            avg_acc = sum(accuracies) / len(accuracies)
            f.write(f"- **{model}**: {avg_acc:.3f} average accuracy ({len(accuracies)} evaluations)\n")
        
        f.write("\n## Evaluation Type Comparison\n\n")
        for eval_type, accuracies in eval_type_comparison.items():
            avg_acc = sum(accuracies) / len(accuracies)
            f.write(f"- **{eval_type}**: {avg_acc:.3f} average accuracy ({len(accuracies)} evaluations)\n")
        
        # Token efficiency analysis
        f.write("\n## Token Efficiency Insights\n\n")
        budget_results = [r for r in results if r.eval_type == "budget"]
        baseline_results = [r for r in results if r.eval_type == "baseline"]
        
        if budget_results and baseline_results:
            avg_budget_tokens = sum(r.avg_tokens_per_question for r in budget_results) / len(budget_results)
            avg_baseline_tokens = sum(r.avg_tokens_per_question for r in baseline_results) / len(baseline_results)
            token_reduction = (avg_baseline_tokens - avg_budget_tokens) / avg_baseline_tokens * 100
            
            f.write(f"- Budget evaluations use {avg_budget_tokens:.1f} tokens/question on average\n")
            f.write(f"- Baseline evaluations use {avg_baseline_tokens:.1f} tokens/question on average\n")
            f.write(f"- Token reduction: {token_reduction:.1f}%\n")
    
    print(f"💾 Analysis insights saved to: {insights_file}")
    
    # 6. Final summary
    print("\n✅ Batch analysis complete!")
    print("=" * 60)
    print(f"📁 All outputs saved to: {output_dir}")
    print(f"📊 Summary report: {output_dir}/evaluation_summary.csv")
    if not args.skip_plots:
        print(f"📈 Plots directory: {plots_dir}")
    print(f"🧠 Insights: {insights_file}")
    
    if not args.skip_missing_check and missing_baseline:
        print(f"\n⚠️  Don't forget to run missing evaluations!")
        print(f"   Commands saved in: {output_dir}/missing_eval_commands.sh")


if __name__ == "__main__":
    main()