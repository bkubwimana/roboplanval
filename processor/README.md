# Natural-Plan Results Processor

Post-processing toolkit for Natural-Plan evaluation results. Generates accuracy vs tokens plots and analysis reports.

## Quick Start

```bash
# Install dependencies
pip install -r processor/requirements.txt

# Check what evaluations you have
python processor/check_coverage.py

# Run complete analysis (recommended)
python processor/batch_analysis.py

# Run static evaluation with Google's scripts
python processor/run_static_eval.py
```

## Key Files

- `batch_analysis.py` - Run everything in one command
- `check_coverage.py` - See what evaluations you have
- `run_static_eval.py` - Run Google's evaluation scripts for ground truth
- `analyze_results.py` - Generate plots and analysis
- `planner_predictions.py` - Merge predictions into datasets

## Outputs

Results go to `processor/analysis_output/`:
- Accuracy vs tokens plots in `plots/`
- Summary report in `evaluation_summary.csv`
- Insights in `analysis_insights.md`

## Directory Structure Expected

```
results/
├── base/{model}/{task}_{timestamp}/results_*.json
└── budget/{model}/{tokens}/{task}_{timestamp}/results_*.json
```

That's it. Run `python processor/batch_analysis.py` and check the outputs.