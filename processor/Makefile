# Natural-Plan Processor Makefile
# Convenient commands for running analysis tasks

.PHONY: help install validate check-missing batch-analysis plots summary clean

# Default target
help:
	@echo "🚀 Natural-Plan Results Processor"
	@echo "=================================="
	@echo ""
	@echo "Available targets:"
	@echo "  install            - Install Python dependencies"
	@echo "  validate           - Validate results directory structure"
	@echo "  check-missing      - Check for missing evaluations"
	@echo "  run-static-eval    - Run static evaluations using Google's scripts"
	@echo "  test-static-eval   - Test the static evaluation workflow"
	@echo "  batch-analysis     - Run comprehensive batch analysis"
	@echo "  full-analysis      - Complete workflow (validate + static eval + analysis)"
	@echo "  plots              - Generate accuracy vs tokens plots only"
	@echo "  summary            - Generate summary report only"
	@echo "  clean              - Clean generated outputs"
	@echo ""
	@echo "Task-specific targets:"
	@echo "  meeting-analysis   - Analyze meeting task only"
	@echo "  calendar-analysis  - Analyze calendar task only"
	@echo "  trip-analysis      - Analyze trip task only"
	@echo ""
	@echo "Usage examples:"
	@echo "  make install"
	@echo "  make validate"
	@echo "  make full-analysis"
	@echo "  make run-static-eval-task TASK=meeting"
	@echo "  make check-missing TASK=meeting"
	@echo ""

# Install dependencies
install:
	@echo "📦 Installing processor dependencies..."
	pip install -r requirements.txt

# Validate results structure
validate:
	@echo "🔍 Validating results directory..."
	python validate_results.py

# Check evaluation coverage
check-coverage:
	@echo "📋 Checking evaluation coverage..."
ifdef TASK
	python check_coverage.py --results-dir ../results --task $(TASK)
else
	python check_coverage.py --results-dir ../results
endif

# Run comprehensive batch analysis
batch-analysis: validate
	@echo "🚀 Running comprehensive batch analysis..."
	python batch_analysis.py

# Generate plots only
plots:
	@echo "📈 Generating accuracy vs tokens plots..."
	python analyze_results.py --plot-type accuracy_vs_tokens

# Generate summary only
summary:
	@echo "📊 Generating summary report..."
	python analyze_results.py --plot-type summary

# Clean generated outputs
clean:
	@echo "🧹 Cleaning generated outputs..."
	rm -rf analysis_output/
	rm -f missing_eval_commands.sh
	@echo "✅ Cleaned!"

# Static evaluation using Google's evaluation scripts
run-static-eval:
	@echo "🚀 Running static evaluations..."
	python run_static_eval.py

run-static-eval-task:
	@echo "🚀 Running static evaluation for $(TASK)..."
	python run_static_eval.py --task $(TASK)

test-static-eval:
	@echo "🧪 Testing static evaluation workflow..."
	python test_static_eval.py

# Quick analysis for specific task
meeting-analysis:
	@echo "📊 Analyzing meeting task..."
	python analyze_results.py --task meeting

calendar-analysis:
	@echo "📊 Analyzing calendar task..."
	python analyze_results.py --task calendar

trip-analysis:
	@echo "📊 Analyzing trip task..."
	python analyze_results.py --task trip

# Complete workflow including static evaluation
full-analysis: validate run-static-eval batch-analysis
	@echo "✅ Complete analysis workflow finished!"