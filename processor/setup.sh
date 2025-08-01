#!/bin/bash
# Natural-Plan Processor Setup Script
# Sets up the processor environment and runs initial analysis

set -e  # Exit on any error

echo "🚀 Natural-Plan Processor Setup"
echo "==============================="
echo ""

# Check if we're in the right directory
if [ ! -f "processor/analyze_results.py" ]; then
    echo "❌ Please run this script from the natural-plan project root directory"
    exit 1
fi

# Install dependencies
echo "📦 Installing processor dependencies..."
pip install -r processor/requirements.txt

# Validate results directory
echo ""
echo "🔍 Validating results directory..."
python processor/validate_results.py

# Check for missing evaluations
echo ""
echo "📋 Checking for missing evaluations..."
python processor/check_coverage.py

# Ask if user wants to run batch analysis
echo ""
read -p "🚀 Run comprehensive batch analysis now? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Running batch analysis..."
    python processor/batch_analysis.py
    
    echo ""
    echo "✅ Setup and analysis complete!"
    echo "📁 Check processor/analysis_output/ for results"
    echo "📈 Plots saved in processor/analysis_output/plots/"
else
    echo "✅ Setup complete!"
    echo ""
    echo "Next steps:"
    echo "  • Run: python processor/batch_analysis.py"
    echo "  • Or use: make batch-analysis"
    echo "  • Check missing evals: make check-missing"
fi

echo ""
echo "📚 See processor/README.md for full documentation"