#!/usr/bin/env python3
"""
Validate Natural-Plan evaluation results structure and data integrity.

This script checks that your results directory has the expected structure
and that result files contain valid data.

Usage:
    python processor/validate_results.py
    python processor/validate_results.py --results-dir custom_results/
    python processor/validate_results.py --fix-issues
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import csv


class ResultsValidator:
    """Validates Natural-Plan evaluation results"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.issues = []
        self.warnings = []
    
    def validate_all(self) -> bool:
        """Run all validation checks"""
        print(f"🔍 Validating results in {self.results_dir}")
        print("=" * 50)
        
        # Check directory structure
        self._validate_directory_structure()
        
        # Check individual result files
        self._validate_result_files()
        
        # Report results
        self._report_results()
        
        return len(self.issues) == 0
    
    def _validate_directory_structure(self):
        """Check that expected directories exist"""
        expected_dirs = ["base", "budget"]
        optional_dirs = ["hightokens", "scaling"]
        
        for dir_name in expected_dirs:
            dir_path = self.results_dir / dir_name
            if not dir_path.exists():
                self.issues.append(f"Missing required directory: {dir_path}")
            elif not dir_path.is_dir():
                self.issues.append(f"Expected directory but found file: {dir_path}")
        
        for dir_name in optional_dirs:
            dir_path = self.results_dir / dir_name
            if not dir_path.exists():
                self.warnings.append(f"Optional directory not found: {dir_path}")
    
    def _validate_result_files(self):
        """Check individual result files for validity"""
        for result_dir in self.results_dir.rglob("*"):
            if not result_dir.is_dir():
                continue
            
            # Check if this looks like a result directory (has timestamp)
            if not self._is_result_directory(result_dir):
                continue
            
            self._validate_single_result_dir(result_dir)
    
    def _is_result_directory(self, dir_path: Path) -> bool:
        """Check if directory looks like a result directory"""
        # Should have pattern: task_YYYYMMDD_HHMMSS
        import re
        pattern = r'^[a-z]+_\d{8}_\d{6}$'
        return bool(re.match(pattern, dir_path.name))
    
    def _validate_single_result_dir(self, result_dir: Path):
        """Validate a single result directory"""
        # Check for required files
        json_files = list(result_dir.glob("results_*.json"))
        csv_files = list(result_dir.glob("detailed_results_*.csv"))
        log_files = list(result_dir.glob("*.log"))
        
        if not json_files:
            self.issues.append(f"Missing results JSON file in {result_dir}")
        elif len(json_files) > 1:
            self.warnings.append(f"Multiple results JSON files in {result_dir}")
        
        if not csv_files:
            self.warnings.append(f"Missing detailed results CSV in {result_dir}")
        
        if not log_files:
            self.warnings.append(f"Missing log file in {result_dir}")
        
        # Validate JSON content if file exists
        if json_files:
            self._validate_json_file(json_files[0])
        
        # Validate CSV content if file exists  
        if csv_files:
            self._validate_csv_file(csv_files[0])
    
    def _validate_json_file(self, json_path: Path):
        """Validate JSON result file content"""
        try:
            with json_path.open() as f:
                data = json.load(f)
            
            # Check required fields
            required_fields = ["accuracy", "total_questions", "question_results"]
            for field in required_fields:
                if field not in data:
                    self.issues.append(f"Missing required field '{field}' in {json_path}")
            
            # Validate question results
            question_results = data.get("question_results", [])
            if not question_results:
                self.issues.append(f"Empty question_results in {json_path}")
            else:
                # Check first question result has expected fields
                first_q = question_results[0]
                expected_q_fields = ["question_id", "generated_text", "is_correct"]
                for field in expected_q_fields:
                    if field not in first_q:
                        self.warnings.append(f"Missing field '{field}' in question results in {json_path}")
            
            # Validate accuracy value
            accuracy = data.get("accuracy")
            if accuracy is not None and (accuracy < 0 or accuracy > 1):
                self.warnings.append(f"Unusual accuracy value {accuracy} in {json_path}")
                
        except json.JSONDecodeError as e:
            self.issues.append(f"Invalid JSON in {json_path}: {e}")
        except Exception as e:
            self.issues.append(f"Error reading {json_path}: {e}")
    
    def _validate_csv_file(self, csv_path: Path):
        """Validate CSV result file content"""
        try:
            with csv_path.open() as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                
                # Check for expected columns
                expected_cols = ["question_id", "subject", "question", "is_correct", 
                               "input_tokens", "output_tokens"]
                missing_cols = [col for col in expected_cols if col not in fieldnames]
                if missing_cols:
                    self.warnings.append(f"Missing CSV columns {missing_cols} in {csv_path}")
                
                # Check if file has data
                row_count = sum(1 for _ in reader)
                if row_count == 0:
                    self.issues.append(f"Empty CSV file: {csv_path}")
                    
        except Exception as e:
            self.issues.append(f"Error reading CSV {csv_path}: {e}")
    
    def _report_results(self):
        """Report validation results"""
        print("\n📋 VALIDATION RESULTS")
        print("=" * 30)
        
        if not self.issues and not self.warnings:
            print("✅ All validation checks passed!")
            return
        
        if self.issues:
            print(f"❌ Found {len(self.issues)} issues:")
            for issue in self.issues:
                print(f"   • {issue}")
        
        if self.warnings:
            print(f"\n⚠️  Found {len(self.warnings)} warnings:")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        print(f"\n📊 Summary: {len(self.issues)} issues, {len(self.warnings)} warnings")


def main():
    parser = argparse.ArgumentParser(description="Validate Natural-Plan evaluation results")
    parser.add_argument("--results-dir", default="results/",
                       help="Directory containing evaluation results")
    parser.add_argument("--fix-issues", action="store_true",
                       help="Attempt to fix common issues (not implemented yet)")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return 1
    
    validator = ResultsValidator(results_dir)
    is_valid = validator.validate_all()
    
    if args.fix_issues:
        print("\n🔧 Fix issues functionality not implemented yet")
    
    return 0 if is_valid else 1


if __name__ == "__main__":
    exit(main())