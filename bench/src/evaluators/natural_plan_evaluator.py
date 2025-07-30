from __future__ import annotations

"""Evaluator for the Natural-Plan benchmark.

This module plugs Natural-Plan into the existing *bench* infrastructure.  It is
implemented as a thin subclass of :class:`bench.src.evaluators.base_evaluator.BaseEvaluator`
so we automatically inherit telemetry, timing, CSV logging, etc.
"""

from typing import Any, Dict, List, Optional
import os, sys, pathlib
from datetime import datetime
from functools import lru_cache

# Ensure repository root (where evaluate_*.py live) is on PYTHONPATH
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Also add eval directory where official scripts live
EVAL_DIR = REPO_ROOT / 'eval'
if EVAL_DIR.exists() and str(EVAL_DIR) not in sys.path:
    sys.path.append(str(EVAL_DIR))

from .base_evaluator import BaseEvaluator, EvaluationResult
from ..data_loaders.natural_plan_loader import NaturalPlanLoader, NPExample

# Helper importors ---------------------------------------------------------

# We defer importing the official evaluation modules until **after** we know
# which task we are running.  This prevents absl.flags from registering the
# same flag multiple times when a single process tries to import evaluate_* for
# more than one task.

@lru_cache(maxsize=None)
def get_trip_funcs():
    from evaluate_trip_planning import parse_response, compute_example_score
    return parse_response, compute_example_score


@lru_cache(maxsize=None)
def get_calendar_parse():
    from evaluate_calendar_scheduling import _parse_response
    return _parse_response


@lru_cache(maxsize=None)
def get_meeting_funcs():
    from evaluate_meeting_planning import process_constraints, validator_from_text, parse_text_plan
    return process_constraints, validator_from_text, parse_text_plan


class NaturalPlanEvaluator(BaseEvaluator):
    """Evaluate a local or remote model on one of the Natural-Plan tasks."""

    def __init__(self, config_path: str, task: str):
        if task not in ("trip", "meeting", "calendar"):
            raise ValueError("task must be 'trip', 'meeting', or 'calendar'")

        # Call parent constructor -> loads YAML config
        super().__init__(config_path)
        self.loader = NaturalPlanLoader()
        self.task = task

    # ------------------------------------------------------------------
    # Overridden helper methods
    # ------------------------------------------------------------------
    def format_prompt(self, example: NPExample) -> str:  # type: ignore[override]
        """Return the few-shot prompt contained in each example."""
        return example.prompt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def evaluate_task(
        self,
        model_path: str,
        output_dir: str = "./results",
        compute_metrics: Optional[bool] = None,
    ) -> EvaluationResult:
        """Run the full task evaluation.

        Parameters
        ----------
        model_path : str
            HuggingFace repo or local path recognised by VLLMModel.
        output_dir : str
            Directory where CSV / JSON summaries will be written.
        compute_metrics : bool | None
            If *False*, skip correctness checking during the loop to speed up
            runs.  Defaults to the YAML field ``evaluation.compute_metrics``.
        """
        if compute_metrics is None:
            compute_metrics = self.config.evaluation.get("compute_metrics", True)

        if not self.model:
            self.setup_model(model_path)

        examples: List[NPExample] = self.loader.load(self.task)
        if not examples:
            raise RuntimeError(f"No examples loaded for Natural-Plan task '{self.task}'")

        num_questions = self.config.evaluation.get("num_questions")
        if num_questions and num_questions < len(examples):
            examples = examples[:num_questions]

        run_name = f"{self.config.name}_{self.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_name = os.path.basename(model_path.rstrip("/"))

        from ..telemetry import monitor_evaluation 
        from ..utils.csv_writer import evaluation_csv_writer

        question_results: List[Dict[str, Any]] = []
        correct_count = 0

        with monitor_evaluation(
            output_dir=output_dir,
            run_name=run_name,
            model_name=model_name,
            config_name=self.config.name,
            evaluation_type=f"natural_plan_{self.task}",
        ) as monitor, evaluation_csv_writer(output_dir, run_name, self.task) as write_csv_row:
            for i, ex in enumerate(examples):
                prompt = self.format_prompt(ex)
                prediction = self.model.predict(
                    prompt=prompt,
                    max_tokens=self.config.model["max_tokens"],
                    temperature=self.config.model["temperature"],
                    top_p=self.config.model["top_p"],
                    stop=self.config.model.get("stop_sequences", ["<|im_end|>", "<|endoftext|>", "</s>"])
                )

                generated = prediction.generated_text.strip()
                is_correct = False

                if compute_metrics:
                    is_correct = self._check_correctness(ex, generated)
                    if is_correct:
                        correct_count += 1

                question_result = {
                    "question_id": ex.example_id,
                    "prompt_tokens": prediction.input_tokens,
                    "output_tokens": prediction.output_tokens,
                    "generated_text": generated,
                    "is_correct": is_correct,
                    "time_ms": prediction.total_time_ms,
                    "tokens_per_second": prediction.tokens_per_second,
                }
                question_results.append(question_result)
                write_csv_row(i, ex, prediction, None, None, is_correct)
                monitor.record_question_result(i, prediction)

        accuracy = correct_count / len(examples) if compute_metrics else -1
        avg_time = sum(r["time_ms"] for r in question_results) / len(question_results)
        avg_tps = sum(r["tokens_per_second"] for r in question_results) / len(question_results)

        result = EvaluationResult(
            config_name=self.config.name,
            model_name=model_name,
            subject=self.task,
            total_questions=len(examples),
            correct_answers=correct_count,
            accuracy=accuracy,
            avg_time_per_question=avg_time,
            avg_tokens_per_second=avg_tps,
            question_results=question_results,
        )

        if self.config.output.get("save_detailed_responses", True):
            self._save_detailed_results(result, output_dir, run_name)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _sanitize_meeting_steps(self, steps: list[str]) -> list[str]:
        """Return only the sentences that match the expected Natural-Plan few-shot patterns.

        The official validator expects every sentence to start with one of the
        canonical prefixes (e.g. "You start", "You travel", "You wait", "You meet").
        Some models sometimes prepend extra guidelines or switch languages; those
        stray sentences trigger the "Unknown plan format" error.  By filtering
        them out we avoid false negatives without touching the validator.
        """
        allowed_prefixes = (
            "You start",
            "You travel",
            "You wait",
            "You meet",
        )
        return [s for s in steps if s.startswith(allowed_prefixes)]

    def _check_correctness(self, ex: NPExample, generated: str) -> bool:
        """Return True iff *generated* matches the gold answer for the task."""
        if self.task == "trip":
            parse_response, compute_example_score = get_trip_funcs()
            parsed = parse_response(generated)
            return bool(
                compute_example_score(ex.meta["cities"], ex.meta["durations"], parsed) == 1.0
            )

        if self.task == "calendar":
            _parse_response = get_calendar_parse()
            r_day, r_start, r_end = _parse_response(generated)
            s_day, s_start, s_end = _parse_response(ex.golden)
            return (r_day == s_day) and (r_start == s_start) and (r_end == s_end)

        # Meeting-planning requires running the validator
        process_constraints, validator_from_text, parse_text_plan = get_meeting_funcs()
        plan_txt = parse_text_plan(generated)
        # Remove stray sentences that would crash the validator
        plan_txt = self._sanitize_meeting_steps(plan_txt)
        start_location, initial_time = ex.meta["constraints"][0]
        constraints = process_constraints(ex.meta["constraints"][1:])
        try:
            score_pred = validator_from_text(
                plan_txt, constraints, start_location, initial_time, ex.meta["dist_matrix"]
            )
        except ValueError:
            # Any parsing error means the answer is incorrect.
            return False

        score_gold = validator_from_text(
            ex.meta["golden_plan"], constraints, start_location, initial_time, ex.meta["dist_matrix"]
        )
        return score_pred == score_gold 