from __future__ import annotations

"""Loader for the Natural Plan benchmark datasets.

"""

from dataclasses import dataclass
from typing import Dict, List, Any
import json
import pathlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class NPExample:
    """Container for a single Natural-Plan example."""

    task: str  # "trip" | "meeting" | "calendar"
    example_id: str  
    prompt: str  
    golden: Any  
    meta: Dict[str, Any]  

    # Convenience attributes so that evaluator code that expects ``question`` works.
    # For Natural-Plan we treat the entire prompt as the "question".
    @property
    def question(self) -> str:  # type: ignore
        return self.prompt

    @property
    def choices(self) -> list:  # type: ignore
        """Return an empty list so that generic CSV writer does not break."""
        return []


class NaturalPlanLoader:
    """Load Natural-Plan data from the original json files located in ``eval/data``.

    Parameters
    ----------
    data_root: str | pathlib.Path
        Directory that contains the three json files. Defaults to
        ``../eval/data`` relative to the *bench* package root.
    """

    FILES = {
        "trip": "trip_planning.json",
        "meeting": "meeting_planning.json",
        "calendar": "calendar_scheduling.json",
    }

    def __init__(self, data_root: str | pathlib.Path | None = None):
        if data_root is None:
            self.data_root = (pathlib.Path(__file__).resolve().parents[3] / "eval/data").resolve()
        else:
            self.data_root = pathlib.Path(data_root).expanduser().resolve()

        if not self.data_root.exists():
            raise FileNotFoundError(f"Natural-Plan data dir not found: {self.data_root}")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def load(self, task: str) -> List[NPExample]:
        """Load all examples for the given task.

        Parameters
        ----------
        task: str
            One of ``trip``, ``meeting``, or ``calendar``.
        """
        if task not in self.FILES:
            raise ValueError(f"Unknown Natural-Plan task: {task}. Expected one of {list(self.FILES)}")

        path = self.data_root / self.FILES[task]
        logger.info("Loading Natural-Plan %s from %s", task, path)

        with path.open() as f:
            raw: Dict[str, Dict[str, Any]] = json.load(f)

        examples: List[NPExample] = []
        for example_id, record in raw.items():
            examples.append(
                NPExample(
                    task=task,
                    example_id=example_id,
                    prompt=record["prompt_5shot"],
                    golden=record["golden_plan"],
                    meta=record,
                )
            )

        logger.info("Loaded %d examples for Natural-Plan %s", len(examples), task)
        return examples 