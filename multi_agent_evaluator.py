"""
multi_agent_evaluator.py

Evaluator for the multi-agent demo.

OpenEvolve will call evaluate(program_path), where program_path is the
path to a candidate version of multi_agent_program.py.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType
from typing import Dict

from dotenv import load_dotenv

# Make sure .env is loaded if someone runs this file directly.
load_dotenv()


def _load_program(path: str) -> ModuleType:
    """
    Dynamically import the candidate program from the given file path.
    """
    spec = importlib.util.spec_from_file_location("candidate_program", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load candidate program from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module  # type: ignore[return-value]


def evaluate(program_path: str) -> Dict[str, float]:
    """
    Called by OpenEvolve for each candidate.

    It must return a dict of scalar metrics.
    We define:
    - success_rate: accuracy on the tiny QA set
    - combined_score: same as success_rate (for now)
    - score: main fitness (also same)
    """
    mod = _load_program(program_path)

    if not hasattr(mod, "run_benchmark"):
        raise AttributeError(
            "Candidate program does not define run_benchmark(). "
            "Make sure multi_agent_program.py has run_benchmark()."
        )

    metrics = mod.run_benchmark()
    success_rate = float(metrics.get("success_rate", 0.0))

    combined_score = success_rate  # simple objective: just accuracy

    return {
        "score": combined_score,
        "combined_score": combined_score,
        "success_rate": success_rate,
    }


if __name__ == "__main__":
    # Simple check: load the current multi_agent_program.py in the same folder and print its score.
    import os

    current_path = os.path.join(os.path.dirname(__file__), "multi_agent_program.py")
    print(evaluate(current_path))
