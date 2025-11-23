"""
run_evolve.py

Small driver script that uses OpenEvolve as a library to optimize
the multi-agent configuration defined in multi_agent_program.py.

Usage (from this folder, with your venv/conda env active):

    python run_evolve.py
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env so that OPENAI_API_KEY is available
load_dotenv()

from openevolve import run_evolution
from openevolve.config import Config, LLMModelConfig


def main() -> None:
    # 1. Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment variables.")

    # 2. Set up OpenEvolve config
    config = Config()

    # Use gpt-4o-mini as the evolution model
    config.llm.models = [
        LLMModelConfig(
            name="gpt-4o-mini",
            api_key=api_key,
        )
    ]

    # Keep runs cheap by limiting iterations.
    config.max_iterations = 8

    # Turn off cascade evaluation since we do not implement stage1 functions.
    config.evaluator.cascade_evaluation = False

    # 3. Run evolution
    result = run_evolution(
        "multi_agent_program.py",
        "multi_agent_evaluator.py",
        config=config,
    )

    print("\nEvolution finished.")
    print("Best score found:", getattr(result, "best_score", None))

    # 4. Save the best program into THIS repo as best_program.py
    project_root = Path(__file__).resolve().parent
    best_path = project_root / "best_program.py"

    best_code = getattr(result, "best_code", None)
    if isinstance(best_code, str) and best_code.strip():
        best_path.write_text(best_code, encoding="utf-8")
        print(f"Saved best program code to {best_path}")
    else:
        print(
            "Note: could not find result.best_code on the result object.\n"
            "OpenEvolve already saved best_program.py under its output directory\n"
            "the temp path printed in the logs. If you want, you can copy that\n"
            "file into this repo manually."
        )


if __name__ == "__main__":
    main()
