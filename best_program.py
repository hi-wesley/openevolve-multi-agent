"""
multi_agent_program.py

Tiny 2-agent QA system:

- Planner agent: optionally breaks questions into steps.
- Solver agent: answers using the question plus an optional plan.

We intentionally start with a BAD configuration so OpenEvolve has
something to improve. The base solver always says "I do not know",
so the initial success_rate should be around 0.0 on the eval set.
"""

from __future__ import annotations

import os
from typing import Dict, Any, List

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env (in this folder).
# This makes OPENAI_API_KEY available via os.environ.
load_dotenv()

# Get the API key from the environment.
_api_key = os.environ.get("OPENAI_API_KEY")
if not _api_key:
    raise RuntimeError(
        "OPENAI_API_KEY not found.\n"
        "Create a .env file in this folder with a line like:\n"
        "  OPENAI_API_KEY=sk-your-real-key-here"
    )

# Global OpenAI client for the multi-agent system.
_client = OpenAI(api_key=_api_key)


# NOTE FOR OPENEOLVE (in case the LLM sees comments directly):
# - Only change AGENT_CONFIG below.
# - Do NOT change run_benchmark or the helper functions.
# - The evaluator expects run_benchmark() to exist and return a dict
#   with at least a "success_rate" float.


# EVOLVE-BLOCK-START
AGENT_CONFIG: Dict[str, Any] = {
    # Start with planner turned OFF so the base system is very weak.
    "use_planner": True,  # Enable planner to improve question structuring

    # Planner is defined but not used initially. OpenEvolve is allowed
    # to turn it on or improve its prompt.
    "planner": {
        "system_prompt": (
            "You are a planning assistant. Given a question, write 2-4 short "
            "steps that would help answer the question. Keep steps concise."
        ),
        "max_tokens": 128,
    },

    # BAD solver prompt on purpose:
    # It tells the model to always answer "I do not know", so our initial
    # success_rate should be close to 0. OpenEvolve should discover that
    # it needs to change this prompt to actually answer questions.
    "solver": {
        "system_prompt": (
            "You are an intelligent assistant. Use the plan provided to answer the question as accurately as possible."
        ),
        "max_tokens": 32,
    },
}
# EVOLVE-BLOCK-END


def _call_llm(system_prompt: str, user_content: str, max_tokens: int = 128) -> str:
    """
    Call the chat completion API with a simple system plus user prompt.
    """
    model = "gpt-4o-mini"  # Use a default model directly
    resp = _client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def run_multi_agent(question: str, config: Dict[str, Any] | None = None) -> str:
    """
    Run the 2-agent pipeline (planner then solver) on a single question.
    """
    if config is None:
        config = AGENT_CONFIG

    # Optional planner step.
    plan_text = ""
    if config.get("use_planner", False):
        planner_cfg = config["planner"]
        plan_text = _call_llm(
            planner_cfg["system_prompt"],
            "Question: " + question,
            max_tokens=int(planner_cfg.get("max_tokens", 128)),
        )

    # Solver step.
    solver_cfg = config["solver"]
    solver_input = "Question: " + question
    if plan_text:
        solver_input += "\nPlan: " + plan_text
    answer = _call_llm(
        solver_cfg["system_prompt"],
        solver_input,
        max_tokens=int(solver_cfg.get("max_tokens", 64)),
    )
    return answer


# Tiny evaluation set for OpenEvolve to optimize on.
# Still small so cost stays low, but not completely trivial.
EVAL_SET: List[Dict[str, str]] = [
    {
        "question": "What is the capital of France?",
        "expected_substring": "paris",
    },
    {
        "question": "Who wrote the novel Pride and Prejudice?",
        "expected_substring": "jane austen",
    },
    {
        "question": "What is the chemical formula for water?",
        "expected_substring": "h2o",
    },
    {
        "question": "What is the capital of Italy?",
        "expected_substring": "rome",
    },
    {
        "question": "What is the largest ocean on Earth?",
        "expected_substring": "pacific",
    },
    {
        "question": "Who painted the Mona Lisa?",
        "expected_substring": "leonardo da vinci",
    },
    {
        "question": "What is the boiling point of water?",
        "expected_substring": "100",
    },
]


def _normalize(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum() or ch.isspace())


def _is_correct(answer: str, expected_substring: str) -> bool:
    norm_ans = _normalize(answer)
    norm_exp = _normalize(expected_substring)
    return norm_exp in norm_ans


def run_benchmark(verbose: bool = False) -> Dict[str, float]:
    """
    Run the multi-agent system on EVAL_SET and compute a simple accuracy metric.

    Returns a dict so the evaluator can pick out "success_rate" as the score.
    """
    correct = 0
    for ex in EVAL_SET:
        ans = run_multi_agent(ex["question"])
        if verbose:
            print("Q:", ex["question"])
            print("A:", ans)
            print()
        if _is_correct(ans, ex["expected_substring"]):
            correct += 1

    success_rate = correct / len(EVAL_SET) if EVAL_SET else 0.0
    if verbose:
        print(f"Correct answers: {correct}/{len(EVAL_SET)}")
    return {
        "success_rate": float(success_rate),
        "num_examples": float(len(EVAL_SET)),
    }


if __name__ == "__main__":
    # Manual smoke test (will call the LLM).
    print(run_benchmark(verbose=True))
