# OpenEvolve Multi-Agent Demo

This repo contains a deliberately weak two-agent question-answering system plus the tooling needed for [OpenEvolve](https://github.com/open-evolve/openevolve) to automatically improve it. A planner agent can decompose a question into short steps, a solver agent answers using the plan, and a tiny evaluation set measures accuracy. OpenEvolve mutates only the `AGENT_CONFIG` block inside `multi_agent_program.py` while `multi_agent_evaluator.py` keeps score, so you get a tight toy example of autonomous prompt/program evolution.

## Why this exists
- Showcase how to wrap an arbitrary Python program (here, `multi_agent_program.py`) so OpenEvolve can iterate on it.
- Provide a reproducible baseline whose initial success rate is near zero because the solver always says "I do not know." This makes it easy to see the optimizer improve prompts and flags like `use_planner`.
- Archive the best discovered configuration (`best_program.py`) so you can restart evolution from a good seed or run the improved system directly.

## Repository layout
| Path | Purpose |
| --- | --- |
| `multi_agent_program.py` | Baseline two-agent pipeline plus `run_benchmark` that the evaluator calls. |
| `multi_agent_evaluator.py` | Imports a candidate module and returns `score`, `combined_score`, and `success_rate` based on its `run_benchmark`. |
| `run_evolve.py` | Driver that configures OpenEvolve (model list, iteration budget, cascade flags) and launches the search. |
| `best_program.py` | Latest best candidate saved from a prior evolution run; same API as the baseline but with improved prompts/config. |
| `LICENSE` | MIT License. |

## Requirements
- Python 3.10+
- Packages: `openevolve`, `openai`, `python-dotenv`
- OpenAI API key with access to `gpt-4o-mini` (or whatever model you set in `AGENT_MODEL`)

Install deps into a virtual environment:

```bash
pip install -U openevolve openai python-dotenv
```

## Environment variables
Create a `.env` file in this folder (loaded automatically via `python-dotenv`) containing at least:

```
OPENAI_API_KEY=sk-your-real-key
# Optional override for the per-agent calls
# AGENT_MODEL=gpt-4o-mini
```

`run_evolve.py` reads the same `OPENAI_API_KEY` when it configures OpenEvolve itself.

## Running the baseline benchmark
```bash
python multi_agent_program.py
```
This script calls the OpenAI API for each question in `EVAL_SET`, prints each answer when `verbose=True`, and returns a dictionary containing `success_rate` so OpenEvolve has a scalar objective.

## Running an OpenEvolve search
```bash
python run_evolve.py
```
The driver loads your API key, sets `gpt-4o-mini` as the evolution model, caps the run at 8 iterations to keep cost predictable, disables cascade evaluation (because the stage-one hooks are not implemented), and then calls `run_evolution`. When the run finishes it prints the best score and writes the winning code into `best_program.py` in this repo.

## Inspecting or reusing the best program
`best_program.py` is a drop-in replacement for `multi_agent_program.py`. Run it directly to confirm its benchmark score:

```bash
python best_program.py
```

If you want OpenEvolve to continue improving from that seed, copy the `AGENT_CONFIG` block from `best_program.py` back into `multi_agent_program.py` and re-run `run_evolve.py`.

## Customizing experiments
- **Change the task.** Edit `EVAL_SET` inside `multi_agent_program.py` to include the questions and expected substrings that represent success for your domain. The evaluator does not need any changes as long as `run_benchmark` still returns a dict with `success_rate`.
- **Switch models or prompts manually.** Update `AGENT_MODEL` (environment variable) or tweak `AGENT_CONFIG` before running the benchmark by hand. OpenEvolve will handle modifications to that block automatically.
- **Adjust the search budget.** Modify `config.max_iterations`, add/remove entries in `config.llm.models`, or tweak other `Config` fields inside `run_evolve.py` to trade off speed, cost, and result quality.

## Troubleshooting
- Missing or invalid `OPENAI_API_KEY` will raise immediately when the scripts start; double-check your `.env` file or export the variable in your shell.
- If you encounter rate limits, lower `config.max_iterations`, switch to a smaller model, or add delays inside `run_evolve.py`.
- Each evolution run overwrites `best_program.py`; copy that file elsewhere if you want to keep multiple checkpoints.

## License
Released under the MIT License (see `LICENSE`).