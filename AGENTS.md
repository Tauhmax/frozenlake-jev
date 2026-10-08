# Repository Guidelines

All development keeps simple and never do bunch of tests. Use git-bash as the tool.

## Project Structure & Module Organization

This repository currently contains `plan.md`, the research and implementation plan for a FrozenLake decision-planning benchmark. The proposed Python layout places configuration in `configs/`, environment wrappers and map generation in `src/environments/`, policies in `src/agents/`, optimal-value solvers in `src/oracles/`, evaluation code in `src/evaluation/`, and the entry point at `src/run.py`. Put tests in `tests/` and generated datasets, predictions, and figures under `results/`. Keep these responsibilities separate as implementation begins.

## Build, Test, and Development Commands

No executable project or dependency manifest exists yet. Add a reproducible setup command and document it alongside the first implementation. The plan calls for Gymnasium FrozenLake; a future local run should use the project's documented entry point, such as `python -m src.run`. Once a test runner is configured, record the exact test and lint commands here and in the README. Do not present planned commands as working until verified.

## Coding Style & Naming Conventions

Use Python's four-space indentation and `snake_case` for modules, functions, and variables. Name tests `test_*.py`, matching their target behavior or module. Keep prompt formatting, action-label mapping, transition logic, and metric calculations in distinct modules. Add formatter and linter configuration with the Python toolchain; apply it consistently to touched files.

## Testing Guidelines

Prioritize deterministic tests for boundaries, holes, terminal states, unreachable maps, and tied optimal actions. Cross-check the BFS oracle against value iteration on small maps. Test choice-token IDs and scoring position before large-model runs. Split benchmark data by map, fix random seeds, and cache model outputs so results can be reproduced without repeated 27B inference. No coverage threshold has been set.

## Commit & Pull Request Guidelines

There is no Git history from which to infer a commit convention. Use short imperative subjects, such as `Add FrozenLake transition tests`. Pull requests should describe the change, link relevant issues, list verification commands, and include sample output or figures when evaluation behavior changes. Record checkpoint, prompt, seed, and configuration changes that affect results.

## Implemented minimal setup

Python 3.9.21 setup: `python -m venv .venv`, activate it, then
`python -m pip install -r requirements.txt`. Oracle entry point:
`python -m src.run` (use a fresh `--output` path on subsequent runs).
Verified checks: `python -m unittest discover -s tests -v`,
`python -m ruff check src tests`, `python -m ruff format --check src tests`.
Optional model dependencies are in `requirements-model.txt`; real model inference
has not yet been verified. Keep the implementation lightweight and tests focused.

## Bulk evaluation

Serve a loaded model with `python -m src.serve --model-dir MODEL --device cuda`.
Use the Python 3.11 / Transformers 5.19 environment for Qwen3.5 and newer models.
Generate map-held-out data with `python -m src.evaluation.dataset --maps 1000`;
score it with `python -m src.evaluation.bulk --batch-size 8`.
The bulk output resumes from JSONL; preserve its manifest and do not mix model,
prompt, precision or batch-size changes into an existing cache. Bind the service
to loopback and keep model weights / results out of Git.
