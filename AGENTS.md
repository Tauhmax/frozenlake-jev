# Repository Guidelines

Keep development simple and tests focused. Use Git Bash on Windows and Bash on Linux.

## Environment

Use Python 3.11 and Transformers 5.19.0 for all supported models. Shared pins live
only in `requirements.txt`; hardware profiles include that file. CUDA uses
`requirements-cuda.txt`; Ascend setup and its CANN-specific profile are documented
in `NPU_ASCEND.md`. Do not introduce per-model Python environments or duplicate
framework pins. Do not modify an environment while a model service is using it.

## Structure and behavior

- `src/environments/`: deterministic Gymnasium transitions and map generation.
- `src/oracles/`: infinite- and finite-horizon Bellman Q values.
- `src/agents/`: prompt, action mapping, local model and choice-token scoring.
- `src/evaluation/`: dataset creation, single-state and complete-episode evaluation.
- `src/run.py`: small-map oracle/model smoke run; `src/serve.py`: local model HTTP service.
- `configs/`: small-map configuration; `tests/`: focused deterministic tests.
- `docs/`: experimental reports; `models/` and `results/`: ignored local artifacts.

Primary evaluation plays one game from S on each map, with a 30-move limit.
The independent four-state-per-map evaluation remains a diagnostic, not a game.
Keep rules, finite/infinite horizon, prompt versions and oracle labels consistent.
Never expose oracle labels in model prompts. Keep map-held-out splits and seeds.
Bind the service to loopback. Never mix models, precision, prompts or batch sizes
in an existing output cache. Preserve the manifest beside the trace.

## Commands and validation

After activating the environment:

```bash
python -m src.run --output results/oracle.json
python -m src.serve --model-dir models/Qwen3.5-0.8B --device cuda
python -m src.evaluation.episodes --dataset results/datasets/frozenlake-1000-state-v2.jsonl --max-steps 30 --batch-size 16
python -m unittest discover -s tests -v
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
```

Install `requirements-dev.txt` for Ruff. Keep the focused checks:
BFS versus VI with boundaries, holes, terminals and ties; finite horizon and final
move semantics; single-token choice position; HTTP batch limits and error details. Validate new checkpoint tokenizers
and an actual forward before large runs. Hardware-specific execution must be
reported as unverified until tested on that hardware.

## Changes and commits

Use four-space indentation and snake_case. Keep prompt formatting, action mapping,
transitions and metrics separate. Use short imperative commit subjects. Describe
behavior, validation and any changed checkpoint, prompt, seed or configuration.
Delete superseded files instead of creating backups. Preserve published experiment
records and actual model/results artifacts unless their removal is requested.
