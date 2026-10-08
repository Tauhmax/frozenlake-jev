# frozenlake-jev

Minimal deterministic FrozenLake benchmark: value-iteration ground truth and
local causal-LM choice-token JEV (one forward pass, zero generated tokens).

## Setup (Git Bash, Windows)

Verified with Python 3.9.21:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
python -m src.run
```

On Linux/macOS use `source .venv/bin/activate` instead.
The default run needs no model. It writes `results/run.json`: 11 safe states,
6 VI iterations, zero Bellman residual on the included map. Existing outputs
are never overwritten; use `--output results/another-run.json` to run again.

## Local model (optional)

```bash
python -m pip install -r requirements-model.txt
# Place a complete HF model/tokenizer in models/base first.
python -m src.run --model-dir models/base --output results/jev.json
# Add --device cuda if your PyTorch installation supports your GPU.
```

`models/base/` is a placeholder. No pretrained weights are downloaded by this
project; model loading uses `local_files_only=True`. Qwen2.5-0.5B-Instruct is
an example small model. The optional stack targets text-only causal models;
it does not promise compatibility with every newer Qwen architecture.
The model dependency installation and real model inference have NOT been
verified in this initial setup. CPU is the default device.

JEV applies the tokenizer chat template when available, requests thinking off,
and appends an explicit `Answer:` assistant prefill. Raw base tokenizers without
a chat template receive a plain prompt. An unclosed `<think>` block is rejected.
For each actual prompt, encoding `prefix + letter` must preserve the prefix and
append exactly one distinct token. Other tokenizers fail explicitly. The last
input position predicts the next token; only A/B/C/D logits are normalized at
T=1. There is no generation, search, or trained head. Inspect the saved prompt
and token IDs when validating a new checkpoint; mock tests do not establish
real tokenizer/template compatibility.

## Ground truth and results

Edit `configs/frozenlake.json` for the rectangular map, gamma, tolerances and seed.
Gymnasium's deterministic `P` transition table drives VI. Reward is 1 on entering
the goal and 0 otherwise; terminal transitions never bootstrap. Gamma must be
less than 1. Stopping uses the recorded Bellman residual. Goal/hole states are
excluded from output; unreachable safe states are labeled and excluded from
model scoring. Ties use absolute tolerance 1e-9 by default.

The prompt states the same discounted objective as the oracle. A/B/C/D map to
LEFT/DOWN/RIGHT/UP (0/1/2/3). Output records Q*, V*, all optimal actions and, when
a model is supplied, prompt, token IDs, logits, restricted probabilities, latency,
chosen action, accuracy, regret, optimal probability mass and binary Brier score.
Restricted softmax is a relative action score, not calibrated optimality probability.
Outputs serve as an offline cache; do not rerun inference to recompute metrics.
Keep a model directory immutable and record its checkpoint revision alongside
results. The seed is recorded; cross-device bitwise reproducibility is not promised.

This intentionally implements one fixed map, not the full research plan:
map generation/splits, plots, search baselines and aggregate calibration are future work.

## Lightweight verification

```bash
python -m unittest discover -s tests -v
python -m ruff check src tests
python -m ruff format --check src tests
```

Two focused tests cover BFS-vs-VI values, ties, unreachable states, holes,
boundaries, terminal values, and synthetic next-token position validation.

Implementation references: [Gymnasium FrozenLake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
and [Hugging Face chat templates](https://huggingface.co/docs/transformers/v4.48.0/chat_templating).
