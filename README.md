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

## Local Qwen 4B (RTX 3070 / 8GB)

The selected checkpoint is [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
a non-thinking instruction model. The download script pins its revision to
`cdbee75f17c01a7cc42f958dc650907174af0554`.

```bash
# Install CUDA PyTorch first (the verified machine uses an NVIDIA GPU).
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-model.txt
python scripts/download_model.py
python -m src.run --model-dir models/Qwen3-4B-Instruct-2507 --device cuda --load-in-4bit --output results/qwen4b.json
```

The download contains roughly 8GB of original BF16 weights. Loading with
`--load-in-4bit` quantizes linear layers to NF4 with double quantization and FP16
compute. This is a quantized baseline, not a full-precision model result.
Keep GPU memory free for loading and evaluation. Model files are ignored by Git.
Only the explicit download script accesses Hugging Face; inference uses
`local_files_only=True`. Re-running the download resumes incomplete files.
The saved result includes checkpoint revision and quantization mode.

Other complete HF causal model folders can be supplied with `--model-dir`.
`models/base/` remains an empty example placeholder. CPU is the default when
`--device` is omitted; the NF4 option requires CUDA.

## Qwen3.5-0.8B comparison

This newer architecture needs a separate Python >= 3.10 / Transformers 5.19 runtime.
On this machine Python 3.11 is available in the existing `decision-pfn` environment;
only its interpreter is used to create a new isolated project environment:

```bash
/d/anaconda/envs/decision-pfn/python.exe -m venv .venv-qwen35
.venv-qwen35/Scripts/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv-qwen35/Scripts/python -m pip install -r requirements-qwen35.txt
.venv/Scripts/python scripts/download_model.py --model qwen3.5-0.8b
.venv-qwen35/Scripts/python -m src.run --model-dir models/Qwen3.5-0.8B --device cuda --output results/qwen35-08b.json
```

The official checkpoint is pinned to `2fc06364715b967f1860aea9cf38778875588b17`.
To run the 4B model in the same modern environment, add the CUDA-only
quantization backend and use the identical Transformers version:

```bash
.venv-qwen35/Scripts/python -m pip install bitsandbytes==0.46.1
.venv/Scripts/python scripts/download_model.py --model qwen3-4b
.venv-qwen35/Scripts/python -m src.run --model-dir models/Qwen3-4B-Instruct-2507 --device cuda --load-in-4bit --output results/qwen4b-t519.json
```

The local pilot scored the same 11 safe states for both models; it is a smoke
comparison on one map, not the held-out benchmark.
It loads the full conditional-generation model but only supplies text inputs.
Thinking is disabled by the chat template. Its roughly 1.75GB weights fit without
quantization; compare as **Qwen3.5-0.8B BF16 vs Qwen3-4B NF4**, not a controlled
parameter-count experiment. Both use the same map, reward, prompt and scoring rule.

JEV applies the tokenizer chat template when available, requests thinking off,
and appends an explicit `Answer:\n` assistant prefill. Raw base tokenizers without
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

Qwen3.8 currently refers here to the official `Qwen3.8-27B` checkpoint, which
has 28B parameters and BF16 weights; it is not feasible to run locally on this
RTX 3070 with 8GB VRAM. It uses the same `qwen3_5` architecture identifier as
Qwen3.5 and the modern loader selects that architecture's multimodal model class.
Transformers 5.19 is pinned in the newer runtime. Qwen3.5-0.8B is the local
comparison model; 27B local inference is not verified.
For Huawei Ascend NPU setup, see [NPU_ASCEND.md](NPU_ASCEND.md). The code imports
`torch_npu` only when `--device npu` is requested and synchronizes NPU timings.

## Local HTTP service and bulk maps

Use the modern Python 3.11 / Transformers 5.19 environment for this workflow.
The model remains loaded between HTTP requests. The service binds only to
`127.0.0.1`, defaults to port 8000, and processes batches sequentially on one device.

```bash
.venv-qwen35/Scripts/python -m src.serve --model-dir models/Qwen3.5-0.8B --device cuda --port 8000
```

Run the following in a second Git Bash terminal:

```bash
curl http://127.0.0.1:8000/health
.venv-qwen35/Scripts/python -m src.evaluation.dataset --maps 1000 --seed 42
.venv-qwen35/Scripts/python -m src.evaluation.bulk --batch-size 8
```

The dataset command refuses to overwrite a file; use a new `--output` to create
another dataset. The default dataset contains 1,000 distinct maps (500 each of
8x8 and 12x12), with one uniformly sampled safe state at distances 1, 2, 4 and 8
per map. Accepted maps must have a reachable start and all four distance groups.
This conditioning is part of the sampling protocol. Each map's VI labels are
cross-checked against BFS before writing. Seed 42 fixes map generation and state
sampling. Every fifth map is development data; the other 800 maps are test data.
This keeps both sizes balanced and never splits states from a map across sets.

`GET /health` returns readiness and the model/runtime fingerprint. `POST /score`
accepts `{"prompts": ["...", "..."]}` and returns `predictions`. Prompts should be
built with `src.agents.prompt.format_prompt`. Batches contain 1..16 inputs with
at most 2,048 tokens each. The service uses one forward pass for a batch and
only requests last-position logits when the model supports it. `forward_seconds`
is the amortized batch time per state; `batch_forward_seconds` and `batch_size`
are also saved. BF16 scores can differ slightly between batch sizes; keep the
same batch size throughout a comparison.

Bulk scoring writes JSONL predictions after each batch to
`results/predictions/qwen35-1000.jsonl`. Re-running the same command resumes that
cache. A manifest locks the dataset checksum, checkpoint, framework, scoring
code and batch size; changes require a different `--output`. A torn final line
from interruption is discarded on resume. `--limit 32` runs an optional pilot;
remove it to score the remaining states. Stop the foreground service with Ctrl+C.

The adjacent `.summary.json` reports dev/test separately, distance groups, map
sizes, unique-optimum subsets, accuracy, Q-regret, optimal probability mass,
binary Brier score, 10-bin ECE, and random-policy expectations. Accuracy and
regret intervals use 1,000 map-level bootstrap resamples. Raw predictions are
retained so analysis does not require another model run. Model weights, server
logs, datasets and predictions stay under ignored local directories.

For Ascend, use the NPU environment from `NPU_ASCEND.md` and start the same
service with `--device npu:0`. NPU execution remains unverified on this machine.

## Detailed state format (v2)

Every newly generated record includes `state_text` and
`prompt_version: frozenlake-state-v2`. The numeric `state` is still the Gymnasium
state ID. `state_text` is the complete model-facing input: full rules, tile
legend, zero-based coordinates, player and goal positions, row/column-labeled
map, deterministic moves, out-of-bounds self-loops, terminal holes and goal,
reward, discount, infinite-horizon objective, and letter-to-action mapping.
It is derived only from the visible board and gamma. Oracle values, distances,
and optimal-action labels are separate fields and are never put into this text.
See [the complete example](docs/state-example.md).

The HTTP service also accepts structured inputs, for example:

```json
{"states": [{"board": "PF\nFG", "gamma": 0.99}]}
```

Send this JSON to `POST /score` instead of `prompts`; the server builds the full
state description. Existing raw `prompts` requests remain available. `/health`
reports the active prompt version. Restart the service after editing prompt code.

The existing 1,000-map data have been regenerated with identical maps, sampled
states and labels at `results/datasets/frozenlake-1000-state-v2.jsonl`. New scores
must use a fresh output file; results from the original short prompt remain an
archived baseline. To score the detailed states:

```bash
.venv-qwen35/Scripts/python -m src.evaluation.bulk --dataset results/datasets/frozenlake-1000-state-v2.jsonl --output results/predictions/qwen35-1000-state-v2.jsonl --batch-size 8
```

The v2 prompt and structured HTTP path were checked on eight development states.
The complete 4,000-state v2 inference is recorded in [the comparison report](docs/bulk-state-v2.md). This is a revised prompt
on the same fixed maps, so a future comparison is exploratory; it is not a fresh
untouched test set. The bulk client rejects stale `state_text` and mismatched
cache manifests rather than mixing prompt versions.

## Full games: 30-move episodes

The 4,000-state benchmark samples four independent positions per map; it does
not measure game completion. To play each of the 1,000 maps from its actual S:

```bash
.venv-qwen35/Scripts/python -m src.evaluation.episodes --max-steps 30 --batch-size 16
```

Start the local model service first as described above. Each map has exactly one
episode. At every step the model sees the complete current board, game rules,
and remaining move budget (`frozenlake-episode-v1`). Choose argmax among the four
choice-token probabilities, execute that action, and repeat until G, H, or 30
moves. Reaching G on move 30 succeeds. Boundary collisions consume moves; loops
are allowed to continue until the budget expires. Batches contain independent
games, never future steps from the same game.

Finite-horizon Bellman backups supply Q values for each remaining budget.
These labels are recorded for analysis and never supplied to the model.
Maps whose shortest safe S-to-G path exceeds 30 moves remain in the evaluation;
reports also give success among maps that an optimal policy can solve in time.
The existing dev/test map split is preserved.

Output `results/episodes/qwen35-max30.jsonl` contains every executed transition,
full model prompt/probabilities, finite-horizon Q labels, and terminal/timeout
flags. Adjacent `.episodes.json`, `.summary.json`, and `.meta.json` files contain
game outcomes, aggregate results, and reproducibility settings. Re-running the
same command validates and replays cached actions, then resumes unfinished
games without scoring saved steps again. Changed settings require a new output.
Raw trajectories and local models are ignored by Git.
