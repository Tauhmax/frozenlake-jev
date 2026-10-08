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
