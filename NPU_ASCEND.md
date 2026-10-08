# Ascend NPU recipe

This recipe targets **Huawei Ascend NPU** through PyTorch/TorchNPU and scores
`Qwen3.5-0.8B` in BF16 on the same local FrozenLake prompts. It does not claim
NPU performance or operator compatibility has been verified in this Windows RTX
workspace. Use a supported Linux host with an installed Ascend driver, firmware,
and CANN toolkit.

## Install a matched runtime

Choose the Python, CANN, PyTorch and TorchNPU versions from the official
[compatibility matrix](https://www.hiascend.com/document/detail/en/Pytorch/700/comref/Envvariables/Environvariables_0001.html)
for the host's exact Ascend card and OS. Install CANN and driver by the official
Ascend instructions, then use the matching PyTorch and TorchNPU wheels. Do not
install the CUDA wheel or the project's `requirements-model.txt` in this NPU env.

For example, the current TorchNPU source quickstart shows CANN 9.1.0 paired with
PyTorch 2.12.0 and TorchNPU 2.12.0; this is only an example and may not match
older cards/drivers. Follow the compatibility table when versions differ.

```bash
source /usr/local/Ascend/cann/set_env.sh
python3 -m venv --system-site-packages .venv-npu
source .venv-npu/bin/activate
python -m pip install --upgrade pip
# Install the matching torch wheel and torch-npu wheel for your CANN/card here.
python -m pip install --no-deps -r requirements-qwen35.txt
python -m pip install huggingface-hub==0.30.2
python -c 'import torch, torch_npu; print(torch.__version__, torch.npu.is_available())'
```

Download the weights on a connected machine (or use already-downloaded local
weights) into `models/Qwen3.5-0.8B`. Model loading itself is local-only.

```bash
python scripts/download_model.py --model qwen3.5-0.8b
python -m src.run \
  --model-dir models/Qwen3.5-0.8B \
  --device npu:0 \
  --output results/qwen35-08b-npu.json
```

The model loader imports `torch_npu`, transfers the model and input IDs to NPU,
and synchronizes NPU before and after a forward pass so recorded latency includes
asynchronous device work. Do not pass `--load-in-4bit`: bitsandbytes NF4 in this
project is CUDA-only. The 0.8B BF16 checkpoint is the first NPU validation target.

## NPU smoke checks

```bash
npu-smi info
python -c 'import torch, torch_npu; x=torch.ones(2, device="npu:0"); print(x, torch.npu.is_available())'
python -m src.run --model-dir models/Qwen3.5-0.8B --device npu:0 --output results/npu-smoke.json
```

If `Qwen3_5ForConditionalGeneration` has an unsupported operator on the selected
CANN/TorchNPU pair, use an Ascend-published Qwen3.5 container with matching CANN
or wait for that pair's support; do not fall back to CPU silently. Upstream
Ascend examples currently publish SGLang containers for Atlas A2/A3, but the
choice-token benchmark needs direct next-token logits, so this recipe uses the
Transformers model forward path instead of text generation through a serving API.
