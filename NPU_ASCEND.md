# Ascend 910B2 / CANN 9.0.1

## 固定环境

| 组件 | 版本 / 要求 |
|---|---|
| 芯片 | Ascend 910B2 |
| 系统 | Linux，x86_64 或 aarch64，glibc ≥ 2.28 |
| Python | 3.11 |
| CANN | 9.0.1；驱动、固件及算子包须匹配该芯片 |
| PyTorch | 2.10.0+cpu |
| torch-npu | 2.10.0.post2 |
| Transformers | 5.19.0，与 CUDA 使用同一份公共依赖 |
| 模型 | 原始 BF16；不使用 CUDA bitsandbytes |

运行库组合依据 [Ascend 官方项目的固定版本安装表](https://github.com/vllm-project/vllm-ascend/blob/v0.23.0rc1/docs/source/installation.md)；torch-npu 的 Python 3.11 wheel 已确认支持上述两种 CPU 架构。本项目不安装 vLLM，直接调用 Transformers forward。**此配方尚未在目标 NPU 上验证实际模型前向。**

CUDA 保留已验证的 PyTorch 2.6.0/cu124；NPU 按 CANN 使用 2.10.0。公共 Python 包与模型版本不分叉。

## 安装

先确认管理员已安装匹配的驱动、固件、CANN 和算子包。在仓库根目录执行：

```bash
npu-smi info
uname -m
ldd --version
source /usr/local/Ascend/cann/set_env.sh  # 按实际安装路径调整
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-npu.txt
python -m pip check
```

每个新终端都要加载 CANN 环境并激活 `.venv`。其他 CANN 版本不能直接复用此配方；必须同步核对 PyTorch 与 torch-npu。不要复制 Windows 虚拟环境到 Linux。

## 验收与运行

```bash
python - <<'PY'
import torch
import torch_npu
import transformers
print(torch.__version__, torch_npu.__version__, transformers.__version__)
assert torch.npu.is_available()
x = torch.ones((2, 2), device='npu:0', dtype=torch.bfloat16)
print((x @ x).cpu())
torch.npu.synchronize('npu:0')
PY
```

复制完整模型目录（含 tokenizer、`checkpoint.json`）与固定数据集到目标机器，或按 README 下载、生成。先做实际模型前向检查，再启动服务：

```bash
python -m src.run --model-dir models/Qwen3.5-0.8B --device npu:0 --output results/npu-smoke.json
python -u -m src.serve --model-dir models/Qwen3.5-0.8B --device npu:0 --port 8000
```

4B 将 `--model-dir` 改为 `models/Qwen3-4B-Instruct-2507`。BF16 4B 仅权重约 7.5 GiB，设备还需容纳前向工作空间。不传 `--load-in-4bit`。遇到不支持的算子应核对运行库，不隐式回退 CPU。

看到 `READY` 后，在第二个同环境终端执行：

```bash
python -u -m src.evaluation.episodes --dataset results/datasets/frozenlake-1000-state-v2.jsonl --output results/episodes/qwen35-npu-max30.jsonl --max-steps 30 --batch-size 16
```

4B 使用新输出名 `qwen4b-npu-max30.jsonl`。批次按设备内存选择；精度、设备或批次改变时使用新缓存。NPU BF16 与 CUDA NF4 是不同精度的比较。

保留运行环境记录：

```bash
mkdir -p results/environment
python -m pip freeze > results/environment/npu-packages.txt
npu-smi info > results/environment/npu-device.txt
curl -fsS http://127.0.0.1:8000/health > results/environment/npu-service.json
```

另记录实际 OS、CANN、驱动和固件版本；它们不包含在 `pip freeze` 中。基础矩阵乘法成功不能代替 Qwen 模型前向验收。
