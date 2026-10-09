# frozenlake-jev

在确定性 FrozenLake 上测试本地语言模型的决策能力。模型读取完整地图、游戏规则和剩余步数，单次前向计算 A/B/C/D 的 next-token 分数；不生成推理文本。Ground Truth 由 Gymnasium 转移表上的 Bellman 更新得到。

## 环境：所有模型共用一套版本

统一使用 **Python 3.11、Transformers 5.19.0**。Qwen3-4B 与 Qwen3.5-0.8B 已在这套框架上运行，不再按模型维护旧版 Transformers 环境。其他模型仍需验证其 tokenizer、架构和算子支持，不能仅凭版本号推断兼容。

| 文件 | 用途 |
|---|---|
| `requirements.txt` | 所有模型与设备共享的依赖；不安装硬件运行库 |
| `requirements-cuda.txt` | 公共依赖 + PyTorch 2.6.0/cu124 + bitsandbytes 0.46.1 |
| `requirements-npu.txt` | 公共依赖 + PyTorch 2.10.0 / torch-npu 2.10.0.post2；910B2 / CANN 9.0.1 |
| `requirements-dev.txt` | 公共依赖 + Ruff；用于开发检查 |

CUDA 已验证：Windows、Python 3.11.16、RTX 3070 8GB。NPU 的主机条件、安装和验收步骤见 [NPU_ASCEND.md](NPU_ASCEND.md)，尚未在实体 NPU 验证。

在仓库根目录操作。Windows 使用 Git Bash，Linux 使用 Bash；创建环境的 `python` 必须是 3.11：

```bash
python --version
python -m venv .venv
source .venv/Scripts/activate       # Windows Git Bash
# source .venv/bin/activate         # Linux
python -m pip install --upgrade pip
python -m pip install -r requirements-cuda.txt
python -m pip check
```

已有的 Python 3.11 / Transformers 5.19 环境可以继续使用，目录名不影响功能。正在运行实验时不要原地更换依赖。只有 CPU 时，先从 PyTorch CPU index 安装 `torch==2.6.0`，再安装 `requirements.txt`。

## 模型

```bash
python scripts/download_model.py --model qwen3.5-0.8b
python scripts/download_model.py --model qwen3-4b
```

下载脚本固定 checkpoint revision，模型目录包含 `checkpoint.json`。也可将完整本地 Hugging Face 模型目录复制到目标机器；推理只读本地文件。模型文件与结果不提交到 Git。

| 模型 | 本机运行方式 | 已验证范围 |
|---|---|---|
| Qwen3.5-0.8B | BF16 | CUDA 单状态及 1,000 局实验 |
| Qwen3-4B-Instruct-2507 | NF4 | CUDA 加载及 choice-token 推理 |

`--load-in-4bit` 使用 bitsandbytes 在加载时量化原始权重，NF4 + double quantization、FP16 计算，仅支持 CUDA。它不改写磁盘权重。NPU 使用未量化权重，不能传此参数。两种精度下的结果不能仅解释为参数规模差异。

## 完整游戏测试：1,000 张地图，每局最多 30 步

第一个终端启动服务，选择其中一个模型：

```bash
python -u -m src.serve --model-dir models/Qwen3.5-0.8B --device cuda --port 8000
# 或：
python -u -m src.serve --model-dir models/Qwen3-4B-Instruct-2507 --device cuda --load-in-4bit --port 8000
```

看到 `READY` 后，第二个已激活环境的终端运行：

```bash
curl http://127.0.0.1:8000/health
python -m src.evaluation.dataset --maps 1000 --seed 42 --output results/datasets/frozenlake-1000-state-v2.jsonl
python -u -m src.evaluation.episodes --dataset results/datasets/frozenlake-1000-state-v2.jsonl --output results/episodes/qwen35-max30.jsonl --max-steps 30 --batch-size 16
```

已有同名数据集时直接复用，跳过生成命令。测试 4B 时将输出改为 `results/episodes/qwen4b-max30.jsonl`。若目标设备无法容纳当前批次，减小 `--batch-size` 并使用新输出文件。

每张地图从真实 S 开始。每次输入包含更新后的位置、完整规则和剩余预算；选择四个动作中分数最大的一个并执行。进入 G 成功、进入 H 失败，其余情况到 30 步超时；第 30 步进入 G 仍成功，撞边界和重复访问也消耗步数。模型不接收 oracle 标签。有限步数 Q 值随剩余预算变化。

生成器固定种子，地图大小 8×8/12×12 各半；接受起点可达且包含距目标 1/2/4/8 步状态的地图。每五张中的一张划为 dev，共 200 dev / 800 test，按地图划分。数据集每图四条单状态记录；整局执行器按地图去重，仍只从 S 跑一局。

输出 `.jsonl` 保存每个实际动作、位置转移、完整提示、概率和 Q 值；相邻 `.episodes.json`、`.summary.json`、`.meta.json` 保存整局结果、汇总及配置。相同命令可续跑；模型、代码、提示词、精度或批次改变时使用新输出。缓存和 manifest 必须一起保留。不同硬件不保证逐位相同。

## 单状态诊断与接口

```bash
python -m src.run --output results/oracle.json
python -m src.evaluation.bulk --dataset results/datasets/frozenlake-1000-state-v2.jsonl --output results/predictions/qwen35-state-v2.jsonl --batch-size 8
```

`src.run` 使用 `configs/frozenlake.json`，默认只算 oracle，可传 `--model-dir` 和 `--device`。`bulk` 独立评分每图抽样的四个位置，不计算通关率。已有输出不会被混入不同配置的结果。

服务仅绑定 `127.0.0.1`。`GET /health` 返回模型与框架版本；`POST /score` 接收以下一种格式：

```json
{"states": [{"board": "PF\nFG", "gamma": 0.99, "remaining_steps": 30}]}
```

也支持 `{"prompts": ["完整提示词"]}`。省略 `remaining_steps` 表示无限时域单状态评分。每批 1–16 个输入，每条最多 2,048 tokens。A/B/C/D 对应 LEFT/DOWN/RIGHT/UP；每个选项必须恰好是一个 next token。输出概率仅在四个动作间归一化，不等同于校准后的正确率。完整提示示例见 [state-example.md](docs/state-example.md)。

## 结构与检查

`src/environments/` 管理地图与转移；`src/oracles/` 计算最优 Q；`src/agents/` 管理提示及模型评分；`src/evaluation/` 管理数据、单状态及整局评估。`models/` 和 `results/` 是本地生成目录。

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
```

保留三个聚焦测试：BFS 与 VI、终止和边界规则；有限时域与第 30 步终止；choice-token 位置验证。NPU 需要在目标机器额外完成实际模型前向验证。

## 实验记录

- [原始短提示单状态实验](docs/bulk-pilot.md)
- [详细提示单状态对比](docs/bulk-state-v2.md)
- [Qwen3.5-0.8B：1,000 局、30 步上限](docs/episodes-max30.md)：成功 225、掉洞 640、超时 135。
- [0.8B 与 4B 的同地图整局对比](docs/episodes-qwen-comparison.md)：4B 成功 298、掉洞 648、超时 54。

这些实验记录保留实际运行时版本与数据哈希；同一批已评估地图上的比较属于探索性结果。
