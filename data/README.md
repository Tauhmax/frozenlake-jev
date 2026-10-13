# Zero-shot decision-token evaluation data

五个公开数据集，每个固定抽取 400 道四选一题目，合计 2,000 题。只保留 `eval`，不设开发集、不提供 few-shot 示例。四组 token 由预先固定的规则选择，答案仅用于运行后的评分，不参与 token 选择、提示词优化或概率校准。

| 目录 | 任务 | 上游划分 |
| --- | --- | --- |
| `arc_challenge/` | 科学知识与推理 | ARC-Challenge test |
| `openbookqa/` | 基础科学与常识组合 | main test |
| `hellaswag/` | 常识性事件续写 | validation（带公开答案） |
| `race/` | 篇章阅读理解 | all validation |
| `mmlu/` | 57 学科知识与推理 | all validation |

这里的 `eval` 是本项目的评估用途；上游叫 validation 不表示我们使用它做开发调参。这是抽样诊断集，不是任何上游完整 benchmark 的官方分数，也无法保证这些公开题目未进入模型预训练。

## 获取与复现

本机已将题目下载到各目录的 `eval.jsonl`。数据正文和原始下载缓存留在本地，Git 保存脚本、固定来源和数据清单。迁移到 NPU 机器可复制整个 `data/`，或者从仓库根目录执行：

```bash
python scripts/prepare_decision_data.py
```

下载脚本只需要 Python 3.11 标准库，不安装依赖、不修改模型环境。已下载分页缓存在 `.cache/`，重新运行时复用。`sources.json` 固定来源 revision，下载时核对服务返回的 revision；`manifest.json` 保存来源行数、筛选统计、类别和答案分布以及每个结果文件的 SHA-256。

参数 `--eval-size 400 --seed 42` 控制每个数据集的最大样本数与确定性抽样。候选不足时保留实际数量，不重复补齐。

## 格式与抽样

每条记录有 `id`、`dataset`、`category`、`context`、`question`、`choices`、`answer`、`source`。`choices` 保留上游顺序；`answer` 为 0 至 3 的正确选项索引。原始题文不改写、不翻译；HellaSwag 使用固定的续写问题指令。

只取恰有四个非空、互异选项且答案有效的题目；按题文和选项进行精确去重。每个文章/视频/问题分组只取一道题，避免把相关题目当成独立样本。按 subject/activity/阅读级别轮换取样，MMLU 覆盖全部 57 个学科；因此不是按原始类别比例随机抽样。所有策略使用相同题目。

## 运行探针

```bash
python -m decision_tokens.probe \
  --model-dir /path/to/Qwen3.8-27B --device npu:0 \
  --datasets data/*/eval.jsonl \
  --output results/decision_tokens/multidataset.jsonl
```

工具为每个策略动态插入四个候选 token ID 作为选项标签，题目中的 `answer` 不进入模型输入。每道题运行无插入基线与四种 marker 插入条件，并记录正确性和分布变化。默认四组策略共 40,000 次单样本前向；这里仅准备数据，不启动推理。

## 来源

- [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc)
- [OpenBookQA](https://huggingface.co/datasets/allenai/openbookqa)
- [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag)
- [RACE](https://huggingface.co/datasets/ehovy/race)
- [MMLU](https://huggingface.co/datasets/cais/mmlu)

来源许可元数据保留在 `sources.json`；各数据集遵循其原始发布条款。
