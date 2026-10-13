# Decision-token probe

在原厂 post-trained Qwen（未经 JevAny 微调）上，选择四个候选 token，在前文同一位置依次插入它们，读取末尾 decision 位置的分布。工具与具体下游任务无关；只做前向，不生成推理文本。

## 四组策略

使用 `model.get_input_embeddings().weight` 的每一行计算 float32 L2 模长。这是输入 embedding 表，不是第一层 Transformer 的激活，也不是输出 LM head。

| 参数 | 选择规则 |
| --- | --- |
| `abcd` | A/B/C/D 分别编码得到的单 token ID |
| `longest` | 模长最大的四个 token，按模长降序 |
| `shortest` | 模长最小的四个 token，按模长升序 |
| `middle` | 按模长升序排列后，取中间连续四项，起始索引 `(N-4)//2` |

模长相同按 token ID 升序。默认从 tokenizer 词表中选择，包括特殊 token；排除 embedding 中没有对应词表 ID 的 padding 行。传入 `--exclude-special` 可排除特殊 token。结果包含 token ID、内部词表表示、解码文本、L2 模长、是否特殊 token。模长只是待验证的选取策略，不预设越大或越小越公平。

## 运行

沿用仓库现有 Python / Transformers 和设备依赖。`--model-dir` 指向本地完整原厂模型目录。以下为目标机器运行命令；27B 尚未在本机验证。

```bash
# 只选 token：仍会加载模型，以取得真实 input embeddings。
python -m decision_tokens.probe \
  --model-dir /path/to/Qwen3.8-27B --device npu:0 \
  --select-only --output results/decision_tokens/selection.jsonl

# 默认运行全部四组策略，并逐组打印进度。
python -m decision_tokens.probe \
  --model-dir /path/to/Qwen3.8-27B --device npu:0 \
  --templates decision_tokens/templates.jsonl \
  --output results/decision_tokens/probe.jsonl
```

CUDA 使用 `--device cuda:0`；CPU 使用 `--device cpu`。`--strategies abcd longest` 可只运行指定组；`--load-in-4bit` 沿用仓库 CUDA 量化加载方式。比较实验须保持模型、精度和模板一致。输出文件必须是新路径，以避免混合不同实验。

## 探针定义

每条 JSONL 模板给出 `id`、`prefix` 和 `suffix`：

```json
{"id":"copy","prefix":"Repeat the following marker exactly.\nMarker: ","suffix":"\nAnswer: "}
```

分别编码 prefix 和 suffix，构造 `prefix_ids + [candidate_id] + suffix_ids`。末尾最后一个位置的 logits 预测下一个 token。四个变体仅插入的 ID 不同，周围 token ID、长度和位置完全一致。直接插 ID 能测量不可打印、字节片段或特殊 token，避免解码再编码改变实验对象。它是精确 ID 干预，不等价于把整个拼接字符串重新 tokenize。

输入是 raw prompt；不自动套 chat template。需要 chat 格式时在模板中明确给出完整前后缀。原厂权重和提示格式都属于实验条件。

每组额外运行一次不插入 token 的基线。这个基线比其他输入短一个 token，因此基于它的概率差也包含长度/位置改变；主要成对比较应使用四个等长插入变体。示例提供：

- `copy`：有效指令要求重复前文 marker，观察指令响应。
- `irrelevant`：无关文档出现 marker，随后完成固定句子，观察无关提及的影响。它不要求输出候选 token，因此候选总概率可能很低。

## 输出与解释

JSONL 第一行为模型配置和四组所选 token；之后每行为一个模板 × 策略。保存 prefix/suffix ID、插入位置（prefix 长度）、距离（suffix 长度）、每个干预的候选 logits、全词表 log probabilities、候选集内 softmax、候选总概率质量，以及相对无插入基线的概率变化。

`centered_logit_influence` 是 4×4 矩阵，行是前文插入 token，列是末尾候选 token。令 `L[i,j]` 为原始 logit：先减去每行均值，再减去每列均值，得到 `M`。因此 `M[i,j]` 衡量相对其他插入条件，token i 对候选 j 的相对提升；消除了所有候选共同的 logit 平移和固定的列先验。对角强表示自增强，非对角项显示交叉影响。原始分布一并保留，以观察被居中操作消除的先验偏好。

此版严格读取选定的四个 token ID，不合并解码相同的其他 ID。它使用 JevAny 式的候选内归一化，但刻意区别于 [JevAny 标签字符串的 exact-decoding 聚合](https://simplejev.org/JevAny/docs/CHOICE_READOUT.html)：embedding 排名的实验单位是 token ID。

首轮比较各组在有效指令中的响应和无关上下文中的偏移，然后增加不同前后文和距离。不能凭一个模板或 embedding 模长本身推断跨任务优劣。

## 验证

```bash
python -m unittest tests.test_decision_tokens -v
python -m ruff check decision_tokens tests/test_decision_tokens.py
python -m ruff format --check decision_tokens tests/test_decision_tokens.py
```
