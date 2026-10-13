# 能否用 decision token 触发已学到的推理？

这是待检验的假设，不把答案概率变化直接解释成“模型做了更多计算”。本项目使用冻结的原厂 post-trained Qwen，目标是一次前向后读取四个候选 token 的分数；不训练新的 pause token，也不生成中间推理文本。

## 文献给出的线索和边界

- [Zero-shot-CoT](https://papers.neurips.cc/paper_files/paper/2022/hash/8bb0d291acd4acf06ef112099c16f326-Abstract-Conference.html)：简短的推理提示能改善一些复杂题，但原方法让模型**生成**思维链。它不证明把同类词放进前缀后，只读一次 choice logits 也会受益。
- [CoT-Decoding](https://papers.neurips.cc/paper_files/paper/2024/hash/7a8e7fd295aa04eac4b470ae27f8785c-Abstract-Conference.html)：改变解码路径可显露模型已有的推理行为；这种方法同样使用额外生成路径。
- [Pause Tokens](https://arxiv.org/abs/2310.02226)和[Quiet-STaR](https://arxiv.org/abs/2403.09629)：额外 token 可提供计算步骤，但论文方法依赖额外训练或生成。不能把未经该训练的 Qwen 稀有 token 当成天然 pause token。
- [Qwen3.8-27B 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)：模型提供原生 thinking 控制。这是比较时的重要正对照；当前探针使用 raw token ID，尚未调用官方 chat template 的 thinking 路径。

## 可证伪的假设

对固定题目、四个选项和标签映射，比较末尾 choice logits：

1. **标签本身的作用。** 前文同一位置换入各 decision token；观察置换后的准确率、正确选项 log-odds 和候选总概率质量。当前探针已经支持这一层。
2. **推理提示的作用。** 在标签之外增加预先固定的短提示，如 `Think carefully` 或 `Check each option`；与 token 数、位置相同的中性提示对照。提示词必须在评估前固定，不能根据正确答案挑选。
3. **交互作用。** 同一道题做“token 集合 × 推理提示 × 标签映射”的配对比较。若只有 `A/B/C/D` 的某一字母概率升高，而映射回实际选项后不增益，说明是标签偏差。若多个任务、多个映射中的难题正确率稳定上升，才构成支持“已有决策能力被提示调动”的行为证据。
4. **机制对照。** 另跑 Qwen 原生 thinking 模式和自由生成思维链，记录生成 token 数及正确率；它们提供上限和计算成本对照，不与单次前向结果混作同一方法。生成 CoT 有效而直接读 logits 无效，也是一项有价值的否定结果。

优先使用现有五个数据集各 400 题，同题配对。按数据集和原始题目长度/难度代理分层，报告准确率、正确选项 log-odds、置换稳定性和推理成本。题目公开且未经独立难度标注，因此不能把长度直接称为推理复杂度；后续可按模型在中性条件下是否做错进行探索性分析，但不能在该集合上挑选最终 token。

**成功标准：** 在冻结模型、固定提示、固定 token 策略且不生成中间文本时，多个数据集和标签映射上的配对准确率有所提升，且不以更大的标签先验失衡为代价。即使达到该标准，也只能说“单次前向的选择表现改善”；还需要内部状态干预等进一步证据，才能声称识别出具体的摊销推理机制。
