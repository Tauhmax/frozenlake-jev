# Detailed-state full evaluation (v2)

Completed all 4,000 states from the same 1,000 maps as the original pilot. Development: 200 maps / 800 states. Test: 800 maps / 3,200 states. Both 8x8 and 12x12 sizes are balanced; each map contributes one state at distances 1, 2, 4 and 8. All map IDs, states, splits and oracle labels match the prior dataset.

Model: Qwen3.5-0.8B, revision `2fc06364715b967f1860aea9cf38778875588b17`. Runtime: Python 3.11.16, PyTorch 2.6.0+cu124, Transformers 5.19.0, RTX 3070, BF16, batch size 8. One forward pass per batch; no generated tokens; reference PyTorch linear-attention implementation.

The only experimental change is the detailed `frozenlake-state-v2` input format: complete rules, reward and discount objective, coordinates, current/goal positions, and numbered map. This is an exploratory prompt comparison on a previously evaluated test set, not a fresh held-out confirmation.

| Test metric | Original short state | Detailed state v2 |
|---|---:|---:|
| Optimal-action accuracy | 0.3459 | 0.5281 |
| Mean Q-regret | 0.1890 | 0.1587 |
| Optimal probability mass | 0.3241 | 0.3478 |
| Mean confidence | 0.3339 | 0.3212 |
| Binary Brier score | 0.2270 | 0.2896 |
| ECE (10 bins) | 0.0120 | 0.2070 |

Accuracy change: +18.22 percentage points; paired map-bootstrap 95% interval [+16.72, +19.78] points (1,000 resamples). Uniform random expected accuracy on these labels: 32.02%.

| Distance | Maps/states | Old accuracy | V2 accuracy | Random expected | V2 Q-regret | V2 unique-only accuracy (N) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 800/800 | 26.88% | 64.00% | 25.00% | 0.0728 | 64.00% (800) |
| 2 | 800/800 | 35.50% | 58.50% | 32.53% | 0.1241 | 58.32% (559) |
| 4 | 800/800 | 37.38% | 44.88% | 34.72% | 0.2264 | 34.97% (489) |
| 8 | 800/800 | 38.62% | 43.88% | 35.81% | 0.2116 | 32.03% (462) |

| Selected action | Original count | V2 count |
|---|---:|---:|
| LEFT | 0 | 205 |
| DOWN | 0 | 288 |
| RIGHT | 4 | 319 |
| UP | 3196 | 2388 |

Different distance bins have different optimal-action tie counts; compare against the distance-specific random baseline and unique-optimum subset. Low ECE alone does not demonstrate planning or useful state-dependent confidence.

Dataset SHA256: `4d64088cc44a62b5c676d89389c6cf75c3d580d03979e5c4151b1b73d347c7e0`.
Scoring code SHA256: `a58f36f7c15b7c40882bd1d6445b096cc9fd5e49333ffda57101d3d091bdde9c`.

Local artifacts: `results/datasets/frozenlake-1000-state-v2.jsonl`, `results/predictions/qwen35-1000-state-v2.jsonl`, the adjacent `.meta.json` and `.summary.json`, and `results/predictions/qwen35-state-v2-distance.csv`. The original result files remain intact.
