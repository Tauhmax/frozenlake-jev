# Qwen3.5-0.8B bulk pilot (2026-10-08)

Completed 1,000 unique maps / 4,000 state scores, fixed seed 42. Both map sizes (8x8, 12x12) have 500 maps. One safe state is sampled per map at each shortest-path distance 1, 2, 4, 8. Accepted maps must have a reachable start and contain all four distance bins. Labels use deterministic Gymnasium, value iteration with gamma 0.99, and an independent BFS cross-check.

The frozen split contains 200 development maps and 800 test maps. The table below uses only the 3,200 test states; there is no map overlap with development. This is the first fixed-prompt run, not a result of prompt tuning on the test set.

Runtime: Python 3.11.16, PyTorch 2.6.0+cu124, Transformers 5.19.0, RTX 3070, BF16, batch size 8. Thinking is disabled; one forward pass per batch scores A/B/C/D, with no generated tokens. Optional optimized linear-attention kernels are absent; PyTorch reference operations are used.

Checkpoint: `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`.
Dataset SHA256: `10d42e449dc7b7b7ca1d21e2ca0b8e992112f6af139d880c9e896c79e25bb9f4`.

| Distance | Maps/states | Accuracy | Random expected | Mean Q-regret | Unique-only accuracy (N) |
|---|---:|---:|---:|---:|---:|
| 1 | 800/800 | 26.88% | 25.00% | 0.1435 | 26.88% (800) |
| 2 | 800/800 | 35.50% | 32.53% | 0.1818 | 28.62% (559) |
| 4 | 800/800 | 37.38% | 34.72% | 0.2159 | 23.93% (489) |
| 8 | 800/800 | 38.62% | 35.81% | 0.2148 | 29.00% (462) |

Overall accuracy: 34.59%; map-clustered bootstrap 95% interval [32.72%, 36.56%] (1,000 resamples). Uniform-random expected accuracy: 32.02%. Mean regret: 0.1890. Mean confidence: 33.39%; 10-bin ECE: 0.0120.

**The model mostly chooses one action:** LEFT=0, DOWN=0, RIGHT=4, UP=3196. This run shows strong action/label preference, not demonstrated map-sensitive planning. The higher accuracy at longer distances coincides with more tied optimal actions (and a higher random baseline); it is not evidence that longer planning is easier. Option-permutation controls are needed before attributing the preference to direction versus label.

Batched vs singleton scoring was checked on 8 development states: actions matched; maximum probability difference was 0.02793 in BF16. Batch size is frozen in the cache manifest. Measured end-to-end throughput was approximately 37 states/s after warm-up. Inference completed for all 4,000 states and resuming reported `Cached 4000, pending 0`.

Local artifacts (ignored by Git): `results/datasets/frozenlake-1000.jsonl`, `results/predictions/qwen35-1000.jsonl`, matching `.meta.json` / `.summary.json`, and `results/server/batch-check.json`.

Commands and HTTP interface are documented in `README.md`. Service: `http://127.0.0.1:8000`; readiness: `GET /health`; scoring: `POST /score`.
