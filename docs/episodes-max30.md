# Qwen3.5-0.8B: complete 30-move games

One episode starts at the actual S tile of each of the same 1,000 maps. The agent receives the updated full map, detailed rules and remaining moves on every decision. Greedy choice-token scoring; no generated reasoning or action sampling. Goal and hole terminate immediately, otherwise the episode truncates after exactly 30 actions. Loops and edge collisions consume the budget.

Runtime: Qwen3.5-0.8B BF16 on RTX 3070; Transformers 5.19.0; batch size 16. Prompt `frozenlake-episode-v1`; gamma 0.99. Each batch advances independent games by one step. Finite-horizon Bellman Q labels never enter model input.

| Split | Games | Success | Hole | Timeout | Success rate | Mean steps | Decisions |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 1000 | 225 | 640 | 135 | 22.50% | 8.06 | 8061 |
| dev | 200 | 44 | 119 | 37 | 22.00% | 9.30 | 1861 |
| test | 800 | 181 | 521 | 98 | 22.62% | 7.75 | 6200 |

| Map size | Games | Success rate | Oracle-solvable within 30 | Success among solvable |
|---|---:|---:|---:|---:|
| 8 x 8 | 500 | 31.80% | 500 | 31.80% |
| 12 x 12 | 500 | 13.20% | 500 | 13.20% |

Executed action counts (0 LEFT, 1 DOWN, 2 RIGHT, 3 UP): `{0: 438, 1: 1398, 2: 910, 3: 5315}`.

This measures closed-loop game completion, unlike the earlier 4,000 independent state decisions. The same previously evaluated map set is reused; this is an exploratory evaluation, not a new held-out test. All 1,000 selected maps have a safe S-to-G route within 30 moves, so the 22.50% result also equals success among oracle-solvable maps.

Validation: replayed every recorded action in Gymnasium and checked state transitions, remaining budgets, terminal/truncation flags and recorded model prompts. Every map appears exactly once, every episode has at most 30 actions, and every nonterminal endpoint is a 30-move timeout. Finite-horizon Q values agree with independent BFS on all 1,000 maps at budgets 1, 2, 8, 29 and 30.

Local full traces: `results/episodes/qwen35-max30.jsonl`. Game summaries: adjacent `.episodes.json`; aggregate report: `.summary.json`; model/checkpoint, source and dataset hashes: `.meta.json`. These generated artifacts are not committed.

Dataset SHA256: `4d64088cc44a62b5c676d89389c6cf75c3d580d03979e5c4151b1b73d347c7e0`.
