# 30-step FrozenLake: Qwen3.5-0.8B vs Qwen3-4B

Both completed one episode from S on each of the same 1,000 deterministic maps with the same dataset SHA256, detailed episode prompt, scoring code, batch size 16, gamma 0.99 and 30-move limit. Every recorded trajectory was replayed in Gymnasium and its endpoint checked.

| Outcome | 0.8B BF16 | 4B NF4 | Change |
|---|---:|---:|---:|
| success | 225/1000 (22.5%) | 298/1000 (29.8%) | +7.3 pp |
| hole | 640/1000 (64.0%) | 648/1000 (64.8%) | +0.8 pp |
| timeout | 135/1000 (13.5%) | 54/1000 (5.4%) | -8.1 pp |

| Map group | 0.8B success | 4B success | Change |
|---|---:|---:|---:|
| 8×8 (n=500) | 31.80% | 39.40% | +7.60 pp; paired bootstrap 95% [+4.00, +11.20] pp |
| 12×12 (n=500) | 13.20% | 20.20% | +7.00 pp; paired bootstrap 95% [+3.80, +10.20] pp |
| dev (n=200) | 22.00% | 30.00% | +8.00 pp; paired bootstrap 95% [+1.50, +14.00] pp |
| test (n=800) | 22.62% | 29.75% | +7.12 pp; paired bootstrap 95% [+4.62, +9.75] pp |

Paired outcomes (rows 0.8B; columns 4B):

| 0.8B \ 4B | Success | Hole | Timeout |
|---|---:|---:|---:|
| success | 183 | 36 | 6 |
| hole | 76 | 524 | 40 |
| timeout | 39 | 88 | 8 |

Total model decisions: 8061 (0.8B), 4545 (4B). Mean episode lengths: 8.06 vs 4.54. Counts differ because trajectories and endpoints differ.

| Diagnostic | 0.8B | 4B |
|---|---:|---:|
| Optimal first action on same 1,000 start states | 478/1000 (47.8%) | 675/1000 (67.5%) |
| Entered hole on first action | 207/1000 (20.7%) | 230/1000 (23.0%) |
| UP actions / all decisions | 5315/8061 (65.9%) | 919/4545 (20.2%) |
| Finite-horizon optimal action on solvable visited states | 2096/6773 (30.9%) | 2554/4215 (60.6%) |
| Hole entries when goal was still reachable | 616 | 647 |
| Hole entries with a safe alternative | 640 | 648 |

Interpretation: 4B improves closed-loop success but does not materially reduce the absolute number of hole endings in this run. The 4B run has fewer timeouts and shorter episodes, including more first-step hole entries. First-action diagnostics use exactly the same 1,000 start states; later visited-state diagnostics compare different paths and are descriptive rather than paired accuracy tests. All 1,000 start states have an oracle route to G within 30 steps.

The models differ in architecture/checkpoint and precision (0.8B BF16 vs 4B runtime NF4); this is not an isolated parameter-count or quantization effect. The map set was reused from earlier development and evaluation, so intervals describe map-to-map variation, not a new untouched test set. Neither model generates reasoning tokens.

Artifacts (ignored by Git): `results/episodes/qwen35-max30.*` and `results/episodes/qwen4b-max30.*`. Both manifests record checkpoint revisions, framework versions, prompt/scoring hashes and dataset checksum.

Dataset SHA256: `4d64088cc44a62b5c676d89389c6cf75c3d580d03979e5c4151b1b73d347c7e0`.
