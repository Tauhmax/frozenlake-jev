# System-1 Decision Planning Evaluation on FrozenLake

> **Status:** Research / implementation plan (v0.1)  
> **Primary environment:** FrozenLake (deterministic first)  
> **Target model:** Qwen3.5-27B, train-free direct choice-token scoring  
> **Central question:** Under a fixed inference budget, how do the decision quality and confidence of a System-1 model change as planning horizon and planning complexity increase?

## 1. Motivation and research questions

LLMs have strong language understanding and zero-shot generalization, but autoregressive decision generation can be slow and imperfectly constrained, and token probabilities need not represent calibrated probabilities of making an optimal decision. A **train-free direct choice-token model** can use one forward pass to score a fixed action space without generating a long response.

We use **FrozenLake** as a controlled testbed because its states, transitions, legal actions, shortest-path distances, and optimal action values are computable. It allows us to test the limits of direct, System-1-like action selection without conflating planning with open-ended text generation.

**Primary research questions**

- **RQ1 — Planning horizon:** Does direct choice-token action quality decline as the shortest safe-path distance to the goal grows (1, 2, 4, 8 steps)?
- **RQ2 — Planning complexity:** At the same distance, is the model worse on maps that require detours or looking past misleading local cues?
- **RQ3 — Confidence:** Does the action probability assigned by the model predict whether its chosen action is optimal? Does calibration degrade with distance or complexity?
- **RQ4 — Relative performance:** How does the direct model compare with simple heuristics, bounded-depth search, and learned RL policies?

**Working hypotheses (not presumed findings)**

- **H1:** Mean optimal-action accuracy decreases and Q-regret increases as goal distance increases.
- **H2:** Holding distance constant, performance worsens on cases requiring more nonlocal planning.
- **H3:** Model choice probabilities may be overconfident on harder cases; confidence can remain high even when accuracy falls.
- **H4:** Explicit search with adequate lookahead increasingly outperforms a one-pass policy on tasks requiring deeper planning.

> **Important distinction:** *Distance to goal* is not the same as *required lookahead*. A straight corridor 8 steps long can be simpler than a 2-step choice with a deceptive branch. We will test these dimensions separately.

## 2. Scope and setup

### Phase 1 — Deterministic FrozenLake (primary)

- Environment: `gymnasium.make("FrozenLake-v1", desc=map_rows, is_slippery=False)`.
- Start from randomly generated, validated maps with safe floor `F`, holes `H`, start/player position `S`/`P`, and goal `G`.
- Candidate map sizes: **8 × 8** and **12 × 12**. Adjust generator as needed so each distance bin has enough valid states.
- Actions: **LEFT, DOWN, RIGHT, UP** (Gymnasium conventions: `0, 1, 2, 3`).
- The model receives the **full map**, rules, current player position, and four action choices in text. It does **not** receive optimal values, shortest-path solutions, or annotated trajectories.
- The model chooses exactly one action. The first evaluation is **single-state/offline** (fixed benchmark states); complete episodes are a secondary evaluation.

### Phase 2 — Stochastic FrozenLake (extension)

- Turn on `is_slippery=True` to introduce transition stochasticity.
- Use the actual transition kernel and a specified reward/discount convention to compute an approximate-exact optimal value via dynamic programming.
- Distinguish stochastic risk and uncertainty from pure planning difficulty. Do not mix Phase 1 and Phase 2 data in a single distance curve.

## 3. Model: train-free direct choice-token

### 3.1 Base model

- Use **Qwen3.5-27B** as a frozen base model (this is the model designation used in the prior design discussion; verify the intended checkpoint if “Qwen3.8-27B” refers to a different model).
- No finetuning and no additional trained classification head.
- Disable thinking/reasoning generation when the checkpoint supports it; verify that the scored token position is the first action-choice token, not a thinking marker or hidden formatting token.
- Perform a single model forward pass per state to obtain logits at the answer position.

### 3.2 Prompt interface

One canonical prompt template for the first experiments:

```text
You are playing FrozenLake.
Rules:
- P is your current position; G is the goal; H is a hole (failure); F is safe floor.
- Choose one action that maximizes the probability of reaching G while avoiding holes.
- You can move LEFT, DOWN, RIGHT, or UP. Moving beyond a board edge leaves you in place.
- Respond with exactly one letter from A, B, C, D.

Map:
<ASCII MAP WITH CURRENT POSITION MARKED P>

Actions:
A: LEFT
B: DOWN
C: RIGHT
D: UP

Answer:
```

**Implementation details**

1. Apply the checkpoint's chat template consistently; confirm the actual generation prefix and final position to be scored.
2. Verify that each of `A`, `B`, `C`, `D` (with the exact prompt prefix) is represented as the intended single next token. If not, use a validated single-token alternative or score full option strings rather than silently selecting the wrong vocabulary IDs.
3. Extract logits `z_A, z_B, z_C, z_D` from the last relevant position.
4. Normalize **only over the four candidate tokens**:

   \[
   p_\theta(a\mid s)=\frac{\exp(z_a/T)}{\sum_{a'\in\mathcal A}\exp(z_{a'}/T)}, \quad T=1\text{ in primary experiments.}
   \]

5. Predict `argmax_a p_theta(a | s)`. Save all four raw logits, restricted probabilities, choice mapping, prompt, and chosen action.
6. Randomize letter-to-direction mappings in a controlled robustness experiment, and invert the mapping at evaluation. This tests choice-token and option-position bias.

**Interpretation warning:** The four-way restricted softmax is a *relative scoring rule over the offered actions*. It is **not automatically** the probability that an action is optimal. Keep `T=1` for the strict train-free baseline; any post-hoc calibration using held-out labels is a separately labeled analysis, not the unmodified model.

## 4. Simulator and reference implementation

- Use Gymnasium FrozenLake for state transitions and episode rollouts.
- Refer to [jev-arcade](https://github.com/CankatSarac/jev-arcade) for the organization of game adapter, player, evaluation harness, replay, and analysis components, but implement only the minimal functionality needed for reproducible FrozenLake benchmarking.
- Separate **environment logic**, **state serialization**, **policy interface**, **oracle solution**, and **evaluation**.
- Set random seeds for map generation, benchmark sampling, policy evaluation, and option-order permutations.
- Cache model outputs for offline metrics to avoid rerunning an expensive 27B forward pass.

Proposed project layout:

```text
system1-frozenlake/
├── plan.md
├── configs/
│   └── frozenlake.yaml
├── src/
│   ├── environments/
│   │   ├── frozenlake.py       # Gymnasium wrapper and state serialization
│   │   └── map_generator.py   # Valid maps and controlled instances
│   ├── agents/
│   │   ├── qwen_choice.py      # Direct choice-token scoring
│   │   ├── heuristics.py       # Random and greedy baselines
│   │   ├── depth_search.py     # Limited lookahead baseline
│   │   └── qlearning.py        # Optional learned policy baseline
│   ├── oracles/
│   │   └── optimal_q.py        # BFS / value iteration and tie handling
│   ├── evaluation/
│   │   ├── benchmark.py        # Fixed state and map splits
│   │   ├── metrics.py          # Accuracy, regret, calibration
│   │   └── plots.py            # Curves and tables
│   └── run.py
├── tests/
│   ├── test_transitions.py
│   ├── test_oracle.py
│   └── test_choice_tokens.py
└── results/
    ├── datasets/
    ├── predictions/
    └── figures/
```

## 5. Dataset construction and difficulty definitions

### 5.1 Distance-to-goal experiment (Experiment A)

For each deterministic map, construct its directed transition graph over nonterminal safe states. Compute the shortest **safe-path** distance to the goal:

\[
d(s)=\min_{\pi}\{\text{number of actions in a safe path from }s\text{ to }G\}.
\]

- Sample benchmark states with `d(s) ∈ {1, 2, 4, 8}`.
- Exclude holes, terminal goal states, and unreachable states.
- Sample **multiple distinct maps** per bin, not only many nearby states from the same map.
- Balance, where possible, map size, obstacle density, action tie count, and goal geometry across bins.
- For controlled accuracy analysis, tag whether the best action is **unique**. Report full-sample and unique-optimal-action results.
- Split by **map**, not by state, to avoid nearly identical boards appearing in both development and test sets.
- Fix the evaluation set before inspecting performance; use a separate development set for prompt debugging.

**Do not use Hamming distance** to denote steps to the goal: Hamming distance counts mismatched coordinates/cells rather than path length. Manhattan distance may be recorded as an additional geometric heuristic, but it ignores holes and detours.

### 5.2 Planning complexity experiment (Experiment B)

Distance alone does not isolate forward-looking planning. Create controlled examples at matched `d(s)` with varying complexity:

- **Straight/open path:** a local greedy move also lies on an optimal path.
- **Detour:** the optimal first action temporarily fails to reduce Manhattan distance to the goal.
- **Misleading branch:** a locally attractive action leads to a dead end, loop, or hole; choosing optimally requires looking farther ahead.
- **Matched-prefix cases (stronger test):** local neighborhoods up to radius `k` are identical, but distant map structure changes the optimal first action.

Record difficulty proxies such as:

- **Detour gap:** `shortest safe-path distance − Manhattan distance`.
- **Greedy mismatch:** whether the Manhattan-greedy action is suboptimal.
- **Minimum distinguishing lookahead depth:** for a *specified depth-limited search algorithm and leaf heuristic*, the smallest search depth at which it selects an optimal action.
- **Action-value gap:** difference between the best and second-best `Q*` (report separately because small gaps can make action selection ambiguous but not necessarily require deeper reasoning).

> Search depth is algorithm-dependent. It is a controlled proxy for planning complexity, not a universal, intrinsic property of the state.

## 6. Ground truth: optimal action values

**Prefer exact planning over RL approximation whenever transitions are known.** FrozenLake provides a small, fully specified MDP, so a solver is cleaner than using a trained RL agent as ground truth.

### Deterministic environment

- Use BFS / shortest-path dynamic programming to identify states with a safe path to the goal and optimal actions for the chosen objective.
- Define the reward and discount clearly. A recommended primary setting is reward `1` on reaching the goal, `0` otherwise, and `gamma = 0.99`. Holes and the goal are terminal.
- This objective favors higher-probability success and, for deterministic paths with success possible, shorter arrival time. States with no path to the goal have zero value and should be excluded from the main shortest-path benchmark.

### Stochastic environment

Use value iteration until Bellman residual is below a recorded tolerance:

\[
Q^*(s,a)=\sum_{s'}P(s'\mid s,a)\left[r(s,a,s')+\gamma V^*(s')\right],
\qquad V^*(s)=\max_a Q^*(s,a).
\]

Save for every state:

- `Q_star[4]`, `V_star`, optimal action set `A_star` (ties determined by a declared numerical tolerance), `d(s)`, and complexity tags.

For a selected model action `a_hat`, define:

\[
\operatorname{regret}(s)=V^*(s)-Q^*(s,\hat a).
\]

**Do not equate an optimal-value softmax with an objective “true” probability of action optimality.** A softmax over `Q*` depends on the chosen temperature. Use the optimal-action set to assess correctness and `Q*` for regret; compare distribution shapes only as a secondary, temperature-explicit analysis.

## 7. Baselines

| Method | Role | Key controls |
|---|---|---|
| Random | Lower-bound reference | Uniform among legal action choices |
| Manhattan greedy | No-search heuristic | Choose action minimizing coordinate distance; fixed tie-breaking |
| Depth-limited search | Explicit forward lookahead | Depth 1, 2, 4, 8; same transition rules; declare leaf heuristic and terminal handling |
| Tabular Q-learning | Learned RL baseline | Fixed training episodes, seed repetitions, evaluation on held-out maps as appropriate |
| Optimal policy / value iteration | Oracle upper bound | Full known transition model; specified discount and tie tolerance |
| Qwen direct choice-token | **Main System-1 model** | Frozen weights, one pass, no rollout, `T=1` |

**Comparability caveats:** Q-learning trained on one fixed tabular map does not automatically generalize to unseen maps; keep within-map RL comparisons distinct from cross-map zero-shot comparisons. Explicit search has access to environment transitions, whereas the LLM may only see text rules and a map. This distinction should be disclosed rather than framed as a perfectly matched information/computation comparison. Measure latency and calls separately when discussing compute budgets.

## 8. Evaluation

### 8.1 Primary offline metrics (fixed state set)

| Metric | Definition | Purpose |
|---|---|---|
| Optimal-action accuracy | `1[a_hat ∈ A_star(s)]` | Whether the selected move is optimal |
| Q-regret | `V_star(s) − Q_star(s, a_hat)` | Cost of an incorrect move |
| Optimal-action probability mass | `sum_{a ∈ A_star} p_theta(a|s)` | How much probability is placed on optimal moves |
| Decision confidence | `max_a p_theta(a|s)` | Raw self-assigned confidence proxy |
| Calibration | Reliability plot, Brier score (binary correct/incorrect), ECE | Whether confidence predicts optimal-action correctness |
| Runtime | Model forward-pass latency, tokens generated (target: 0) | Efficiency of direct scoring |

Calibration target: define `Y(s) = 1[a_hat ∈ A_star(s)]` and compare `max_a p_theta(a|s)` with empirical `P(Y=1)` across confidence bins. This is a **diagnostic** of the restricted-choice score, not an assumption that it is calibrated.

### 8.2 Episode metrics (secondary)

- Success rate reaching the goal before a step limit.
- Failure rate falling into holes.
- Steps to goal among successful runs.
- Episode return under the declared reward and discount convention.

### 8.3 Analyses and plots

1. **Accuracy vs. shortest-path distance**: separate bins `1 / 2 / 4 / 8`.
2. **Q-regret vs. distance**: reveal cost as well as mistakes.
3. **Confidence and empirical accuracy vs. distance**: detect overconfidence at long horizons.
4. **Reliability diagrams / ECE by difficulty stratum**: assess calibration rather than only entropy.
5. **Matched-distance complexity comparisons**: straight vs detour vs misleading branch.
6. **Depth-limited search vs. direct choice-token**: compare performance as search depth grows.
7. **Option-order robustness**: compare predictions under randomized letter-to-action mappings.

Report uncertainty intervals using **map-level clustered bootstrap** or a suitable hierarchical analysis, because multiple sampled states from one map are not independent. Show counts of maps and states in each group.

## 9. Execution milestones

### M1 — Environment and oracle

- [ ] Set up Gymnasium FrozenLake wrapper with custom maps and ASCII state rendering.
- [ ] Implement map generator, path-reachability checks, and per-state shortest-path distances.
- [ ] Implement deterministic optimal actions / `Q*`; cross-check against value iteration on small maps.
- [ ] Unit tests for holes, walls/boundaries, terminal behavior, tied optimal actions, and unreachable states.

**Deliverable:** reproducible labeled benchmark candidates with exact optimal values.

### M2 — Qwen train-free scoring

- [ ] Load the specified Qwen checkpoint and tokenizer once.
- [ ] Confirm chat template, thinking setting, scoring position, and candidate token IDs.
- [ ] Implement one-forward-pass A/B/C/D restricted-logit scoring.
- [ ] Validate with hand-designed trivial maps and letter-action permutations.
- [ ] Cache per-state logits, probability vectors, and latency.

**Deliverable:** stable `predict_action(state_text) -> {action, logits, probabilities}` interface.

### M3 — Main distance experiment

- [ ] Generate a map-held-out fixed state test set with distance groups `1 / 2 / 4 / 8`.
- [ ] Run Qwen, random, Manhattan-greedy, depth-limited search, and oracle policies.
- [ ] Compute optimal-action accuracy, Q-regret, optimal-action probability mass, and calibration.
- [ ] Produce distance curves with uncertainty intervals and relevant sample counts.

**Deliverable:** first evidence for/against H1 and H3.

### M4 — Planning-complexity controls

- [ ] Generate matched-distance straight, detour, and misleading-branch instances.
- [ ] Define and compute complexity proxies, including algorithm-specific minimum lookahead depth.
- [ ] Compare methods within distance strata and unique-optimal-action subsets.
- [ ] Run candidate-label permutation robustness checks.

**Deliverable:** tests isolating H2 from mere geometric distance.

### M5 — Extensions

- [ ] Add a tabular Q-learning baseline with transparent training/evaluation protocol.
- [ ] Run full-episode rollouts on a fixed map split.
- [ ] Add stochastic FrozenLake and value-iteration oracle.
- [ ] Optional: compare the direct model to a prompted, explicitly reasoning model under separately reported time/token budgets.

**Deliverable:** broader picture of System-1 vs. search/RL trade-offs.

## 10. Risks and controls

| Risk | Control |
|---|---|
| Distance confused with difficulty | Matched-distance complexity experiment; separate `d(s)` from depth-required proxies |
| Multiple optimal actions distort confidence/accuracy | Evaluate membership in `A_star`; report unique-optimum subsets and probability mass on all optimal actions |
| Prompt or option-token artifacts | Validate single-token mapping; shuffle action labels; freeze prompt after development |
| Train-free claim muddied by calibration | Keep primary evaluation frozen and `T=1`; label any fitted calibration separately |
| Known maps / memorized layouts | Procedurally generate new maps; split by whole map; vary obstacle layouts |
| Oracle definition ambiguous | Specify transition kernel, terminal treatment, reward, gamma, tolerance |
| Large-model inference expensive | Batch states where feasible, cache output, keep model loaded; time measured separately |
| Strong claims from correlated states | Cluster uncertainty estimates at the map level |
| RL policy and oracle conflated | Treat VI/BFS as oracle, tabular Q-learning as a learned baseline |

## 11. Criteria for a convincing finding

A convincing result is **not merely** that action accuracy is lower at distance 8 than at distance 1. It should show some combination of:

1. **Performance degradation with distance**, documented on held-out procedurally generated maps.
2. **Independent effect of complexity**, within matched-distance instances that require nonlocal decisions.
3. **Confidence-quality mismatch**, e.g., accuracy declines sharply while restricted-choice confidence remains high.
4. **Regret significance**, showing whether errors are consequential rather than harmless near-ties.
5. **Search comparison**, showing where explicit extra computation overtakes direct prediction, without pretending the compute and information inputs are identical.

## 12. Immediate next actions

1. Freeze the exact Qwen checkpoint identifier and prompting/scoring interface.
2. Implement deterministic custom-map FrozenLake plus BFS/VI oracle, and generate a small manually inspected benchmark.
3. Verify the four action tokens and one-pass logits on trivial states.
4. Run the initial distance curves (`1 / 2 / 4 / 8`) before expanding to matched-distance planning-complexity tasks.

---

**References**

- [Gymnasium FrozenLake documentation](https://gymnasium.farama.org/environments/toy_text/frozen_lake/)
- [jev-arcade repository](https://github.com/CankatSarac/jev-arcade)
