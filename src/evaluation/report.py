"""Offline metrics and confidence intervals clustered by whole map."""

from collections import Counter, defaultdict

import numpy as np

from src.agents.actions import DIRECTIONS


def summarize(rows):
    if not rows:
        return {"states": 0, "maps": 0}
    metrics = [r["metrics"] for r in rows]
    means = {key: float(np.mean([m[key] for m in metrics])) for key in metrics[0]}
    ece = 0.0
    for i in range(10):
        selected = [m for m in metrics if min(int(m["confidence"] * 10), 9) == i]
        if selected:
            accuracy = np.mean([m["optimal_action_accuracy"] for m in selected])
            confidence = np.mean([m["confidence"] for m in selected])
            ece += len(selected) / len(rows) * abs(accuracy - confidence)
    by_map = defaultdict(list)
    for row in rows:
        by_map[row["map_id"]].append(row)
    counts = np.array([len(group) for group in by_map.values()])
    sample = np.random.default_rng(42).integers(0, len(counts), (1000, len(counts)))
    intervals = {}
    for key in ("optimal_action_accuracy", "q_regret"):
        totals = np.array(
            [sum(r["metrics"][key] for r in group) for group in by_map.values()]
        )
        bootstrap = totals[sample].sum(axis=1) / counts[sample].sum(axis=1)
        intervals[key] = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    actions = Counter(r["prediction"]["action"] for r in rows)
    unique = [r for r in rows if r["unique_optimal"]]
    return {
        "action_counts": {name: actions[i] for i, name in enumerate(DIRECTIONS)},
        "unique_optimal_states": len(unique),
        "unique_optimal_accuracy": (
            float(np.mean([r["metrics"]["optimal_action_accuracy"] for r in unique]))
            if unique
            else None
        ),
        "states": len(rows),
        "maps": len(by_map),
        **means,
        "ece_10_bins": float(ece),
        "map_bootstrap_95_ci": intervals,
        "random_expected_accuracy": float(
            np.mean([len(r["optimal_actions"]) / 4 for r in rows])
        ),
        "random_expected_regret": float(
            np.mean([max(r["q_star"]) - np.mean(r["q_star"]) for r in rows])
        ),
        "amortized_forward_seconds": float(
            np.mean([r["prediction"]["forward_seconds"] for r in rows])
        ),
    }


def build_report(rows):
    report = {}
    for split in ("dev", "test"):
        subset = [row for row in rows if row["split"] == split]
        report[split] = {
            "overall": summarize(subset),
            "by_distance": {
                str(d): summarize([r for r in subset if r["distance"] == d])
                for d in (1, 2, 4, 8)
            },
            "by_size": {
                str(s): summarize([r for r in subset if r["size"] == s])
                for s in (8, 12)
            },
            "unique_optimal": summarize([r for r in subset if r["unique_optimal"]]),
        }
    return report
