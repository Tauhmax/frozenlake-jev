"""Play one complete, resumable episode from S on each distinct dataset map."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from src.agents.prompt import EPISODE_PROMPT_VERSION, format_prompt
from src.environments.frozenlake import make_env, render_state
from src.environments.map_generator import distances_to_goal
from src.evaluation.bulk import request
from src.evaluation.metrics import score
from src.oracles.optimal_q import finite_horizon_q, optimal_actions


def outcome(reward, terminated, step, max_steps):
    if terminated:
        return "success" if reward == 1 else "hole"
    return "timeout" if step == max_steps else None


def summarize(episodes):
    counts = Counter(e["outcome"] for e in episodes)
    feasible = [e for e in episodes if e["start_distance"] <= e["max_steps"]]
    return {
        "episodes": len(episodes),
        "outcomes": dict(counts),
        "success_rate": counts["success"] / len(episodes),
        "mean_steps": float(np.mean([e["steps"] for e in episodes])),
        "decisions": sum(e["steps"] for e in episodes),
        "oracle_feasible_episodes": len(feasible),
        "success_rate_on_feasible": (
            sum(e["outcome"] == "success" for e in feasible) / len(feasible)
            if feasible
            else None
        ),
        "mean_discounted_return": float(np.mean([e["return"] for e in episodes])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="results/datasets/frozenlake-1000-state-v2.jsonl"
    )
    parser.add_argument("--output", default="results/episodes/qwen35-max30.jsonl")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--request-timeout", type=float, help="Seconds; default: no timeout"
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.max_steps < 1 or args.batch_size < 1 or args.limit < 0:
        parser.error("Invalid horizon, batch size, or limit")
    if args.request_timeout is not None and (
        not math.isfinite(args.request_timeout) or args.request_timeout <= 0
    ):
        parser.error("request-timeout must be finite and positive")
    dataset = Path(args.dataset)
    maps = {}
    for line in dataset.read_text().splitlines():
        row = json.loads(line)
        old = maps.setdefault(row["map_id"], row)
        if any(old[k] != row[k] for k in ("map", "split", "gamma")):
            raise ValueError("Inconsistent map records")
    selected = list(maps.values())[: args.limit or None]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    health = request(args.url + "/health", timeout=args.request_timeout)
    if (
        not health.get("ready")
        or health["model"].get("episode_prompt_version") != EPISODE_PROMPT_VERSION
    ):
        raise RuntimeError("Restart server with episode prompt support")
    metadata = {
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "model": health["model"],
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "prompt_version": EPISODE_PROMPT_VERSION,
        "code_sha256": hashlib.sha256(
            b"".join(
                Path(p).read_bytes()
                for p in (
                    __file__,
                    "src/agents/prompt.py",
                    "src/oracles/optimal_q.py",
                    "src/evaluation/metrics.py",
                    "src/environments/frozenlake.py",
                )
            )
        ).hexdigest(),
        "protocol": (
            "One episode per map from S; greedy choice-token action; "
            "terminal or horizon"
        ),
    }
    manifest = output.with_suffix(".meta.json")
    if manifest.exists():
        if json.loads(manifest.read_text()) != metadata:
            raise RuntimeError("Cache configuration changed; use a new output")
    elif output.exists():
        raise RuntimeError("Trace exists without manifest")
    else:
        manifest.write_text(json.dumps(metadata, indent=2))
    cached = {r["map_id"]: [] for r in selected}
    if output.exists():
        with output.open("rb+") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    stream.truncate(offset)
                    break
                event = json.loads(line)
                cached[event["map_id"]].append(event)
    completed, active = [], []
    pending = iter(selected)

    def finish(item, event):
        row = item["row"]
        completed.append(
            {
                "map_id": row["map_id"],
                "split": row["split"],
                "size": row["size"],
                "map": row["map"],
                "start_distance": item["distance"],
                "max_steps": args.max_steps,
                "steps": event["step"],
                "outcome": event["outcome"],
                "final_state": event["next_state"],
                "actions": item["actions"],
                "return": row["gamma"] ** (event["step"] - 1)
                if event["outcome"] == "success"
                else 0,
            }
        )
        item["env"].close()

    def refill():
        while len(active) < args.batch_size:
            row = next(pending, None)
            if row is None:
                break
            env = make_env(row["map"])
            state, _ = env.reset(seed=42)
            item = {
                "row": row,
                "env": env,
                "state": int(state),
                "actions": [],
                "distance": distances_to_goal(row["map"])[int(state)],
            }
            last = None
            for index, event in enumerate(cached[row["map_id"]], 1):
                if last and last["outcome"]:
                    raise RuntimeError("Cached steps after episode end")
                if event["step"] != index or event["state"] != item["state"]:
                    raise RuntimeError("Cached trajectory is discontinuous")
                nxt, reward, terminated, _, _ = env.step(event["action"])
                expected = outcome(reward, terminated, index, args.max_steps)
                if (int(nxt), float(reward), expected) != (
                    event["next_state"],
                    event["reward"],
                    event["outcome"],
                ):
                    raise RuntimeError("Cached transition disagrees with environment")
                item["state"] = int(nxt)
                item["actions"].append(event["action"])
                last = event
            if last and last["outcome"]:
                finish(item, last)
            else:
                item["q"] = finite_horizon_q(env, args.max_steps, row["gamma"])
                active.append(item)

    refill()
    print(f"Resumed {len(completed)} complete episodes", flush=True)
    batches = 0
    with output.open("a", encoding="utf-8", newline="\n") as stream:
        while active:
            prompts = [
                format_prompt(
                    render_state(i["env"], i["state"]),
                    i["row"]["gamma"],
                    args.max_steps - len(i["actions"]),
                )
                for i in active
            ]
            predictions = request(
                args.url + "/score", {"prompts": prompts}, timeout=args.request_timeout
            )["predictions"]
            if len(predictions) != len(active):
                raise RuntimeError("Incorrect prediction count")
            survivors = []
            for item, prompt, prediction in zip(active, prompts, predictions):
                probs = np.asarray(prediction["probabilities"])
                action = prediction["action"]
                if (
                    probs.shape != (4,)
                    or not np.isfinite(probs).all()
                    or (probs < 0).any()
                    or abs(probs.sum() - 1) > 1e-5
                    or action != int(probs.argmax())
                ):
                    raise RuntimeError("Invalid prediction")
                remaining = args.max_steps - len(item["actions"])
                q = item["q"][remaining, item["state"]]
                nxt, reward, terminated, _, _ = item["env"].step(action)
                step = len(item["actions"]) + 1
                result = outcome(reward, terminated, step, args.max_steps)
                event = {
                    "map_id": item["row"]["map_id"],
                    "step": step,
                    "state": item["state"],
                    "next_state": int(nxt),
                    "action": action,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": result == "timeout",
                    "outcome": result,
                    "remaining_steps": remaining,
                    "state_text": prompt,
                    "q_star": q.tolist(),
                    "solvable_within_budget": bool(q.max() > 0),
                    "optimal_actions": optimal_actions(q),
                    "prediction": prediction,
                    "metrics": score(
                        action, prediction["probabilities"], q, optimal_actions(q)
                    ),
                }
                stream.write(json.dumps(event) + "\n")
                item["actions"].append(action)
                item["state"] = int(nxt)
                if result:
                    finish(item, event)
                else:
                    survivors.append(item)
            stream.flush()
            active = survivors
            refill()
            batches += 1
            if batches % 10 == 0 or not active:
                print(
                    f"Episodes {len(completed)}/{len(selected)}; "
                    f"outcomes {dict(Counter(e['outcome'] for e in completed))}",
                    flush=True,
                )
    output.with_suffix(".episodes.json").write_text(json.dumps(completed, indent=2))
    report = {
        "complete": len(completed) == len(selected),
        "metadata": metadata,
        "overall": summarize(completed),
    }
    for split in ("dev", "test"):
        group = [e for e in completed if e["split"] == split]
        if group:
            report[split] = summarize(group)
    report["by_size"] = {
        str(size): summarize([e for e in completed if e["size"] == size])
        for size in sorted({e["size"] for e in completed})
    }
    output.with_suffix(".summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["overall"], indent=2), flush=True)


if __name__ == "__main__":
    main()
