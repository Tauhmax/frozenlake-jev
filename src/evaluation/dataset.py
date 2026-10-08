"""Create the complete labeled dataset before any model scoring."""

import argparse
import hashlib
import json
from pathlib import Path

from src.environments.frozenlake import make_env, render_state
from src.environments.map_generator import generate_maps
from src.oracles.optimal_q import optimal_actions, value_iteration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/datasets/frozenlake-1000.jsonl")
    args = parser.parse_args()
    if args.maps < 5:
        parser.error("Use at least 5 maps for the map-held-out split")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    gamma = 0.99
    metadata = {
        "maps": args.maps,
        "seed": args.seed,
        "sizes": [8, 12],
        "distance_bins": [1, 2, 4, 8],
        "safe_probability": 0.75,
        "gamma": gamma,
        "vi_tolerance": 1e-12,
        "tie_tolerance": 1e-9,
        "split": "every fifth generated map is dev; remaining maps are test",
        "sampling": "one uniform state per distance bin per accepted map",
        "filter": "S reachable; all four distance bins present; distinct full maps",
    }
    with output.open("x", encoding="utf-8") as stream:
        for index, (rows, states, distances) in enumerate(
            generate_maps(args.maps, args.seed)
        ):
            map_id = hashlib.sha256("\n".join(rows).encode()).hexdigest()
            env = make_env(rows)
            try:
                q_values, solver = value_iteration(env, gamma)
                goal = "".join(rows).index("G")
                size = len(rows)
                for state in states:
                    distance = distances[state]
                    q = q_values[state]
                    # Independent BFS cross-check catches wrong terminal rewards.
                    if abs(float(max(q)) - gamma ** (distance - 1)) > 1e-9:
                        raise RuntimeError("VI / BFS disagreement")
                    optimal = optimal_actions(q)
                    manhattan = abs(state // size - goal // size) + abs(
                        state % size - goal % size
                    )
                    row = {
                        "id": f"{map_id}:{state}",
                        "map_id": map_id,
                        "map": rows,
                        "size": size,
                        "state": state,
                        "split": "dev" if index % 5 == 0 else "test",
                        "board": render_state(env, state),
                        "distance": distance,
                        "detour_gap": distance - manhattan,
                        "gamma": gamma,
                        "q_star": q.tolist(),
                        "optimal_actions": optimal,
                        "unique_optimal": len(optimal) == 1,
                        "solver": solver,
                    }
                    stream.write(json.dumps(row) + "\n")
            finally:
                env.close()
    metadata["sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved {args.maps} maps / {args.maps * 4} states to {output}")


if __name__ == "__main__":
    main()
