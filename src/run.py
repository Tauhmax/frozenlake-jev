"""Run the oracle alone, or score every safe state using a local model."""

import argparse
import json
from pathlib import Path

from src.agents.actions import DIRECTIONS, LETTERS
from src.agents.jev import ChoiceTokenJev
from src.agents.prompt import PROMPT_VERSION, format_prompt
from src.environments.frozenlake import make_env, render_state
from src.evaluation.metrics import score
from src.oracles.optimal_q import optimal_actions, value_iteration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/frozenlake.json")
    parser.add_argument("--model-dir", help="Local HF model; omit for oracle only")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--output", default="results/run.json")
    parser.add_argument("--max-input-tokens", type=int)
    args = parser.parse_args()
    if args.max_input_tokens is not None and args.max_input_tokens < 1:
        parser.error("max-input-tokens must be positive")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.output)
    if output.exists():
        parser.error(f"Output exists: {output}. Use a new path to preserve cache.")
    policy = (
        ChoiceTokenJev(
            args.model_dir,
            args.device,
            args.load_in_4bit,
            max_input_tokens=args.max_input_tokens,
        )
        if args.model_dir
        else None
    )
    if policy:
        policy.torch.manual_seed(config["seed"])
    env = make_env(config["map"])
    try:
        env.reset(seed=config["seed"])
        q_values, solver = value_iteration(env, config["gamma"], config["tolerance"])
        records = []
        for state, cell in enumerate(env.desc.flatten()):
            if cell in (b"H", b"G"):
                continue
            q = q_values[state]
            optimal = optimal_actions(q, config["tie_tolerance"])
            row = {
                "state": state,
                "board": render_state(env, state),
                "reachable": bool(max(q) > 0),
                "q_star": q.tolist(),
                "v_star": float(max(q)),
                "optimal_actions": optimal,
            }
            row["prompt_version"] = PROMPT_VERSION
            row["state_text"] = format_prompt(row["board"], config["gamma"])
            if policy and row["reachable"]:
                prediction = policy.predict(row["state_text"])
                row["prediction"] = prediction
                row["metrics"] = score(
                    prediction["action"], prediction["probabilities"], q, optimal
                )
            records.append(row)
        result = {
            "config": config,
            "solver": solver,
            "action_mapping": dict(zip(LETTERS, DIRECTIONS)),
            "mode": "jev" if policy else "oracle",
            "states": records,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
        print(f"Saved {len(records)} safe states to {output}; {solver}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
