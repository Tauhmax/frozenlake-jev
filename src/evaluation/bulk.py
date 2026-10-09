"""Score a frozen JSONL dataset through the local service with resumable output."""

import argparse
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

from src.agents.prompt import PROMPT_VERSION, format_prompt
from src.evaluation.metrics import score
from src.evaluation.report import build_report


def request(url, data=None):
    payload = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {detail}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="results/datasets/frozenlake-1000.jsonl")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="results/predictions/qwen35-1000.jsonl")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--limit", type=int, default=0, help="Pilot size; 0 scores all states"
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 0:
        parser.error("Batch size must be positive and limit nonnegative")
    dataset_path = Path(args.dataset)
    dataset = [json.loads(line) for line in dataset_path.read_text().splitlines()]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output.with_suffix(".meta.json")
    health = request(args.url + "/health")
    if not health.get("ready"):
        raise RuntimeError("Model server is not ready")
    metadata = {
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "model": health["model"],
        "batch_size": args.batch_size,
        "prompt_version": PROMPT_VERSION,
        "metric_code_sha256": hashlib.sha256(
            Path("src/evaluation/metrics.py").read_bytes()
        ).hexdigest(),
        "prompt_sha256": hashlib.sha256(
            Path("src/agents/prompt.py").read_bytes()
        ).hexdigest(),
        "latency": "forward_seconds is amortized batch latency per state",
    }
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != metadata:
            raise RuntimeError("Cache configuration differs; choose a new output file")
    elif output.exists():
        raise RuntimeError("Existing predictions have no manifest; choose a new output")
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2))
    cached = []
    if output.exists():
        # Discard only a torn final write; all complete rows remain cached.
        with output.open("rb+") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    stream.truncate(offset)
                    break
                cached.append(json.loads(line))
    completed = {row["id"] for row in cached}
    if not completed.issubset({row["id"] for row in dataset}):
        raise RuntimeError("Prediction cache contains IDs outside this dataset")
    if len(completed) != len(cached):
        raise RuntimeError("Duplicate IDs in prediction cache")
    selected = dataset[: args.limit] if args.limit else dataset
    pending = [row for row in selected if row["id"] not in completed]
    print(f"Cached {len(cached)}, pending {len(pending)}", flush=True)
    started = time.perf_counter()
    with output.open("a", encoding="utf-8", newline="\n") as stream:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            prompts = []
            for row in batch:
                text = format_prompt(row["board"], row["gamma"])
                if "state_text" in row and (
                    row.get("prompt_version") != PROMPT_VERSION
                    or row["state_text"] != text
                ):
                    raise RuntimeError(
                        "Dataset state_text is stale; regenerate the dataset"
                    )
                prompts.append(text)
            predictions = request(args.url + "/score", {"prompts": prompts})[
                "predictions"
            ]
            if len(predictions) != len(batch):
                raise RuntimeError("Server returned an incorrect batch size")
            for row, prediction in zip(batch, predictions):
                probs = prediction["probabilities"]
                if (
                    len(probs) != 4
                    or any(not math.isfinite(p) or p < 0 for p in probs)
                    or abs(sum(probs) - 1) > 1e-5
                    or prediction["action"] != max(range(4), key=probs.__getitem__)
                ):
                    raise RuntimeError("Invalid model probability vector or action")
                result = {
                    **row,
                    "prompt_version": PROMPT_VERSION,
                    "state_text": format_prompt(row["board"], row["gamma"]),
                    "prediction": prediction,
                    "metrics": score(
                        prediction["action"],
                        prediction["probabilities"],
                        row["q_star"],
                        row["optimal_actions"],
                    ),
                }
                stream.write(json.dumps(result) + "\n")
                cached.append(result)
            stream.flush()
            done = min(start + len(batch), len(pending))
            if start == 0 or done % 128 < args.batch_size or done == len(pending):
                elapsed = time.perf_counter() - started
                print(
                    f"Scored {done}/{len(pending)}; {done / elapsed:.2f} states/s",
                    flush=True,
                )
    report = build_report(cached)
    report["dataset_states"] = len(dataset)
    report["complete"] = len(cached) == len(dataset)
    report["metadata"] = metadata
    output.with_suffix(".summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["test"], indent=2), flush=True)


if __name__ == "__main__":
    main()
