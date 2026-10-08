"""Download a pinned Qwen checkpoint; model loading itself stays offline."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

CHECKPOINTS = {
    "qwen3-4b": (
        "Qwen/Qwen3-4B-Instruct-2507",
        "cdbee75f17c01a7cc42f958dc650907174af0554",
    ),
    "qwen3.5-0.8b": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", choices=CHECKPOINTS, default="qwen3-4b")
args = parser.parse_args()
repo, revision = CHECKPOINTS[args.model]
folder = Path("models") / repo.split("/")[-1]
manifest = folder / "checkpoint.json"
folder.mkdir(parents=True, exist_ok=True)
manifest.write_text(json.dumps({"repo_id": repo, "revision": revision}, indent=2))
print(f"Downloading {repo}@{revision} to {folder}", flush=True)
snapshot_download(
    repo,
    revision=revision,
    local_dir=folder,
    allow_patterns=[
        "*.json",
        "*.safetensors",
        "*.txt",
        "*.jinja",
        "*.model",
        "LICENSE*",
    ],
    max_workers=2,
)
print("Download complete", flush=True)
