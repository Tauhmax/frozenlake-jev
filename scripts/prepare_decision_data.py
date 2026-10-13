"""Download and sample five four-choice datasets using only the standard library."""

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def fetch_page(source, offset, cache):
    path = cache / f"{offset:06d}.json"
    if not path.exists():
        query = urllib.parse.urlencode(
            {
                "dataset": source["repo"],
                "config": source["config"],
                "split": source["split"],
                "offset": offset,
                "length": 100,
            },
            safe="/",
        )
        url = "https://datasets-server.huggingface.co/rows?" + query
        for attempt in range(6):
            try:
                time.sleep(1)
                with urllib.request.urlopen(url, timeout=30) as response:
                    revision = response.headers.get("x-revision")
                    payload = json.load(response)
                break
            except (OSError, ValueError):
                if attempt == 5:
                    raise
                time.sleep(min(30, 5 * 2**attempt))
        if revision != source["revision"]:
            raise ValueError(
                f"Source revision changed for {source['name']}: {revision}"
            )
        path.write_text(
            json.dumps({"revision": revision, "url": url, "payload": payload}),
            encoding="utf-8",
        )
    cached = json.loads(path.read_text(encoding="utf-8"))
    if cached["revision"] != source["revision"]:
        raise ValueError(f"Cached revision mismatch: {path}")
    payload = cached["payload"]
    if payload.get("partial") or any(x["truncated_cells"] for x in payload["rows"]):
        raise ValueError(f"Incomplete source page: {path}")
    return payload


def normalize(source, entry):
    row = entry["row"]
    name = source["name"]
    context = ""
    category = name
    original_id = str(row.get("id", row.get("ind", entry["row_idx"])))
    if name in ("arc_challenge", "openbookqa"):
        question = row.get("question", row.get("question_stem"))
        choices = row["choices"]["text"]
        answer = row["choices"]["label"].index(row["answerKey"])
        group = question
    elif name == "hellaswag":
        context = row["ctx"]
        question = "Which continuation best completes the passage?"
        choices = row["endings"]
        answer = int(row["label"])
        group = row["source_id"]
        category = row["activity_label"]
    elif name == "race":
        context = row["article"]
        question = row["question"]
        choices = row["options"]
        answer = "ABCD".index(row["answer"])
        original_id = row["example_id"]
        group = " ".join(context.split())
        category = "high" if original_id.startswith("high") else "middle"
    elif name == "mmlu":
        question = row["question"]
        choices = row["choices"]
        answer = int(row["answer"])
        group = question
        category = row["subject"]
    else:
        raise ValueError(name)
    if (
        len(choices) != 4
        or not 0 <= answer < 4
        or not question.strip()
        or any(not x.strip() for x in choices)
        or len(set(choices)) != 4
    ):
        return None
    content = json.dumps([context, question, choices], ensure_ascii=False)
    return {
        "id": f"{name}/{entry['row_idx']}",
        "dataset": name,
        "category": category,
        "context": context,
        "question": question,
        "choices": choices,
        "answer": answer,
        "group_id": digest(name + ":" + group),
        "content_sha256": digest(content),
        "source": {
            **{k: source[k] for k in ("repo", "revision", "config", "split")},
            "row_index": entry["row_idx"],
            "original_id": original_id,
        },
    }


def sample_rows(rows, seed):
    # One question per article/video/question group, then stratify by category.
    groups = {}
    for row in sorted(rows, key=lambda x: digest(f"{seed}:{x['id']}")):
        groups.setdefault(row["group_id"], row)
    categories = defaultdict(list)
    for row in groups.values():
        categories[row["category"]].append(row)
    names = sorted(categories, key=lambda x: digest(f"{seed}:{x}"))
    sampled = []
    while names:
        for name in names:
            sampled.append(categories[name].pop())
        names = [name for name in names if categories[name]]
    return sampled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--sources", type=Path, default=Path("data/sources.json"))
    parser.add_argument("--eval-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.eval_size < 1:
        parser.error("Evaluation size must be positive")
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    manifest = {
        "version": 1,
        "seed": args.seed,
        "eval_size": args.eval_size,
        "datasets": {},
    }
    seen = set()
    for source in sources:
        name = source["name"]
        cache = args.output_dir / ".cache" / name / source["revision"]
        cache.mkdir(parents=True, exist_ok=True)
        first = fetch_page(source, 0, cache)
        count = first["num_rows_total"]
        offsets = list(range(0, count, 100))
        print(f"{name}: fetching {count} source rows", flush=True)
        pages = [fetch_page(source, n, cache) for n in offsets]
        entries = [entry for page in pages for entry in page["rows"]]
        if [x["row_idx"] for x in entries] != list(range(count)):
            raise ValueError(f"Missing or reordered source rows: {name}")
        rows = [normalize(source, entry) for entry in entries]
        valid = [x for x in rows if x is not None]
        unique = []
        for row in valid:
            if row["content_sha256"] not in seen:
                seen.add(row["content_sha256"])
                unique.append(row)
        sampled = sample_rows(unique, args.seed)
        folder = args.output_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        splits = {
            "eval": sampled[: args.eval_size],
        }
        details = {}
        for split, records in splits.items():
            path = folder / f"{split}.jsonl"
            text = "".join(
                json.dumps({**row, "split": split}, ensure_ascii=False) + "\n"
                for row in records
            )
            path.write_text(text, encoding="utf-8", newline="\n")
            details[split] = {
                "count": len(records),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "answers": dict(sorted(Counter(x["answer"] for x in records).items())),
                "categories": len({x["category"] for x in records}),
            }
        manifest["datasets"][name] = {
            "source": source,
            "source_rows": count,
            "eligible_rows": len(valid),
            "deduplicated_rows": len(unique),
            "independent_groups": len(sampled),
            "source_pages_sha256": digest(
                "".join(
                    hashlib.sha256((cache / f"{n:06d}.json").read_bytes()).hexdigest()
                    for n in offsets
                )
            ),
            "splits": details,
        }
        print(f"{name}: eval={len(splits['eval'])}", flush=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
