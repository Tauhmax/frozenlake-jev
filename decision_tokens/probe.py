"""Select input-embedding tokens and measure their next-token influence."""

import argparse
import json
from pathlib import Path

from src.agents.jev import ChoiceTokenJev

STRATEGIES = ("abcd", "longest", "shortest", "middle")


def rank_tokens(norms, token_ids, strategy):
    """Select four rows, breaking norm ties by ascending token ID."""
    if len(token_ids) < 4:
        raise ValueError("At least four vocabulary tokens are required")
    ordered = sorted(token_ids, key=lambda i: (norms[i], i))
    if strategy == "shortest":
        return ordered[:4]
    if strategy == "longest":
        return sorted(token_ids, key=lambda i: (-norms[i], i))[:4]
    if strategy == "middle":
        start = (len(ordered) - 4) // 2
        return ordered[start : start + 4]
    raise ValueError(f"Unknown norm strategy: {strategy}")


def select_tokens(engine, strategies, exclude_special=False):
    torch = engine.torch
    tokenizer = engine.tokenizer
    weight = engine.model.get_input_embeddings().weight
    # Float32 norms, in chunks to avoid another full embedding-table allocation.
    with torch.inference_mode():
        norms = torch.cat(
            [block.float().norm(dim=1).cpu() for block in weight.detach().split(4096)]
        ).tolist()
    special = set(tokenizer.all_special_ids)
    valid = sorted(
        i
        for i in set(tokenizer.get_vocab().values())
        if 0 <= i < len(norms) and (not exclude_special or i not in special)
    )
    groups = {}
    for strategy in strategies:
        if strategy == "abcd":
            ids = []
            for label in "ABCD":
                encoded = tokenizer.encode(label, add_special_tokens=False)
                if len(encoded) != 1:
                    raise ValueError(f"{label!r} does not encode as one token")
                ids.append(encoded[0])
            if len(set(ids)) != 4:
                raise ValueError("A/B/C/D must have distinct token IDs")
        else:
            ids = rank_tokens(norms, valid, strategy)
        groups[strategy] = [
            {
                "id": i,
                "token": tokenizer.convert_ids_to_tokens(i),
                "text": tokenizer.decode([i], clean_up_tokenization_spaces=False),
                "l2_norm": norms[i],
                "special": i in special,
            }
            for i in ids
        ]
    return groups


def score(engine, input_ids, candidates):
    torch = engine.torch
    if not input_ids:
        raise ValueError("The decision position requires a nonempty context")
    inputs = torch.tensor([input_ids], device=engine.device)
    with torch.inference_mode():
        logits = (
            engine.model(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                use_cache=False,
                **engine.forward_options,
            )
            .logits[0, -1]
            .float()
        )
        selected = logits[candidates]
        log_probs = selected - torch.logsumexp(logits, dim=0)
        probabilities = torch.softmax(selected, dim=0)
    return {
        "logits": selected.cpu().tolist(),
        "probabilities": probabilities.cpu().tolist(),
        "vocab_log_probabilities": log_probs.cpu().tolist(),
        "candidate_mass": log_probs.exp().sum().item(),
    }


def run_probe(engine, template, tokens):
    """Insert exact IDs, preserving all surrounding IDs across interventions."""
    tokenizer = engine.tokenizer
    prefix = tokenizer.encode(template["prefix"], add_special_tokens=False)
    suffix = tokenizer.encode(template["suffix"], add_special_tokens=False)
    candidates = [token["id"] for token in tokens]
    baseline = score(engine, prefix + suffix, candidates)
    rows = []
    for token_id in candidates:
        result = score(engine, prefix + [token_id] + suffix, candidates)
        result["inserted_id"] = token_id
        result["probability_delta_vs_absent"] = [
            p - b for p, b in zip(result["probabilities"], baseline["probabilities"])
        ]
        rows.append(result)
    # Center each row, then each column: token-specific relative logit effects.
    centered = [
        [v - sum(row["logits"]) / len(candidates) for v in row["logits"]]
        for row in rows
    ]
    means = [sum(column) / len(rows) for column in zip(*centered)]
    influence = [[v - m for v, m in zip(row, means)] for row in centered]
    return {
        "template": template,
        "prefix_ids": prefix,
        "suffix_ids": suffix,
        "distance_to_decision": len(suffix),
        "tokens": tokens,
        "baseline_absent": baseline,
        "interventions": rows,
        "centered_logit_influence": influence,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--strategies", nargs="+", choices=STRATEGIES, default=STRATEGIES
    )
    parser.add_argument("--exclude-special", action="store_true")
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--templates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    templates = []
    if not args.select_only:
        if args.templates is None:
            parser.error("--templates is required unless --select-only is set")
        for line in args.templates.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if not all(isinstance(item.get(k), str) for k in ("prefix", "suffix")):
                    parser.error("Each template needs string prefix and suffix fields")
                templates.append(item)
        if not templates:
            parser.error("No templates found")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    engine = ChoiceTokenJev(args.model_dir, args.device, args.load_in_4bit)
    groups = select_tokens(engine, args.strategies, args.exclude_special)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        metadata = {
            "type": "selection",
            "model_dir": engine.model_dir,
            "checkpoint": engine.checkpoint,
            "model_config": engine.model.config.to_dict(),
            "device": str(engine.device),
            "dtype": str(engine.model.dtype),
            "load_in_4bit": args.load_in_4bit,
            "exclude_special": args.exclude_special,
            "readout": "exact_token_id",
            "prompt_format": "raw_segmented_token_ids",
            "strategies": groups,
        }
        output.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        output.flush()
        print(json.dumps(groups, ensure_ascii=True, indent=2), flush=True)
        total = len(templates) * len(groups)
        done = 0
        for template in templates:
            for strategy, tokens in groups.items():
                result = run_probe(engine, template, tokens)
                result.update(type="probe", strategy=strategy)
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                done += 1
                print(
                    f"[{done}/{total}] {template.get('id', '')} {strategy}", flush=True
                )


if __name__ == "__main__":
    main()
