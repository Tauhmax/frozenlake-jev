"""One forward pass, no generation, local files only."""

import time
from pathlib import Path

from src.agents.prompt import prepare_choice_tokens


class ChoiceTokenJev:
    def __init__(self, model_dir, device="cpu"):
        path = Path(model_dir).resolve()
        if not (path / "config.json").is_file():
            raise ValueError(f"No local model in {path}; see models/base/README.md")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.model_dir = str(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, torch_dtype="auto"
        ).to(self.device)
        self.model.eval()

    def predict(self, prompt):
        torch = self.torch
        prefix, ids, choices = prepare_choice_tokens(self.tokenizer, prompt)
        inputs = torch.tensor([ids], device=self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = (
                self.model(
                    input_ids=inputs,
                    attention_mask=torch.ones_like(inputs),
                    use_cache=False,
                )
                .logits[0, -1, choices]
                .float()
            )
            probabilities = torch.softmax(logits, dim=-1)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency = time.perf_counter() - started
        return {
            "action": int(probabilities.argmax().item()),
            "logits": logits.cpu().tolist(),
            "probabilities": probabilities.cpu().tolist(),
            "token_ids": choices,
            "prompt": prefix,
            "input_tokens": len(ids),
            "forward_seconds": latency,
            "generated_tokens": 0,
            "model_dir": self.model_dir,
        }
