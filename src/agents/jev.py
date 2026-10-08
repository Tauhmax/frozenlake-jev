"""One forward pass, no generation, local files only."""

import json
import time
from pathlib import Path

from src.agents.prompt import prepare_choice_tokens


class ChoiceTokenJev:
    def __init__(self, model_dir, device="cpu", load_in_4bit=False):
        path = Path(model_dir).resolve()
        if not (path / "config.json").is_file():
            raise ValueError(f"No local model in {path}; see models/base/README.md")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.torch_npu = None
        if str(device).startswith("npu"):
            import torch_npu

            self.torch_npu = torch_npu
        self.device = torch.device(device)
        self.model_dir = str(path)
        manifest = path / "checkpoint.json"
        self.checkpoint = (
            json.loads(manifest.read_text()) if manifest.exists() else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.load_in_4bit = load_in_4bit
        options = {"local_files_only": True, "torch_dtype": "auto"}
        if load_in_4bit:
            if self.device.type != "cuda" or not torch.cuda.is_available():
                raise ValueError("4-bit loading requires a CUDA device")
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            options["device_map"] = {"": str(self.device)}
        model_class = AutoModelForCausalLM
        model_type = json.loads((path / "config.json").read_text())["model_type"]
        if model_type == "qwen3_5":
            from transformers import Qwen3_5ForConditionalGeneration

            model_class = Qwen3_5ForConditionalGeneration
        self.model = model_class.from_pretrained(path, **options)
        if not load_in_4bit:
            self.model.to(self.device)
        self.model.eval()

    def _synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        elif self.device.type == "npu":
            self.torch.npu.synchronize(self.device)

    def predict(self, prompt):
        torch = self.torch
        prefix, ids, choices = prepare_choice_tokens(self.tokenizer, prompt)
        inputs = torch.tensor([ids], device=self.device)
        self._synchronize()
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
        self._synchronize()
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
            "checkpoint": self.checkpoint,
            "quantization": "nf4" if self.load_in_4bit else None,
        }
