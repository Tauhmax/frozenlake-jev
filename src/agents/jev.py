"""One forward pass, no generation, local files only."""

import inspect
import json
import time
from pathlib import Path

from src.agents.prompt import prepare_choice_tokens


class ChoiceTokenJev:
    def __init__(
        self, model_dir, device="cpu", load_in_4bit=False, max_input_tokens=None
    ):
        path = Path(model_dir).resolve()
        if not (path / "config.json").is_file():
            raise ValueError(f"No local model in {path}; see README.md")
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
        self.max_input_tokens = max_input_tokens
        options = {"local_files_only": True, "dtype": "auto"}
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
        self.forward_options = {}
        if "logits_to_keep" in inspect.signature(self.model.forward).parameters:
            self.forward_options["logits_to_keep"] = 1

    def _synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        elif self.device.type == "npu":
            self.torch.npu.synchronize(self.device)

    def predict(self, prompt):
        return self.predict_batch([prompt])[0]

    def predict_batch(self, prompts):
        if not prompts:
            raise ValueError("Batch must contain at least one prompt")
        torch = self.torch
        prepared = [prepare_choice_tokens(self.tokenizer, p) for p in prompts]
        lengths = [len(ids) for _, ids, _ in prepared]
        if self.max_input_tokens is not None and max(lengths) > self.max_input_tokens:
            raise ValueError(f"Prompt exceeds max_input_tokens={self.max_input_tokens}")
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        width = max(lengths)
        inputs = torch.tensor(
            [[pad] * (width - len(ids)) + ids for _, ids, _ in prepared],
            device=self.device,
        )
        mask = torch.tensor(
            [[0] * (width - n) + [1] * n for n in lengths], device=self.device
        )
        self._synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            last_logits = self.model(
                input_ids=inputs,
                attention_mask=mask,
                use_cache=False,
                **self.forward_options,
            ).logits[:, -1, :]
            choices = torch.tensor([c for _, _, c in prepared], device=self.device)
            logits = last_logits.gather(1, choices).float()
            probabilities = torch.softmax(logits, dim=-1)
        self._synchronize()
        latency = time.perf_counter() - started
        all_logits = logits.cpu().tolist()
        all_probabilities = probabilities.cpu().tolist()
        results = []
        for i, (prefix, ids, token_ids) in enumerate(prepared):
            probs = all_probabilities[i]
            results.append(
                {
                    "action": max(range(4), key=probs.__getitem__),
                    "logits": all_logits[i],
                    "probabilities": probs,
                    "token_ids": token_ids,
                    "prompt": prefix,
                    "input_tokens": len(ids),
                    "forward_seconds": latency / len(prompts),
                    "batch_forward_seconds": latency,
                    "batch_size": len(prompts),
                    "generated_tokens": 0,
                    "model_dir": self.model_dir,
                    "checkpoint": self.checkpoint,
                    "quantization": "nf4" if self.load_in_4bit else None,
                }
            )
        return results
