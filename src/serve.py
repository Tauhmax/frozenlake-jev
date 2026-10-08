"""Local choice-logit HTTP service; model stays loaded between requests."""

import argparse
import hashlib
import json
import platform
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.metadata import version
from pathlib import Path

from src.agents.jev import ChoiceTokenJev
from src.agents.prompt import PROMPT_VERSION, format_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="models/Qwen3.5-0.8B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    policy = ChoiceTokenJev(args.model_dir, args.device, args.load_in_4bit)
    policy.torch.manual_seed(42)
    source = b"".join(
        Path(p).read_bytes()
        for p in ["src/agents/jev.py", "src/agents/prompt.py", "src/agents/actions.py"]
    )
    metadata = {
        "checkpoint": policy.checkpoint,
        "model_dir": policy.model_dir,
        "quantization": "nf4" if args.load_in_4bit else None,
        "device": args.device,
        "dtype": str(policy.model.dtype),
        "python": platform.python_version(),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "scoring_code_sha256": hashlib.sha256(source).hexdigest(),
        "seed": 42,
        "prompt_version": PROMPT_VERSION,
        "max_batch_size": 16,
        "max_input_tokens": 2048,
    }

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, data):
            payload = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"ready": True, "model": metadata})
            else:
                self.reply(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/score":
                self.reply(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1_000_000:
                    raise ValueError("Request body must be between 1 byte and 1 MB")
                data = json.loads(self.rfile.read(length))
                if "states" in data and "prompts" in data:
                    raise ValueError("Supply either states or prompts, not both")
                prompts = data.get("prompts")
                if "states" in data:
                    states = data["states"]
                    if (
                        not isinstance(states, list)
                        or not 1 <= len(states) <= 16
                        or any(
                            not isinstance(s, dict) or "board" not in s for s in states
                        )
                    ):
                        raise ValueError(
                            "states must contain 1 to 16 objects with board"
                        )
                    prompts = [
                        format_prompt(s["board"], s.get("gamma", 0.99)) for s in states
                    ]
                if (
                    not isinstance(prompts, list)
                    or not 1 <= len(prompts) <= 16
                    or any(not isinstance(p, str) or not p for p in prompts)
                ):
                    raise ValueError("prompts must contain 1 to 16 nonempty strings")
                self.reply(200, {"predictions": policy.predict_batch(prompts)})
            except (ValueError, TypeError, AttributeError) as error:
                self.reply(400, {"error": str(error)})
            except Exception as error:
                traceback.print_exc()
                self.reply(500, {"error": str(error)})

        def log_message(self, *_):
            pass

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"READY http://127.0.0.1:{args.port} {json.dumps(metadata)}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
