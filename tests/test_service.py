"""HTTP batch limits and error details, without loading model weights."""

import threading
import unittest
from http.server import HTTPServer
from types import SimpleNamespace
from unittest.mock import patch

from src import serve
from src.evaluation.bulk import request


class ServiceTests(unittest.TestCase):
    def test_configured_batch_limit(self):
        self.check_service(32)

    def test_unlimited_batch(self):
        self.check_service(None)

    def check_service(self, limit):
        ready = threading.Event()
        servers = []

        class Server(HTTPServer):
            def serve_forever(self):
                ready.set()
                super().serve_forever(poll_interval=0.01)

        def factory(address, handler):
            server = Server((address[0], 0), handler)
            servers.append(server)
            return server

        policy = SimpleNamespace(
            checkpoint=None,
            model_dir="test",
            model=SimpleNamespace(dtype="float32"),
            torch=SimpleNamespace(manual_seed=lambda seed: None),
            predict_batch=lambda prompts: [{"action": 0} for _ in prompts],
        )
        with (
            patch.object(serve, "ChoiceTokenJev", return_value=policy),
            patch.object(serve, "HTTPServer", side_effect=factory),
            patch(
                "sys.argv",
                ["serve"] + (["--max-batch-size", str(limit)] if limit else []),
            ),
            patch("builtins.print"),
        ):
            thread = threading.Thread(target=serve.main, daemon=True)
            thread.start()
            try:
                self.assertTrue(ready.wait(5), "Server did not start")
                url = f"http://127.0.0.1:{servers[0].server_port}"
                self.assertEqual(
                    request(url + "/health")["model"]["max_batch_size"], limit
                )
                for payload in (
                    {"prompts": ["test"] * 32},
                    {"states": [{"board": "PG", "remaining_steps": 30}] * 32},
                ):
                    self.assertEqual(
                        len(request(url + "/score", payload)["predictions"]), 32
                    )
                if limit:
                    with self.assertRaisesRegex(
                        RuntimeError, r"HTTP 400.*max_batch_size=32"
                    ):
                        request(url + "/score", {"prompts": ["test"] * 33})
                else:
                    self.assertEqual(
                        len(
                            request(url + "/score", {"prompts": ["test"] * 64})[
                                "predictions"
                            ]
                        ),
                        64,
                    )
                with patch.object(
                    policy, "predict_batch", side_effect=ValueError("forward failed")
                ):
                    with patch.object(serve.traceback, "print_exc") as log_error:
                        with self.assertRaisesRegex(
                            RuntimeError, "HTTP 500.*inference.*forward failed"
                        ):
                            request(url + "/score", {"prompts": ["test"]}, timeout=5)
                        log_error.assert_called_once()
            finally:
                if ready.is_set():
                    servers[0].shutdown()
                thread.join(timeout=5)
