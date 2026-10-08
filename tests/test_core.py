import unittest
from collections import deque

import numpy as np

from src.agents.prompt import prepare_choice_tokens
from src.environments.frozenlake import make_env
from src.oracles.optimal_q import optimal_actions, value_iteration


class CoreTests(unittest.TestCase):
    def test_vi_against_bfs_and_terminal_rules(self):
        for rows in (["SF", "FG"], ["SHG"], ["SFFF", "FHFH", "FFFH", "HFFG"]):
            env = make_env(rows)
            self.addCleanup(env.close)
            q, info = value_iteration(env)
            self.assertLessEqual(info["bellman_residual"], 1e-12)
            goal = list("".join(rows)).index("G")
            distances = {goal: 0}
            queue = deque([goal])
            while queue:
                target = queue.popleft()
                for s, cell in enumerate(env.desc.flatten()):
                    if cell in (b"H", b"G") or s in distances:
                        continue
                    if any(env.P[s][a][0][1] == target for a in range(4)):
                        distances[s] = distances[target] + 1
                        queue.append(s)
            for s, cell in enumerate(env.desc.flatten()):
                expected = 0.99 ** (distances[s] - 1) if s in distances else 0
                if cell in (b"H", b"G"):
                    expected = 0
                self.assertAlmostEqual(max(q[s]), expected)
            self.assertEqual(env.P[0][0][0][1], 0)
        env = make_env(["SF", "FG"])
        self.addCleanup(env.close)
        self.assertEqual(optimal_actions(value_iteration(env)[0][0]), [1, 2])
        env = make_env(["SHG"])
        self.addCleanup(env.close)
        self.assertEqual(env.P[0][2][0], (1.0, 1, 0.0, True))
        np.testing.assert_array_equal(value_iteration(env)[0], np.zeros((3, 4)))

    def test_choice_position_and_rejection(self):
        class Tokenizer:
            chat_template = None

            def encode(self, text, **kwargs):
                return [ord(c) for c in text]

        prefix, ids, choices = prepare_choice_tokens(Tokenizer(), "Map")
        self.assertTrue(prefix.endswith("Answer:"))
        self.assertEqual(ids[-1], ord(":"))
        self.assertEqual(choices, [65, 66, 67, 68])

        class BadTokenizer(Tokenizer):
            def encode(self, text, **kwargs):
                return super().encode(text) + ([0] if text.endswith("A") else [])

        with self.assertRaises(ValueError):
            prepare_choice_tokens(BadTokenizer(), "Map")


if __name__ == "__main__":
    unittest.main()
