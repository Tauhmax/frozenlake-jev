"""Focused checks for token ranking and controlled interventions."""

import unittest
from types import SimpleNamespace

from decision_tokens.probe import rank_tokens, run_probe


class DecisionTokenTests(unittest.TestCase):
    def test_rankings_and_ties(self):
        norms = [9, 1, 5, 3, 7, 2, 8, 4]
        ids = list(range(len(norms)))
        self.assertEqual(rank_tokens(norms, ids, "shortest"), [1, 5, 3, 7])
        self.assertEqual(rank_tokens(norms, ids, "longest"), [0, 6, 4, 2])
        self.assertEqual(rank_tokens(norms, ids, "middle"), [3, 7, 2, 4])
        self.assertEqual(rank_tokens([1] * 5, [4, 3, 2, 1, 0], "longest"), [0, 1, 2, 3])

    def test_exact_insertion_and_probability_spaces(self):
        import torch

        seen = []

        def model(input_ids, **kwargs):
            ids = input_ids[0].tolist()
            seen.append(ids)
            logits = torch.zeros(1, len(ids), 8)
            if len(ids) == 3:
                logits[0, -1, ids[1]] = 2
            return SimpleNamespace(logits=logits)

        engine = SimpleNamespace(
            torch=torch,
            device="cpu",
            model=model,
            forward_options={},
            tokenizer=SimpleNamespace(encode=lambda text, **kwargs: [int(text)]),
        )
        result = run_probe(
            engine,
            {"prefix": "6", "suffix": "7"},
            [{"id": i} for i in range(4)],
        )
        self.assertEqual(seen, [[6, 7]] + [[6, i, 7] for i in range(4)])
        self.assertAlmostEqual(result["baseline_absent"]["candidate_mass"], 0.5)
        matrix = result["centered_logit_influence"]
        for i, row in enumerate(result["interventions"]):
            self.assertAlmostEqual(sum(row["probabilities"]), 1, places=6)
            self.assertEqual(max(range(4), key=row["probabilities"].__getitem__), i)
            self.assertAlmostEqual(matrix[i][i], 1.5)
            self.assertAlmostEqual(sum(matrix[i]), 0)
        for column in zip(*matrix):
            self.assertAlmostEqual(sum(column), 0)


if __name__ == "__main__":
    unittest.main()
