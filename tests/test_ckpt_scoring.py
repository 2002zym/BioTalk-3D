import unittest

from tools.ckpt_scoring import pick_best_by_rank, score_rows_by_rank


class CkptScoringTests(unittest.TestCase):
    def test_pick_best_by_rank_uses_four_metrics(self):
        rows = [
            {"ckpt": "a.pt", "MVE": 0.10, "LVE": 0.20, "UFVE": 0.30, "FDD": 0.40},
            {"ckpt": "b.pt", "MVE": 0.11, "LVE": 0.10, "UFVE": 0.10, "FDD": 0.10},
            {"ckpt": "c.pt", "MVE": 0.08, "LVE": 0.50, "UFVE": 0.50, "FDD": 0.50},
        ]
        best = pick_best_by_rank(
            rows,
            metrics=("MVE", "LVE", "UFVE", "FDD"),
            weights={"MVE": 0.20, "LVE": 0.15, "UFVE": 0.35, "FDD": 0.30},
        )
        self.assertEqual(best["ckpt"], "b.pt")

    def test_score_rows_includes_rank_score(self):
        rows = [
            {"ckpt": "a.pt", "MVE": 1.0, "LVE": 1.0, "UFVE": 1.0, "FDD": 1.0},
            {"ckpt": "b.pt", "MVE": 2.0, "LVE": 2.0, "UFVE": 2.0, "FDD": 2.0},
        ]
        scored = score_rows_by_rank(rows)
        self.assertEqual(scored[0]["ckpt"], "a.pt")
        self.assertIn("rank_score", scored[0])


if __name__ == "__main__":
    unittest.main()
