import unittest

from tools.run_biotalk_targeted_fix import (
    MetricPair,
    pick_audio_candidates_by_visual_and_metrics,
    pick_best_by_weighted_rank,
    relative_improvement,
    should_run_stage2,
)


class BioTalkTargetedFixTests(unittest.TestCase):
    def test_relative_improvement(self):
        self.assertAlmostEqual(relative_improvement(0.10, 0.08), 0.2)
        self.assertAlmostEqual(relative_improvement(0.10, 0.12), -0.2)

    def test_should_run_stage2_when_any_metric_below_threshold(self):
        audio = MetricPair(mve=1.0, bs_mae=1.0)
        cand = MetricPair(mve=0.94, bs_mae=0.98)
        self.assertTrue(
            should_run_stage2(
                audio,
                cand,
                min_mve_improve=0.05,
                min_bs_improve=0.03,
            )
        )

    def test_should_not_run_stage2_when_both_metrics_meet_threshold(self):
        audio = MetricPair(mve=1.0, bs_mae=1.0)
        cand = MetricPair(mve=0.90, bs_mae=0.95)
        self.assertFalse(
            should_run_stage2(
                audio,
                cand,
                min_mve_improve=0.05,
                min_bs_improve=0.03,
            )
        )

    def test_pick_best_by_weighted_rank_prefers_balanced_ckpt(self):
        rows = [
            {
                "ckpt": "a.pt",
                "MVE": 0.10,
                "LVE": 0.40,
                "UFVE": 0.60,
                "FDD": 0.40,
            },
            {
                "ckpt": "b.pt",
                "MVE": 0.11,
                "LVE": 0.20,
                "UFVE": 0.25,
                "FDD": 0.20,
            },
            {
                "ckpt": "c.pt",
                "MVE": 0.08,
                "LVE": 0.60,
                "UFVE": 0.65,
                "FDD": 0.55,
            },
        ]
        best = pick_best_by_weighted_rank(rows)
        self.assertEqual(best["ckpt"], "b.pt")

    def test_pick_audio_candidates_by_visual_and_metrics_returns_two(self):
        rows = [
            {
                "ckpt": "a.pt",
                "MVE": 0.10,
                "bs_mae_51d": 0.20,
                "mouth_mae_51d": 0.20,
                "upper_face_mae_51d": 0.20,
                "velocity_mae": 0.20,
                "acceleration_mae": 0.20,
                "FDD": 0.20,
                "LVD": 0.20,
                "UFVE": 0.20,
            },
            {
                "ckpt": "b.pt",
                "MVE": 0.11,
                "bs_mae_51d": 0.21,
                "mouth_mae_51d": 0.21,
                "upper_face_mae_51d": 0.21,
                "velocity_mae": 0.10,
                "acceleration_mae": 0.10,
                "FDD": 0.10,
                "LVD": 0.10,
                "UFVE": 0.10,
            },
            {
                "ckpt": "c.pt",
                "MVE": 0.09,
                "bs_mae_51d": 0.25,
                "mouth_mae_51d": 0.25,
                "upper_face_mae_51d": 0.25,
                "velocity_mae": 0.30,
                "acceleration_mae": 0.30,
                "FDD": 0.30,
                "LVD": 0.30,
                "UFVE": 0.30,
            },
        ]
        picked = pick_audio_candidates_by_visual_and_metrics(
            metric_rows=rows,
            candidate_count=2,
            visual_weight=0.5,
        )
        self.assertEqual(len(picked), 2)
        self.assertIn(picked[0]["ckpt"], {"a.pt", "b.pt"})


if __name__ == "__main__":
    unittest.main()
