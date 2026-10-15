import unittest
import tempfile
from pathlib import Path


from tools.process_eatalker_v2 import (
    build_timecode_seconds_sequence,
    build_marker_pairs,
    compute_end_pad_seconds,
    compute_effective_tail_pad_ms,
    discover_audio_base_dir,
    discover_slate_session_dirs,
    normalize_eeg_preprocess_mode,
    normalize_timecode_mode,
    normalize_timecode_repair_mode,
    parse_dirty_samples,
    parse_session_pad_overrides,
    repair_timecode_jump_stitch,
    resolve_session_end_pad_ms,
    summarize_timecode_sequence,
    split_counts_8_1_1,
)


class ProcessEaTalkerV2Tests(unittest.TestCase):
    def test_discover_audio_base_dir_supports_audio_and_Audio(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "Audio").mkdir(parents=True)
            got = discover_audio_base_dir(root)
            self.assertEqual(got, root / "Audio")

    def test_discover_slate_session_dirs_sorted_and_non_contiguous(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in [
                "20260124_MySlate_21",
                "20260124_MySlate_17",
                "20260124_MySlate_19",
                "20260124_MySlate_18",
            ]:
                (root / name).mkdir(parents=True)
            got = discover_slate_session_dirs(root)
            self.assertEqual(
                got,
                [
                    (17, "20260124_MySlate_17"),
                    (18, "20260124_MySlate_18"),
                    (19, "20260124_MySlate_19"),
                    (21, "20260124_MySlate_21"),
                ],
            )

    def test_build_marker_pairs(self):
        onsets = [0.0, 1.2, 2.0, 4.3, 5.0, 7.0]
        pairs = build_marker_pairs(onsets)
        self.assertEqual(pairs, [(0.0, 1.2), (2.0, 4.3), (5.0, 7.0)])

    def test_build_marker_pairs_rejects_odd(self):
        with self.assertRaises(ValueError):
            build_marker_pairs([0.1, 1.2, 3.4])

    def test_split_counts_8_1_1_for_25(self):
        train_n, val_n, test_n = split_counts_8_1_1(25)
        self.assertEqual((train_n, val_n, test_n), (20, 2, 3))

    def test_split_counts_small_n(self):
        self.assertEqual(split_counts_8_1_1(1), (1, 0, 0))
        self.assertEqual(split_counts_8_1_1(2), (1, 0, 1))
        self.assertEqual(split_counts_8_1_1(3), (1, 1, 1))

    def test_parse_dirty_samples(self):
        raw = "s5_sentence001, s7_sentence023\ns8_sentence125  s6_sentence010"
        parsed = parse_dirty_samples(raw)
        self.assertEqual(
            parsed,
            {
                "s5_sentence001",
                "s6_sentence010",
                "s7_sentence023",
                "s8_sentence125",
            },
        )

    def test_compute_end_pad_seconds_only_on_selected_samples(self):
        selected = {"s7_sentence005", "s7_sentence116"}
        self.assertEqual(compute_end_pad_seconds("s7_sentence005", 120, selected), 0.12)
        self.assertEqual(compute_end_pad_seconds("s7_sentence117", 120, selected), 0.0)
        self.assertEqual(compute_end_pad_seconds("s7_sentence116", 0, selected), 0.0)

    def test_resolve_session_end_pad_ms_prefers_session_override(self):
        selected = {"s14_sentence001", "s15_sentence007"}
        overrides = {"s14": 600, "s15": 500}
        self.assertEqual(
            resolve_session_end_pad_ms(
                sample_id="s14_sentence001",
                session_name="s14",
                default_end_pad_ms=450,
                selected_samples=selected,
                session_overrides=overrides,
            ),
            600,
        )
        self.assertEqual(
            resolve_session_end_pad_ms(
                sample_id="s15_sentence007",
                session_name="s15",
                default_end_pad_ms=450,
                selected_samples=selected,
                session_overrides=overrides,
            ),
            500,
        )
        self.assertEqual(
            resolve_session_end_pad_ms(
                sample_id="s13_sentence001",
                session_name="s13",
                default_end_pad_ms=450,
                selected_samples=selected,
                session_overrides=overrides,
            ),
            0,
        )

    def test_compute_effective_tail_pad_ms_keeps_needed_tail_then_stops(self):
        sr = 1000
        body = [0.10] * 800  # strong speech body
        # first 180ms remains voiced, then silence
        added = [0.08] * 180 + [0.0] * 320
        signal = body + added
        pad_ms = compute_effective_tail_pad_ms(
            signal=signal,
            sample_rate=sr,
            base_end_sample=len(body),
            max_end_sample=len(signal),
            alpha=0.25,
            abs_floor=0.015,
            frame_ms=10,
            min_keep_ms=40,
        )
        # 180ms voiced + 40ms keep ~= 220ms
        self.assertGreaterEqual(pad_ms, 210)
        self.assertLessEqual(pad_ms, 230)

    def test_compute_effective_tail_pad_ms_returns_zero_for_silent_added_region(self):
        sr = 1000
        body = [0.10] * 600
        added = [0.0] * 400
        signal = body + added
        pad_ms = compute_effective_tail_pad_ms(
            signal=signal,
            sample_rate=sr,
            base_end_sample=len(body),
            max_end_sample=len(signal),
            alpha=0.25,
            abs_floor=0.015,
            frame_ms=10,
            min_keep_ms=40,
        )
        self.assertEqual(pad_ms, 0)

    def test_normalize_eeg_preprocess_mode(self):
        self.assertEqual(normalize_eeg_preprocess_mode("strict"), "strict")
        self.assertEqual(normalize_eeg_preprocess_mode(" NONE "), "none")
        with self.assertRaises(ValueError):
            normalize_eeg_preprocess_mode("bad_mode")

    def test_parse_session_pad_overrides(self):
        got = parse_session_pad_overrides('{"s14": 600, "15": 500}')
        self.assertEqual(got, {"s14": 600, "s15": 500})

    def test_timecode_sequence_robust_handles_unstable_msec(self):
        seq = [
            "06:56:54:13.000",
            "06:56:54:13.998",
            "06:56:54:15.000",
            "06:56:54:15.999",
            "06:56:54:17.000",
        ]
        out = build_timecode_seconds_sequence(seq, fps=60.0, mode="robust")
        deltas_ms = [(out[i] - out[i - 1]) * 1000.0 for i in range(1, len(out))]
        self.assertTrue(all(d > 0 for d in deltas_ms))
        self.assertTrue(all(d < 40 for d in deltas_ms))
        self.assertAlmostEqual(deltas_ms[0], 16.6333333, places=2)
        self.assertAlmostEqual(deltas_ms[1], 16.7000000, places=2)

    def test_timecode_sequence_robust_keeps_30fps_pattern(self):
        seq = [
            "00:00:00:00.010",
            "00:00:00:02.970",
            "00:00:00:04.020",
            "00:00:00:06.960",
        ]
        out = build_timecode_seconds_sequence(seq, fps=60.0, mode="robust")
        deltas_ms = [(out[i] - out[i - 1]) * 1000.0 for i in range(1, len(out))]
        for d in deltas_ms:
            self.assertLess(abs(d - 33.3333333), 0.25)

    def test_timecode_sequence_robust_suppresses_extreme_jump(self):
        seq = [
            "00:00:00:54.000",
            "00:00:00:55.000",
            "00:00:00:56.000",
            "00:00:02:11.000",  # nominal ff-only gap would be +1.25s, treat as outlier
            "00:00:02:12.000",
        ]
        out = build_timecode_seconds_sequence(seq, fps=60.0, mode="robust")
        deltas_ms = [(out[i] - out[i - 1]) * 1000.0 for i in range(1, len(out))]
        self.assertTrue(all(d > 0 for d in deltas_ms))
        self.assertTrue(all(d < 100 for d in deltas_ms))

    def test_timecode_jump_stitch_repair_fixes_large_run(self):
        fps = 60.0
        exp = 1.0 / fps
        # Simulate one jump followed by a corrupted run that shows repeated 75ms cadence.
        seq = [i * exp for i in range(40)]
        t = seq[-1]
        for _ in range(10):
            t += 0.075
            seq.append(t)
        for _ in range(20):
            t += exp
            seq.append(t)

        before = summarize_timecode_sequence(seq, fps=fps)
        self.assertGreater(before["gt_40ms_count"], 0)

        repaired, diag = repair_timecode_jump_stitch(seq, fps=fps)
        after = summarize_timecode_sequence(repaired, fps=fps)
        self.assertEqual(after["neg_count"], 0)
        self.assertEqual(after["gt_40ms_count"], 0)
        self.assertTrue(diag["applied"])
        self.assertGreater(diag["applied_offset_ms"], 0.0)

    def test_timecode_jump_stitch_repair_keeps_clean_sequence(self):
        fps = 60.0
        exp = 1.0 / fps
        seq = [i * exp for i in range(120)]
        repaired, diag = repair_timecode_jump_stitch(seq, fps=fps)
        self.assertEqual(len(repaired), len(seq))
        self.assertFalse(diag["applied"])
        self.assertAlmostEqual(repaired[-1], seq[-1], places=8)

    def test_normalize_timecode_repair_mode(self):
        self.assertEqual(normalize_timecode_repair_mode("stitch"), "stitch")
        self.assertEqual(normalize_timecode_repair_mode(" NONE "), "none")
        with self.assertRaises(ValueError):
            normalize_timecode_repair_mode("bad_mode")

    def test_normalize_timecode_mode(self):
        self.assertEqual(normalize_timecode_mode("robust"), "robust")
        self.assertEqual(normalize_timecode_mode(" ADAPTIVE "), "adaptive")
        self.assertEqual(normalize_timecode_mode("raw"), "raw")
        with self.assertRaises(ValueError):
            normalize_timecode_mode("bad_mode")

    def test_timecode_sequence_adaptive_avoids_50ms_artifacts_in_30fps_like_stream(self):
        seq = [
            "08:43:17:56.997",
            "08:43:17:58.997",
            "08:43:18:00.997",
            "08:43:18:02.998",
            "08:43:18:04.997",
            "08:43:18:06.998",
            "08:43:18:08.998",
            "08:43:18:11.000",
            "08:43:18:13.001",
            "08:43:18:14.999",
            "08:43:18:17.001",
            "08:43:18:19.002",
            "08:43:18:21.002",
            "08:43:18:23.001",
        ]
        robust = build_timecode_seconds_sequence(seq, fps=60.0, mode="robust")
        adaptive = build_timecode_seconds_sequence(seq, fps=60.0, mode="adaptive")
        robust_stat = summarize_timecode_sequence(robust, fps=60.0)
        adaptive_stat = summarize_timecode_sequence(adaptive, fps=60.0)
        self.assertGreater(robust_stat["gt_40ms_count"], 0)
        self.assertEqual(adaptive_stat["gt_40ms_count"], 0)
        self.assertEqual(adaptive_stat["neg_count"], 0)


if __name__ == "__main__":
    unittest.main()
