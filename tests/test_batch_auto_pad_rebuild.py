import tempfile
import unittest
from pathlib import Path

from tools.batch_auto_pad_rebuild import (
    discover_subjects,
    parse_subjects_arg,
    should_mark_suspect,
    write_secondary_rows_csv,
)


class BatchAutoPadRebuildTests(unittest.TestCase):
    def test_should_mark_suspect(self):
        self.assertTrue(should_mark_suspect(0.08, 0.05, 0.05, 0.04))
        self.assertFalse(should_mark_suspect(0.03, 0.05, 0.05, 0.04))
        self.assertFalse(should_mark_suspect(0.08, 0.02, 0.05, 0.04))

    def test_parse_subjects_arg(self):
        got = parse_subjects_arg("lc, wy, zf")
        self.assertEqual(got, ["lc", "wy", "zf"])

    def test_discover_subjects(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "lc" / "audio").mkdir(parents=True)
            (root / "lc" / "EEG").mkdir(parents=True)
            (root / "wy" / "Audio").mkdir(parents=True)
            (root / "wy" / "EEG").mkdir(parents=True)
            (root / "notes").mkdir(parents=True)
            (root / "labels.xlsx").write_text("x", encoding="utf-8")
            got = discover_subjects(root)
            self.assertEqual(got, ["lc", "wy"])

    def test_write_secondary_rows_csv_has_expected_fields(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "secondary.csv"
            rows = [
                {
                    "subject": "zym",
                    "sample_id": "s1_sentence001",
                    "session": "s1",
                    "emotion": "happy",
                    "end_pad_ms": 120,
                    "end_rms": 0.061,
                    "prev_rms": 0.045,
                    "reason": "tail_rms_high",
                }
            ]
            write_secondary_rows_csv(rows, out)
            text = out.read_text(encoding="utf-8")
            self.assertIn("subject,sample_id,session,emotion,end_pad_ms,end_rms,prev_rms,reason", text)
            self.assertIn("zym,s1_sentence001,s1,happy,120,0.061,0.045,tail_rms_high", text)


if __name__ == "__main__":
    unittest.main()
