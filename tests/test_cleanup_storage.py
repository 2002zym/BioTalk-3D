import json
import tempfile
import unittest
import os
from pathlib import Path

from tools.cleanup_storage import _collect_referenced_paths, _prune_ckpts


class CleanupStorageTests(unittest.TestCase):
    def test_collect_referenced_paths(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            a = td_path / "a.npy"
            b = td_path / "b.npy"
            a.write_bytes(b"x")
            b.write_bytes(b"y")
            manifest = td_path / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "entries": {
                            "k1": {
                                "audio_feature_path": str(a),
                                "eeg_feature_path": str(b),
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            refs = _collect_referenced_paths([manifest])
            self.assertIn(a.resolve(), refs)
            self.assertIn(b.resolve(), refs)

    def test_prune_ckpts_keep_best_and_latest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "exp" / "audio+eeg"
            run_dir.mkdir(parents=True, exist_ok=True)
            ckpt1 = run_dir / "eatalker_001.pt"
            ckpt2 = run_dir / "eatalker_002.pt"
            ckpt3 = run_dir / "eatalker_003.pt"
            ckpt1.write_bytes(b"a")
            ckpt2.write_bytes(b"bb")
            ckpt3.write_bytes(b"ccc")
            os.utime(ckpt1, (1, 1))
            os.utime(ckpt2, (2, 2))
            os.utime(ckpt3, (3, 3))
            (run_dir / "best_ckpt.txt").write_text(str(ckpt2), encoding="utf-8")
            stat = _prune_ckpts(root, current_run_dir=None, dry_run=False)
            self.assertEqual(stat["removed_ckpts"], 1.0)
            self.assertFalse(ckpt1.exists())
            self.assertTrue(ckpt2.exists())
            self.assertTrue(ckpt3.exists())


if __name__ == "__main__":
    unittest.main()
