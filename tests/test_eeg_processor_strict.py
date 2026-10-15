import unittest

import numpy as np

from dataset.eeg_processor import EEGProcessor


class EEGProcessorStrictTests(unittest.TestCase):
    def test_preprocess_requires_exact_channel_count(self):
        proc = EEGProcessor(eeg_channels=60, target_sfreq=1000.0)
        data = np.random.randn(59, 2000).astype(np.float32)
        with self.assertRaisesRegex(ValueError, "expected 60"):
            proc.preprocess_data(data, original_sfreq=1000.0)


if __name__ == "__main__":
    unittest.main()
