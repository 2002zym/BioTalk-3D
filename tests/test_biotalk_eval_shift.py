import unittest

import numpy as np


from tools.eval_biotalk_protocol import shift_eeg_array


class BioTalkEvalShiftTests(unittest.TestCase):
    def test_shift_positive_moves_signal_right(self):
        src = np.arange(10, dtype=np.float32).reshape(5, 2)
        got = shift_eeg_array(src, shift_steps=1, channels=2)
        self.assertEqual(got.shape, src.shape)
        self.assertTrue(np.allclose(got[0], 0.0))
        self.assertTrue(np.allclose(got[1:], src[:-1]))

    def test_shift_negative_moves_signal_left(self):
        src = np.arange(10, dtype=np.float32).reshape(5, 2)
        got = shift_eeg_array(src, shift_steps=-2, channels=2)
        self.assertEqual(got.shape, src.shape)
        self.assertTrue(np.allclose(got[:-2], src[2:]))
        self.assertTrue(np.allclose(got[-2:], 0.0))


if __name__ == "__main__":
    unittest.main()
