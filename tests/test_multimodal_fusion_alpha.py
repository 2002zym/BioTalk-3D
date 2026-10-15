import unittest

import torch

from models.multimodal_fusion import EfficientMultimodalFusion


class FusionAlphaTests(unittest.TestCase):
    def test_alpha_zero_degenerates_to_audio_only(self):
        torch.manual_seed(0)
        model = EfficientMultimodalFusion(
            audio_dim=8,
            eeg_dim=4,
            hidden_dim=8,
            num_heads=2,
            alpha_mode="scalar",
            alpha_init=0.0,
        )
        audio = torch.randn(2, 5, 8)
        eeg = torch.randn(2, 5, 4)
        out = model(audio, eeg)
        self.assertTrue(torch.allclose(out, audio, atol=1e-7, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
