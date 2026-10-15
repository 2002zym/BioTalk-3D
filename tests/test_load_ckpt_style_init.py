import os
import tempfile
import unittest

import torch
import torch.nn as nn

from EATalker.utils.utils import load_ckpt


class _DummyDecoder(nn.Module):
    def __init__(self, identity_num: int, emb_dim: int):
        super().__init__()
        self.learnable_style_emb = nn.Embedding(identity_num, emb_dim)
        self.audio_feature_map = nn.Linear(emb_dim, emb_dim, bias=False)
        self.multimodal_feature_map = nn.Linear(emb_dim, emb_dim, bias=False)


class _DummyModel(nn.Module):
    def __init__(self, identity_num: int = 2, emb_dim: int = 4):
        super().__init__()
        self.decoder = _DummyDecoder(identity_num, emb_dim)


class LoadCkptStyleInitTest(unittest.TestCase):
    def test_out_of_range_style_indices_fallback_without_raise(self):
        model = _DummyModel(identity_num=2, emb_dim=4)
        before_second = model.decoder.learnable_style_emb.weight.detach().clone()[1]

        ckpt = {
            "decoder.learnable_style_emb.weight": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        }
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fp:
            ckpt_path = fp.name
        try:
            torch.save(ckpt, ckpt_path)
            load_ckpt(model, ckpt_path, style_emb_init_indices=[919, 920])
        finally:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

        got = model.decoder.learnable_style_emb.weight.detach()
        self.assertTrue(torch.allclose(got[0], ckpt["decoder.learnable_style_emb.weight"][0]))
        self.assertTrue(torch.allclose(got[1], before_second))

    def test_valid_style_indices_still_applied(self):
        model = _DummyModel(identity_num=2, emb_dim=4)
        ckpt = {
            "decoder.learnable_style_emb.weight": torch.tensor([
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0],
            ]),
        }
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as fp:
            ckpt_path = fp.name
        try:
            torch.save(ckpt, ckpt_path)
            load_ckpt(model, ckpt_path, style_emb_init_indices=[1, 2])
        finally:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

        got = model.decoder.learnable_style_emb.weight.detach()
        self.assertTrue(torch.allclose(got[0], ckpt["decoder.learnable_style_emb.weight"][1]))
        self.assertTrue(torch.allclose(got[1], ckpt["decoder.learnable_style_emb.weight"][2]))


if __name__ == "__main__":
    unittest.main()
