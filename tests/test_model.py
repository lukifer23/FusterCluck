from __future__ import annotations

import torch

from fustercluck.models.components import FusterCluckDecoder


def test_forward_shape():
    model = FusterCluckDecoder(vocab_size=128, dim=256, num_layers=2, num_heads=8, num_kv_heads=2)
    input_ids = torch.randint(0, 128, (2, 16))
    logits = model(input_ids)
    assert logits.shape == (2, 16, 128)
