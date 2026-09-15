from __future__ import annotations

import unittest

import torch
from torch import nn

from tinyad_infer.model import AnomalyDetector, DetectorConfig
from tinyad_infer.scoring import aggregate_image_score


class StubBackbone(nn.Module):
    def get_intermediate_layers(self, images, n, return_class_token=False):
        batch = images.shape[0]
        values = torch.linspace(0.0, 1.0, 4 * 768, device=images.device)
        return (values.reshape(1, 4, 768).repeat(batch, 1, 1),)


class InferenceTests(unittest.TestCase):
    def test_forward_shapes(self):
        config = DetectorConfig(image_size=16, proposal_count=2, patch_size=8)
        model = AnomalyDetector(StubBackbone(), config=config).eval()
        result = model(torch.zeros(1, 3, 16, 16))
        self.assertEqual(tuple(result["patch_scores"].shape), (1, 4))
        self.assertEqual(tuple(result["anomaly_map"].shape), (1, 1, 16, 16))
        self.assertEqual(tuple(result["proposals"].shape), (1, 2, 2))

    def test_score_is_finite(self):
        values = torch.arange(12, dtype=torch.float32).reshape(1, 12)
        score = aggregate_image_score(values, "example")
        self.assertTrue(torch.isfinite(score).all())


if __name__ == "__main__":
    unittest.main()
