from __future__ import annotations

import unittest

import numpy as np

from agents.adapters import AgentPrediction, augment_fusion_training, fuse_predictions, fusion_matrix
from agents.runtime import risk_level


class RuntimeTests(unittest.TestCase):
    def test_risk_levels(self) -> None:
        self.assertEqual(risk_level(0.49), "low")
        self.assertEqual(risk_level(0.5), "medium")
        self.assertEqual(risk_level(0.75), "high")

    def test_confidence_weighted_fusion_prefers_covered_agent(self) -> None:
        score = fuse_predictions(
            {
                "covered": AgentPrediction(0.8, 0.9, 1.0, "test", []),
                "missing": AgentPrediction(0.1, 0.8, 0.05, "test", []),
            }
        )
        self.assertGreater(score, 0.7)

    def test_learned_fusion_uses_six_features(self) -> None:
        matrix = fusion_matrix(
            np.asarray([0.2, 0.8]),
            np.asarray([0.3, 0.7]),
            np.ones(2),
            np.ones(2),
        )
        self.assertEqual(matrix.shape, (2, 6))
        augmented, labels = augment_fusion_training(matrix, np.asarray([0, 1]))
        self.assertEqual(augmented.shape, (6, 6))
        self.assertEqual(labels.tolist(), [0, 1, 0, 1, 0, 1])


if __name__ == "__main__":
    unittest.main()
