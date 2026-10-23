"""Synthetic checks for optional reporting utilities, not research reanalysis."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.cluster_statistics import benjamini_hochberg, pairwise_summary, participant_sign_flip
from tools.evaluate_frozen_predictions import summarize


class ReportingTests(unittest.TestCase):
    def test_bh_adjustment(self):
        self.assertEqual(benjamini_hochberg([0.01, 0.03, 0.04]), [0.03, 0.04, 0.04])

    def test_cluster_unit_is_person_not_session(self):
        result = participant_sign_flip(
            ["a", "a", "b", "b"], [0, 0, 0, 0], [1, 1, 1, 1], repetitions=19
        )
        self.assertEqual(result["n_participants"], 2)
        self.assertEqual(result["n_sessions"], 4)

    def test_invalid_cluster_input_rejected(self):
        with self.assertRaises(ValueError):
            participant_sign_flip(["a"], [0], [1])

    def test_di_worst_is_not_best_pair(self):
        rates = {str(i): {"tpr": 0.8, "fpr": 0.1, "ppp": p} for i, p in enumerate([0.1, 0.49, 0.5])}
        result = pairwise_summary(rates)
        self.assertAlmostEqual(result["di_worst_pair"], 0.2)
        self.assertAlmostEqual(result["di_best_pair"], 0.98)

    @staticmethod
    def document(person="external:a", threshold=0.24):
        return {
            "threshold_source": "validation",
            "threshold": threshold,
            "artifact_sha256": "synthetic",
            "predictions": [
                {"participant": person, "session": "s", "ic": 1, "gold_noise": 1, "p_noise": 0.8}
            ],
        }

    def test_dataset_and_pooled_metrics(self):
        result = summarize([("MSC", self.document("MSC:a")), ("HCP", self.document("HCP:b"))])
        self.assertEqual(result["pooled"]["n"], 2)
        self.assertEqual(result["pooled"]["f1"], 1)

    def test_pooling_different_thresholds_rejected(self):
        with self.assertRaises(ValueError):
            summarize([("MSC", self.document("MSC:a")), ("HCP", self.document("HCP:b", 0.28))])

    def test_pooling_duplicate_component_rejected(self):
        with self.assertRaises(ValueError):
            summarize([("one", self.document()), ("two", self.document())])


if __name__ == "__main__":
    unittest.main()
