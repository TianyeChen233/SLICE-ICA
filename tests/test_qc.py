"""Numerical and mocked-file regression tests; no research experiments."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.qc import compute_fc_figure_pack_v2_parallel as fc
from analysis.qc.summary_statistics import (
    adjust_bh,
    network_block_means,
    paired_wilcoxon,
    session_network_summaries,
)
from analysis.qc.taskA_qc_before_after import compute_tsnr, paired_stats
from analysis.qc.taskA_tsnr_recalc import compute_tsnr_variants


class StatisticsTests(unittest.TestCase):
    def test_wilcoxon_matches_scipy_and_is_not_a_t_statistic(self):
        before = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        after = before + [0.1, -0.2, 0.3, 0.6, 0.7, 0.9]
        expected = wilcoxon(after - before, alternative="two-sided", method="approx")
        result = paired_wilcoxon(before, after)
        self.assertAlmostEqual(result["p_value"], expected.pvalue)
        self.assertAlmostEqual(result["wilcoxon_stat"], expected.statistic)
        self.assertNotIn("t_stat", result)
        self.assertEqual(result["analysis_unit"], "session")

    def test_all_zero_differences_are_nonsignificant(self):
        result = paired_wilcoxon([1, 2, 3], [1, 2, 3])
        self.assertEqual((result["wilcoxon_stat"], result["p_value"]), (0, 1))

    def test_missing_pairs_not_independently_dropped(self):
        result = paired_wilcoxon([1, np.nan, 3, 4], [2, 9, np.nan, 7])
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["n_missing_pairs"], 2)
        self.assertTrue(np.isnan(result["p_value"]))
        with self.assertRaises(ValueError):
            paired_wilcoxon([1, 2], [1])

    def test_bh_preserves_undefined_tests(self):
        np.testing.assert_allclose(
            adjust_bh([0.01, np.nan, 0.03, 0.04]), [0.03, np.nan, 0.04, 0.04], equal_nan=True
        )
        with self.assertRaises(ValueError):
            adjust_bh([1.1])

    def test_qc_wrapper_uses_wilcoxon_and_complete_pairs(self):
        result = paired_stats(
            pd.DataFrame({"b": [1, 2, 3, np.nan], "a": [2, 4, 6, 9]}), "b", "a", "demo"
        )
        self.assertEqual(result["test"], "wilcoxon_two_sided")
        self.assertEqual(result["n_missing_pairs"], 1)

    def test_tsnr_matches_absolute_mean_equation(self):
        volume = np.array([-4.0, -2.0, -3.0, -5.0]).reshape(1, 1, 1, 4)
        mask = np.ones((1, 1, 1), dtype=bool)
        self.assertGreater(compute_tsnr(volume, mask), 0)
        self.assertAlmostEqual(
            compute_tsnr(volume, mask), compute_tsnr_variants(volume, mask)["tSNR_absmean_mean"]
        )

    def test_within_network_excludes_self_correlations(self):
        matrix = np.array(
            [[1, 0.2, 0.1, 0.1], [0.2, 1, 0.1, 0.1], [0.1, 0.1, 1, 0.4], [0.1, 0.1, 0.4, 1]]
        )
        actual = network_block_means(matrix, np.array([0, 0, 1, 1]), 2)
        np.testing.assert_allclose(actual, [[0.2, 0.1], [0.1, 0.4]])
        matrix[0, 1] = np.nan
        with self.assertRaises(ValueError):
            network_block_means(matrix, np.array([0, 0, 1, 1]), 2)

    def test_mean_of_session_ratios_differs_from_ratio_of_group_means(self):
        actual = session_network_summaries([[[0.4, 0.2], [0.2, 0.4]], [[0.8, 0.1], [0.1, 0.8]]])
        np.testing.assert_allclose(actual["within_between_ratio"], [2, 8])
        self.assertNotAlmostEqual(
            actual["within_between_ratio"].mean(),
            actual["within"].mean() / actual["between"].mean(),
        )
        actual = session_network_summaries([[[0.4, 0], [0, 0.4]]])
        self.assertTrue(np.isnan(actual["within_between_ratio"][0]))


class FCWorkerTests(unittest.TestCase):
    def test_failed_session_never_enters_merged_statistics_as_zeros(self):
        with tempfile.TemporaryDirectory(prefix="slice_ica_synthetic_") as directory:
            out = Path(directory)
            sessions = [out / "synthetic_person" / f"rfMRI_{i}" for i in range(4)]
            atlas = np.arange(1, 5).reshape(2, 2, 1)
            roi_ids = np.arange(1, 5)
            labels = [
                "7Networks_LH_A_1",
                "7Networks_RH_A_2",
                "7Networks_LH_B_1",
                "7Networks_RH_B_2",
            ]
            tri_i, tri_j = np.triu_indices(4, k=1)
            prepared = (
                atlas,
                sessions,
                roi_ids,
                labels,
                ["A", "A", "B", "B"],
                ["A", "B"],
                np.array([0, 0, 1, 1]),
                [np.array([i]) for i in range(4)],
                np.zeros((4, 3)),
                tri_i,
                tri_j,
            )
            rng = np.random.default_rng(42)
            before, after = rng.normal(size=(2, 2, 1, 20)), rng.normal(size=(2, 2, 1, 20))

            def fake_load(path):
                path = Path(path)
                if path.parent.name == "rfMRI_1":
                    raise FileNotFoundError("Intentionally missing synthetic input")
                if path.name == "mask.nii.gz":
                    values = np.ones((2, 2, 1))
                elif path.name == "filtered_func_data_regress.nii.gz":
                    values = after
                elif path.name == "filtered_func_data.nii.gz":
                    values = before
                else:
                    values = atlas.astype(float)
                return nib.Nifti1Image(values, np.eye(4))

            with (
                patch.object(fc, "prep_common", return_value=prepared),
                patch.object(fc.nib, "load", side_effect=fake_load),
            ):
                fc.worker(out, 1, 1)
                with np.load(out / "worker1_arrays.npz", allow_pickle=False) as arrays:
                    self.assertEqual(arrays["edge_before"].shape, (3, 6))
                    self.assertFalse(np.any(np.all(arrays["edge_before"] == 0, axis=1)))
                fc.merge(out, 1)
                with self.assertRaises(FileExistsError):
                    fc.worker(out, 1, 1)
            summary = json.loads((out / "fc_figure_pack_summary.json").read_text())
            self.assertEqual(summary["n_sessions_merged"], 3)
            self.assertEqual(summary["paired_test"], "wilcoxon_two_sided")
            rows = pd.read_csv(out / "session_network_summaries.csv")
            self.assertNotIn("rfMRI_1", rows["session"].tolist())
            statistics = pd.read_csv(out / "network_summary_paired_stats.csv")
            self.assertEqual(set(statistics["analysis_unit"]), {"session"})


if __name__ == "__main__":
    unittest.main()
