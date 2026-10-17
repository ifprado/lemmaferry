"""Checks of known synthetic ground truth, controls, and held-out hygiene."""

import unittest

import numpy as np

from lemmaferry.alignment import (
    AffineRidge,
    OrthogonalProcrustes,
    Standardizer,
    regression_metrics,
    symmetric_cloud_ambiguity,
    synthetic_alignment_experiment,
)


class AlignmentTests(unittest.TestCase):
    def test_known_affine_map_recovers_heldout_different_dimensions(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(100, 5))
        w = rng.normal(size=(3, 5))
        b = np.array([4.0, -2.0, 1.5])
        y = x @ w.T + b
        model = AffineRidge(alpha=0).fit(x[:60], y[:60])
        np.testing.assert_allclose(model.predict(x[60:]), y[60:], atol=1e-12)
        self.assertEqual(model.coef_.shape, (3, 5))

    def test_correct_anchors_beat_wrong_anchors_and_mean(self):
        report, _ = synthetic_alignment_experiment(seed=7)
        metrics = report["metrics"]
        self.assertLess(metrics["paired_affine"]["normalized_rmse"], 0.03)
        self.assertGreater(metrics["wrong_pairs"]["normalized_rmse"], 0.7)
        self.assertGreater(metrics["training_mean"]["normalized_rmse"], 0.7)
        self.assertGreater(metrics["paired_affine"]["receiver_task_accuracy"], 0.95)
        self.assertLess(metrics["wrong_pairs"]["receiver_task_accuracy"], 0.7)

    def test_test_distribution_never_changes_training_preprocessing(self):
        x = np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]])
        y = np.column_stack((x[:, 0] * 3 + 10, x[:, 1] * 2 - 5))
        model = AffineRidge(alpha=0).fit(x, y)
        source_mean = model.source_scaler_.mean.copy()
        target_scale = model.target_scaler_.scale.copy()
        extreme_test = np.array([[1e6, -1e6]])
        model.predict(extreme_test)
        np.testing.assert_array_equal(model.source_scaler_.mean, source_mean)
        np.testing.assert_array_equal(model.target_scaler_.scale, target_scale)
        np.testing.assert_allclose(source_mean, x.mean(axis=0))
        test_target = np.array([[100.0, 1000.0]])
        prediction = test_target + target_scale
        metrics = regression_metrics(prediction, test_target, model.target_scaler_)
        self.assertAlmostEqual(metrics["normalized_rmse"], 1.0)

    def test_unpaired_geometry_and_cycles_can_hide_wrong_semantics(self):
        report = symmetric_cloud_ambiguity()
        for candidate in ("identity", "half_turn"):
            self.assertLess(report[candidate]["unpaired_nearest_neighbor_rmse"], 1e-14)
            self.assertEqual(report[candidate]["round_trip_rmse"], 0.0)
        self.assertEqual(report["identity"]["semantic_label_accuracy"], 1.0)
        self.assertEqual(report["half_turn"]["semantic_label_accuracy"], 0.0)
        self.assertGreater(report["half_turn"]["paired_rmse"], 0.9)

    def test_orthogonal_procrustes_recovers_rigid_map(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(100, 4))
        q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        y = x @ q + 7
        model = OrthogonalProcrustes().fit(x[:60], y[:60])
        np.testing.assert_allclose(model.predict(x[60:]), y[60:], atol=1e-12)

    def test_invalid_arrays_and_unfitted_prediction_fail_clearly(self):
        with self.assertRaises(RuntimeError):
            AffineRidge().predict([[1, 2]])
        with self.assertRaises(ValueError):
            Standardizer.fit([[np.nan]])
        with self.assertRaises(ValueError):
            AffineRidge().fit([[1], [2]], [[1]])
        scaler = Standardizer.fit([[2.0], [2.0]])
        np.testing.assert_array_equal(scaler.scale, [1.0])


if __name__ == "__main__":
    unittest.main()
