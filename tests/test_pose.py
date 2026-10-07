"""Geometric and failure-contract tests for the independent rigid-body fits."""

from __future__ import annotations

import unittest

import cv2
import numpy as np
from numpy.typing import NDArray

from perception.pose import PoseEstimator
from perception.types import KEYPOINT_NAMES, KeypointDetection

FloatArray = NDArray[np.float64]

# Coordinates copied from the upstream system_config.example.yaml contract,
# not from PoseEstimator's private defaults.
_POINTS: dict[str, tuple[float, float, float]] = {
    "TL": (-0.100000, -0.040000, 0.040000),
    "TR": (-0.100000, 0.040000, 0.040000),
    "BL": (-0.100000, -0.040000, -0.040000),
    "BR": (-0.100000, 0.040000, -0.040000),
    "ring": (0.000000, 0.000000, 0.000000),
    "light_BR": (-0.112000, 0.114870, 0.035953),
    "light_TR": (-0.112000, 0.114870, 0.167312),
    "shell_R": (-0.112000, 0.084871, 0.276449),
    "shell_M": (-0.112000, 0.000000, 0.325449),
    "shell_L": (-0.112000, -0.084871, 0.276449),
    "light_TL": (-0.112000, -0.114870, 0.167312),
    "light_BL": (-0.112000, -0.114870, 0.035953),
}
_K = np.array([[820.0, 0.0, 320.0], [0.0, 810.0, 240.0], [0.0, 0.0, 1.0]])
_RVEC = np.array([[0.15], [-0.35], [0.04]], dtype=np.float64)
_STATION_T = np.array([[0.02], [-0.02], [1.50]], dtype=np.float64)
_SLIDE_M = 0.07
_PILLAR_T = _STATION_T + cv2.Rodrigues(_RVEC)[0] @ np.array(
    [[_SLIDE_M], [0.0], [0.0]], dtype=np.float64
)


def _detection(
    *,
    scores: FloatArray | None = None,
    coordinates: FloatArray | None = None,
    distortion: FloatArray | None = None,
) -> KeypointDetection:
    pillar = np.array([_POINTS[name] for name in KEYPOINT_NAMES[:5]], dtype=np.float64)
    station = np.array([_POINTS[name] for name in KEYPOINT_NAMES[5:]], dtype=np.float64)
    pillar_xy = cv2.projectPoints(pillar, _RVEC, _PILLAR_T, _K, distortion)[0].reshape(-1, 2)
    station_xy = cv2.projectPoints(station, _RVEC, _STATION_T, _K, distortion)[0].reshape(-1, 2)
    xy = np.vstack((pillar_xy, station_xy)) if coordinates is None else coordinates
    return KeypointDetection(
        bbox_xyxy=np.array([100.0, 40.0, 560.0, 430.0]),
        detection_score=0.95,
        keypoints_xy=xy,
        keypoint_scores=np.ones(12) if scores is None else scores,
        image_shape_hw=(480, 640),
    )


class PoseEstimatorTests(unittest.TestCase):
    def test_independent_station_and_sliding_pillar(self) -> None:
        result = PoseEstimator().estimate(_detection(), camera_matrix=_K)
        self.assertTrue(result.station.success, result.station.failure_reason)
        self.assertTrue(result.pillar.success, result.pillar.failure_reason)
        self.assertEqual(result.station.fit_rank, 2)
        self.assertTrue(result.station.planar_geometry)
        self.assertEqual(result.pillar.fit_rank, 3)
        self.assertFalse(result.pillar.planar_geometry)
        self.assertFalse(result.station.pose_uniqueness_verified)
        self.assertFalse(result.pillar.pose_uniqueness_verified)
        self.assertIsNotNone(result.station.inlier_mean_error_px)
        self.assertIsNotNone(result.pillar.inlier_mean_error_px)
        assert result.station.inlier_mean_error_px is not None
        assert result.pillar.inlier_mean_error_px is not None
        self.assertLess(result.station.inlier_mean_error_px, 1e-4)
        self.assertLess(result.pillar.inlier_mean_error_px, 1e-4)
        self.assertIsNotNone(result.T_camera_from_station)
        self.assertIsNotNone(result.T_camera_from_pillar)
        assert result.pillar.transform is not None
        np.testing.assert_allclose(result.pillar.transform[:3, 3], _PILLAR_T.reshape(3), atol=2e-3)
        self.assertFalse(result.pillar.transform.flags.writeable)

    def test_moving_and_rotating_pillar_preserves_station_pose(self) -> None:
        estimator = PoseEstimator()
        original = estimator.estimate(_detection(), _K)
        pillar_points = np.array([_POINTS[name] for name in KEYPOINT_NAMES[:5]], dtype=np.float64)
        rotation_vector = np.array([[-0.25], [0.20], [0.18]], dtype=np.float64)
        translation_vector = np.array([[0.08], [0.03], [1.35]], dtype=np.float64)
        xy = _detection().keypoints_xy.copy()
        xy[:5] = cv2.projectPoints(
            pillar_points, rotation_vector, translation_vector, _K, None
        )[0].reshape(-1, 2)
        moved = estimator.estimate(_detection(coordinates=xy), _K)
        for result in (original, moved):
            self.assertTrue(result.station.success, result.station.failure_reason)
            self.assertTrue(result.pillar.success, result.pillar.failure_reason)
        assert original.station.transform is not None
        assert moved.station.transform is not None
        assert moved.pillar.transform is not None
        expected_station = np.eye(4, dtype=np.float64)
        expected_station[:3, :3] = cv2.Rodrigues(_RVEC)[0]
        expected_station[:3, 3] = _STATION_T.reshape(3)
        expected_pillar = np.eye(4, dtype=np.float64)
        expected_pillar[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
        expected_pillar[:3, 3] = translation_vector.reshape(3)
        np.testing.assert_allclose(moved.station.transform, original.station.transform, atol=1e-9)
        np.testing.assert_allclose(moved.station.transform, expected_station, atol=1e-5)
        np.testing.assert_allclose(moved.pillar.transform, expected_pillar, atol=1e-5)

    def test_one_body_below_score_threshold_does_not_clear_the_other(self) -> None:
        scores = np.ones(12, dtype=np.float64)
        scores[5:] = 0.59
        scores[5:8] = 0.60
        result = PoseEstimator().estimate(_detection(scores=scores), _K)
        self.assertFalse(result.station.success)
        self.assertEqual(result.station.failure_reason, "fewer_than_4_usable_points")
        self.assertEqual(len(result.station.used_names), 3)
        self.assertTrue(result.pillar.success, result.pillar.failure_reason)
        self.assertIsNone(result.T_camera_from_station)
        self.assertIsNotNone(result.T_camera_from_pillar)

    def test_threshold_is_inclusive_for_four_station_points(self) -> None:
        scores = np.ones(12, dtype=np.float64)
        scores[5:] = 0.59
        scores[5:9] = 0.60
        result = PoseEstimator().estimate(_detection(scores=scores), _K)
        self.assertEqual(len(result.station.used_names), 4)
        self.assertTrue(result.station.success, result.station.failure_reason)
        self.assertTrue(result.station.planar_geometry)

    def test_missing_ring_is_reported_as_planar_pillar_fit(self) -> None:
        scores = np.ones(12, dtype=np.float64)
        scores[4] = 0.59
        result = PoseEstimator().estimate(_detection(scores=scores), _K)
        self.assertTrue(result.pillar.success, result.pillar.failure_reason)
        self.assertEqual(result.pillar.used_names, KEYPOINT_NAMES[:4])
        self.assertEqual(result.pillar.fit_rank, 2)
        self.assertTrue(result.pillar.planar_geometry)
        self.assertFalse(result.pillar.pose_uniqueness_verified)
        self.assertTrue(result.station.success, result.station.failure_reason)

    def test_four_point_pillar_with_ring_retains_rank_three(self) -> None:
        scores = np.ones(12, dtype=np.float64)
        scores[3] = 0.59
        result = PoseEstimator().estimate(_detection(scores=scores), _K)
        self.assertTrue(result.pillar.success, result.pillar.failure_reason)
        self.assertEqual(result.pillar.used_names, ("TL", "TR", "BL", "ring"))
        self.assertEqual(result.pillar.fit_rank, 3)
        self.assertFalse(result.pillar.planar_geometry)
        self.assertTrue(result.station.success, result.station.failure_reason)

    def test_ransac_rejects_station_outlier_and_reports_all_residuals(self) -> None:
        xy = _detection().keypoints_xy.copy()
        xy[8] += (65.0, -45.0)
        result = PoseEstimator().estimate(_detection(coordinates=xy), _K)
        self.assertTrue(result.station.success, result.station.failure_reason)
        self.assertTrue(result.pillar.success, result.pillar.failure_reason)
        self.assertNotIn("shell_M", result.station.inlier_names)
        self.assertEqual(len(result.station.reprojection_errors_px), 7)
        self.assertGreater(result.station.reprojection_errors_px[3], 50.0)
        assert result.station.inlier_mean_error_px is not None
        self.assertLess(result.station.inlier_mean_error_px, 1.0)
        self.assertGreater(result.station.all_points_mean_error_px or 0.0, 5.0)

    def test_reprojection_gate_fails_only_station_without_ransac(self) -> None:
        xy = _detection().keypoints_xy.copy()
        xy[8] += (65.0, -45.0)
        result = PoseEstimator(use_ransac=False).estimate(_detection(coordinates=xy), _K)
        self.assertFalse(result.station.success)
        self.assertEqual(result.station.failure_reason, "reprojection_threshold_exceeded")
        self.assertIsNone(result.station.transform)
        self.assertEqual(len(result.station.reprojection_errors_px), 7)
        self.assertGreater(result.station.inlier_mean_error_px or 0.0, 5.0)
        self.assertTrue(result.pillar.success, result.pillar.failure_reason)

    def test_degenerate_image_and_object_fail_independently(self) -> None:
        xy = _detection().keypoints_xy.copy()
        xy[5:, 0] = 100.0 + np.arange(7, dtype=np.float64) * 10.0
        xy[5:, 1] = 200.0 + np.arange(7, dtype=np.float64) * 5.0
        image_result = PoseEstimator().estimate(_detection(coordinates=xy), _K)
        self.assertEqual(image_result.station.failure_reason, "collinear_image_points")
        self.assertTrue(image_result.pillar.success, image_result.pillar.failure_reason)

        station_names = KEYPOINT_NAMES[5:]
        line = {name: (float(index) * 0.01, 0.0, 0.0) for index, name in enumerate(station_names)}
        object_result = PoseEstimator(station_points=line).estimate(_detection(), _K)
        self.assertEqual(object_result.station.failure_reason, "collinear_object_points")
        self.assertEqual(object_result.station.fit_rank, 1)
        self.assertTrue(object_result.pillar.success, object_result.pillar.failure_reason)

        huge = {name: (1e308, 1e308, 1e308) for name in station_names}
        overflow_result = PoseEstimator(station_points=huge).estimate(_detection(), _K)
        self.assertEqual(overflow_result.station.failure_reason, "degenerate_object_geometry")
        self.assertTrue(overflow_result.pillar.success, overflow_result.pillar.failure_reason)

    def test_distortion_vector_supported(self) -> None:
        distortion = np.array([0.01, -0.02, 0.001, -0.0005, 0.002], dtype=np.float64)
        result = PoseEstimator().estimate(_detection(distortion=distortion), _K, distortion)
        self.assertTrue(result.station.success, result.station.failure_reason)
        self.assertTrue(result.pillar.success, result.pillar.failure_reason)
        for size in (4, 8, 12, 14):
            with self.subTest(size=size):
                zero_distortion = np.zeros(size, dtype=np.float64)
                zero_result = PoseEstimator().estimate(_detection(), _K, zero_distortion)
                self.assertTrue(zero_result.station.success, zero_result.station.failure_reason)
                self.assertTrue(zero_result.pillar.success, zero_result.pillar.failure_reason)

    def test_bad_camera_or_distortion_raises(self) -> None:
        estimator = PoseEstimator()
        detection = _detection()
        bad_matrices = (
            np.eye(2),
            np.array([[0.0, 0.0, 320.0], [0.0, 810.0, 240.0], [0.0, 0.0, 1.0]]),
            np.array([[820.0, 2.0, 320.0], [0.0, 810.0, 240.0], [0.0, 0.0, 1.0]]),
            np.array([[820.0, 0.0, 320.0], [0.0, 810.0, 240.0], [0.0, 1.0, 1.0]]),
            np.array([[np.nan, 0.0, 320.0], [0.0, 810.0, 240.0], [0.0, 0.0, 1.0]]),
        )
        for bad in bad_matrices:
            with self.subTest(K=bad), self.assertRaises(ValueError):
                estimator.estimate(detection, bad)
        for bad in (np.zeros(3), np.full(5, np.nan), np.zeros((2, 2))):
            with self.subTest(distortion=bad), self.assertRaises(ValueError):
                estimator.estimate(detection, _K, bad)

    def test_bad_geometry_and_solver_settings_raise(self) -> None:
        station = {name: _POINTS[name] for name in KEYPOINT_NAMES[5:]}
        for replacement in (
            {name: point for name, point in station.items() if name != "shell_M"},
            {**station, "extra": (0.0, 0.0, 0.0)},
            {**station, "shell_M": (np.inf, 0.0, 0.0)},
            {**station, "shell_M": (0.0, 0.0)},
        ):
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                PoseEstimator(station_points=replacement)
        for options in (
            {"min_keypoint_score": float("nan")},
            {"min_points": 3},
            {"ransac_confidence": 1.0},
            {"ransac_iterations": 0},
            {"ransac_reprojection_error_px": 0.0},
            {"max_reprojection_error_px": -1.0},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                PoseEstimator(**options)


if __name__ == "__main__":
    unittest.main()
