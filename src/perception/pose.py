"""Independent camera-from-station and camera-from-pillar PnP fits.

The seven station points and five pillar points are separate rigid templates.
Their default coordinates are in metres from the upstream system configuration;
the pillar transform therefore describes the current pillar, including slide.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from .types import BodyPose, FloatArray, KeypointDetection, PILLAR_NAMES, PoseResult, STATION_NAMES

_DEFAULT_PILLAR_POINTS: dict[str, tuple[float, float, float]] = {
    "TL": (-0.100000, -0.040000, 0.040000),
    "TR": (-0.100000, 0.040000, 0.040000),
    "BL": (-0.100000, -0.040000, -0.040000),
    "BR": (-0.100000, 0.040000, -0.040000),
    "ring": (0.000000, 0.000000, 0.000000),
}
_DEFAULT_STATION_POINTS: dict[str, tuple[float, float, float]] = {
    "light_BR": (-0.112000, 0.114870, 0.035953),
    "light_TR": (-0.112000, 0.114870, 0.167312),
    "shell_R": (-0.112000, 0.084871, 0.276449),
    "shell_M": (-0.112000, 0.000000, 0.325449),
    "shell_L": (-0.112000, -0.084871, 0.276449),
    "light_TL": (-0.112000, -0.114870, 0.167312),
    "light_BL": (-0.112000, -0.114870, 0.035953),
}

PointMap: TypeAlias = Mapping[str, Sequence[float] | FloatArray]
_IndexArray: TypeAlias = NDArray[np.intp]


def _finite_number(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _validated_geometry(
    provided: PointMap | None,
    defaults: Mapping[str, tuple[float, float, float]],
    names: tuple[str, ...],
    label: str,
) -> FloatArray:
    if provided is not None and not isinstance(provided, Mapping):
        raise ValueError(f"{label}_points must map exact point names to coordinates")
    points: PointMap = defaults if provided is None else provided
    if set(points) != set(names):
        raise ValueError(f"{label}_points must contain exactly {names}")
    try:
        geometry = np.stack([np.asarray(points[name], dtype=np.float64) for name in names])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}_points must contain finite three-dimensional metre coordinates") from exc
    if geometry.shape != (len(names), 3) or not np.isfinite(geometry).all():
        raise ValueError(f"{label}_points must contain finite three-dimensional metre coordinates")
    geometry.setflags(write=False)
    return geometry


def _validated_camera(
    camera_matrix: FloatArray,
    distortion: FloatArray | None,
) -> tuple[FloatArray, FloatArray]:
    try:
        camera = np.array(camera_matrix, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("K must be a finite 3x3 camera matrix") from exc
    if camera.shape != (3, 3) or not np.isfinite(camera).all():
        raise ValueError("K must be a finite 3x3 camera matrix")
    if (
        camera[0, 0] <= 0.0
        or camera[1, 1] <= 0.0
        or not np.allclose(camera[[0, 1], [1, 0]], 0.0, rtol=0.0, atol=1e-12)
        or not np.allclose(camera[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-12)
    ):
        raise ValueError("K needs positive focal lengths, zero skew, and bottom row [0, 0, 1]")
    if distortion is None:
        return camera, np.zeros(5, dtype=np.float64)
    try:
        coefficients = np.array(distortion, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("distortion must be a finite OpenCV coefficient vector") from exc
    if coefficients.ndim == 2 and 1 in coefficients.shape:
        coefficients = coefficients.reshape(-1)
    if coefficients.ndim != 1 or coefficients.size not in (4, 5, 8, 12, 14):
        raise ValueError("distortion must have 4, 5, 8, 12, or 14 coefficients")
    if not np.isfinite(coefficients).all():
        raise ValueError("distortion coefficients must be finite")
    return camera, coefficients


def _affine_rank(points: FloatArray, tolerance: float) -> int:
    if len(points) == 0:
        return 0
    with np.errstate(over="raise", invalid="raise"):
        return int(np.linalg.matrix_rank(points - points.mean(axis=0), tol=tolerance))


def _positive_depth(
    all_object_points: FloatArray,
    rotation_vector: FloatArray,
    translation_vector: FloatArray,
) -> bool:
    rotation = cv2.Rodrigues(rotation_vector)[0]
    depths = (all_object_points @ rotation.T + translation_vector.reshape(1, 3))[:, 2]
    return bool(np.isfinite(depths).all() and np.min(depths) > 1e-6)


def _reprojection_errors(
    object_points: FloatArray,
    image_points: FloatArray,
    camera: FloatArray,
    distortion: FloatArray,
    rotation_vector: FloatArray,
    translation_vector: FloatArray,
) -> FloatArray:
    projected = cv2.projectPoints(
        object_points, rotation_vector, translation_vector, camera, distortion
    )[0].reshape(-1, 2)
    errors = np.linalg.norm(projected - image_points, axis=1)
    if not np.isfinite(errors).all():
        raise ValueError("nonfinite_reprojection")
    return np.asarray(errors, dtype=np.float64)


class PoseEstimator:
    """Fit each named rigid body from one twelve-point detection.

    Heatmap peaks at or above ``min_keypoint_score`` are eligible. Reprojection
    acceptance uses the inlier mean; errors for every eligible point are kept
    in ``BodyPose`` even when RANSAC rejects an observation. Planar fits do not
    establish a unique physical pose.
    """

    def __init__(
        self,
        *,
        station_points: PointMap | None = None,
        pillar_points: PointMap | None = None,
        min_keypoint_score: float = 0.60,
        min_points: int = 4,
        use_ransac: bool = True,
        ransac_reprojection_error_px: float = 3.0,
        ransac_confidence: float = 0.99,
        ransac_iterations: int = 100,
        max_reprojection_error_px: float = 5.0,
    ) -> None:
        self.station_points = _validated_geometry(
            station_points, _DEFAULT_STATION_POINTS, STATION_NAMES, "station"
        )
        self.pillar_points = _validated_geometry(
            pillar_points, _DEFAULT_PILLAR_POINTS, PILLAR_NAMES, "pillar"
        )
        score = _finite_number(min_keypoint_score, "min_keypoint_score")
        confidence = _finite_number(ransac_confidence, "ransac_confidence")
        ransac_error = _finite_number(ransac_reprojection_error_px, "ransac_reprojection_error_px")
        max_error = _finite_number(max_reprojection_error_px, "max_reprojection_error_px")
        if not 0.0 <= score <= 1.0:
            raise ValueError("min_keypoint_score must be in [0, 1]")
        if not 0.0 < confidence < 1.0:
            raise ValueError("ransac_confidence must be strictly between 0 and 1")
        if ransac_error <= 0.0 or max_error <= 0.0:
            raise ValueError("reprojection thresholds must be positive")
        if isinstance(min_points, bool) or not isinstance(min_points, int) or not 4 <= min_points <= 5:
            raise ValueError("min_points must be 4 or 5 so both bodies can be fitted")
        if not isinstance(use_ransac, bool):
            raise ValueError("use_ransac must be boolean")
        if (
            isinstance(ransac_iterations, bool)
            or not isinstance(ransac_iterations, int)
            or ransac_iterations <= 0
        ):
            raise ValueError("ransac_iterations must be a positive integer")
        self.min_keypoint_score = score
        self.min_points = min_points
        self.use_ransac = use_ransac
        self.ransac_reprojection_error_px = ransac_error
        self.ransac_confidence = confidence
        self.ransac_iterations = ransac_iterations
        self.max_reprojection_error_px = max_error

    def estimate(
        self,
        detection: KeypointDetection,
        camera_matrix: FloatArray,
        distortion: FloatArray | None = None,
    ) -> PoseResult:
        """Return independent station and pillar camera transforms in metres.

        Invalid camera inputs raise ``ValueError`` before fitting either body;
        a numerical failure in one body is recorded only on that body.
        """
        camera, coefficients = _validated_camera(camera_matrix, distortion)
        station = self._fit_body(detection, STATION_NAMES, self.station_points, camera, coefficients)
        pillar = self._fit_body(detection, PILLAR_NAMES, self.pillar_points, camera, coefficients)
        return PoseResult(station=station, pillar=pillar)

    def _fit_body(
        self,
        detection: KeypointDetection,
        names: tuple[str, ...],
        geometry: FloatArray,
        camera: FloatArray,
        distortion: FloatArray,
    ) -> BodyPose:
        all_indices = [detection.keypoint_names.index(name) for name in names]
        selected = [
            index for index, detection_index in enumerate(all_indices)
            if detection.keypoint_scores[detection_index] >= self.min_keypoint_score
        ]
        used_names = tuple(names[index] for index in selected)
        object_points = geometry[selected]
        image_points = detection.keypoints_xy[[all_indices[index] for index in selected]]
        try:
            rank = _affine_rank(object_points, 1e-9)
        except (FloatingPointError, np.linalg.LinAlgError):
            return BodyPose(None, "degenerate_object_geometry", used_names=used_names)
        def failed(
            reason: str,
            *,
            inlier_names: tuple[str, ...] = (),
            inlier_mean: float | None = None,
            all_mean: float | None = None,
            errors: tuple[float, ...] = (),
            candidate_count: int = 0,
            fit_rank: int = rank,
        ) -> BodyPose:
            return BodyPose(
                transform=None,
                failure_reason=reason,
                used_names=used_names,
                inlier_names=inlier_names,
                inlier_mean_error_px=inlier_mean,
                all_points_mean_error_px=all_mean,
                reprojection_errors_px=errors,
                fit_rank=fit_rank,
                planar_geometry=fit_rank == 2,
                candidate_count=candidate_count,
            )

        if len(selected) < self.min_points:
            return failed(f"fewer_than_{self.min_points}_usable_points")
        if rank < 2:
            return failed("collinear_object_points")
        try:
            if _affine_rank(image_points, 1e-6) < 2:
                return failed("collinear_image_points")
        except (FloatingPointError, np.linalg.LinAlgError):
            return failed("degenerate_image_geometry")

        inliers: _IndexArray = np.arange(len(selected), dtype=np.intp)
        try:
            if self.use_ransac:
                selected_inliers = self._ransac_inliers(
                    object_points, image_points, camera, distortion, rank
                )
                if selected_inliers is None:
                    return failed("ransac_fewer_than_minimum_inliers")
                inliers = selected_inliers
            inlier_names = tuple(used_names[index] for index in inliers)
            fit_object = object_points[inliers]
            fit_image = image_points[inliers]
            fit_rank = _affine_rank(fit_object, 1e-9)
            if fit_rank < 2 or _affine_rank(fit_image, 1e-6) < 2:
                return failed("degenerate_inlier_geometry", inlier_names=inlier_names, fit_rank=fit_rank)
            ok, rotations, translations, *_ = cv2.solvePnPGeneric(
                fit_object, fit_image, camera, distortion, flags=cv2.SOLVEPNP_SQPNP
            )
            if not ok:
                return failed("sqpnp_failed", inlier_names=inlier_names, fit_rank=fit_rank)
            best: tuple[float, float, FloatArray, FloatArray, FloatArray] | None = None
            candidate_count = 0
            for raw_rotation, raw_translation in zip(rotations, translations, strict=True):
                try:
                    rotation_vector: FloatArray = np.asarray(raw_rotation, dtype=np.float64).reshape(3, 1)
                    translation_vector: FloatArray = np.asarray(raw_translation, dtype=np.float64).reshape(3, 1)
                    if not (np.isfinite(rotation_vector).all() and np.isfinite(translation_vector).all()):
                        continue
                    if not _positive_depth(geometry, rotation_vector, translation_vector):
                        continue
                    refined_rotation, refined_translation = cv2.solvePnPRefineLM(
                        fit_object, fit_image, camera, distortion, rotation_vector, translation_vector
                    )
                    rotation_vector = np.asarray(refined_rotation, dtype=np.float64)
                    translation_vector = np.asarray(refined_translation, dtype=np.float64)
                    if not (
                        np.isfinite(rotation_vector).all()
                        and np.isfinite(translation_vector).all()
                        and _positive_depth(geometry, rotation_vector, translation_vector)
                    ):
                        continue
                    all_errors = _reprojection_errors(
                        object_points, image_points, camera, distortion,
                        rotation_vector, translation_vector,
                    )
                    inlier_mean = float(all_errors[inliers].mean())
                    all_mean = float(all_errors.mean())
                    candidate_count += 1
                    if best is None or inlier_mean < best[0]:
                        best = (inlier_mean, all_mean, all_errors, rotation_vector, translation_vector)
                except (cv2.error, FloatingPointError, ValueError, np.linalg.LinAlgError):
                    continue
            if best is None:
                return failed(
                    "no_finite_positive_depth_candidate",
                    inlier_names=inlier_names,
                    fit_rank=fit_rank,
                )
            inlier_mean, all_mean, all_errors, rotation_vector, translation_vector = best
            error_values = tuple(float(error) for error in all_errors)
            if inlier_mean > self.max_reprojection_error_px:
                return failed(
                    "reprojection_threshold_exceeded",
                    inlier_names=inlier_names,
                    inlier_mean=inlier_mean,
                    all_mean=all_mean,
                    errors=error_values,
                    candidate_count=candidate_count,
                    fit_rank=fit_rank,
                )
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
            transform[:3, 3] = translation_vector.reshape(3)
            transform.setflags(write=False)
            return BodyPose(
                transform=transform,
                failure_reason=None,
                used_names=used_names,
                inlier_names=inlier_names,
                inlier_mean_error_px=inlier_mean,
                all_points_mean_error_px=all_mean,
                reprojection_errors_px=error_values,
                fit_rank=fit_rank,
                planar_geometry=fit_rank == 2,
                candidate_count=candidate_count,
            )
        except (cv2.error, FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            return failed(f"pnp_failed: {exc}")

    def _ransac_inliers(
        self,
        object_points: FloatArray,
        image_points: FloatArray,
        camera: FloatArray,
        distortion: FloatArray,
        rank: int,
    ) -> _IndexArray | None:
        """Select inliers, retaining the planar-only OpenCV AP3P fallback."""
        try:
            ok, _, _, selected_inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                camera,
                distortion,
                iterationsCount=self.ransac_iterations,
                reprojectionError=self.ransac_reprojection_error_px,
                confidence=self.ransac_confidence,
                flags=cv2.SOLVEPNP_SQPNP,
            )
        except cv2.error:
            if rank != 2:
                raise
            ok, selected_inliers = False, None
        if rank == 2 and (not ok or selected_inliers is None or len(selected_inliers) < self.min_points):
            # OpenCV's SQPnP RANSAC can reject even noiseless planar
            # seven-point sets. AP3P only chooses the inliers; the
            # accepted pose is still solved with SQPnP and refined LM.
            ok, _, _, selected_inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                camera,
                distortion,
                iterationsCount=self.ransac_iterations,
                reprojectionError=self.ransac_reprojection_error_px,
                confidence=self.ransac_confidence,
                flags=cv2.SOLVEPNP_AP3P,
            )
        if not ok or selected_inliers is None or len(selected_inliers) < self.min_points:
            return None
        return np.asarray(selected_inliers, dtype=np.intp).reshape(-1)
