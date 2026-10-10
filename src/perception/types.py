"""Numerical contracts shared by recognition and pose estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
ImageArray: TypeAlias = NDArray[np.uint8]

PILLAR_NAMES = ("TL", "TR", "BL", "BR", "ring")
STATION_NAMES = (
    "light_BR", "light_TR", "shell_R", "shell_M", "shell_L", "light_TL", "light_BL"
)
KEYPOINT_NAMES = PILLAR_NAMES + STATION_NAMES


@dataclass(frozen=True, slots=True)
class KeypointDetection:
    """One class-1 target; all coordinates refer to the original BGR image.

    ``keypoint_scores`` are raw heatmap peaks, not visibility probabilities.
    The twelve names have fixed object semantics, independent of screen order.
    """

    bbox_xyxy: FloatArray
    detection_score: float
    keypoints_xy: FloatArray
    keypoint_scores: FloatArray
    image_shape_hw: tuple[int, int]
    keypoint_names: tuple[str, ...] = KEYPOINT_NAMES

    def __post_init__(self) -> None:
        if tuple(self.keypoint_names) != KEYPOINT_NAMES:
            raise ValueError("keypoint_names must match the canonical twelve-point schema")
        if len(self.image_shape_hw) != 2 or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0
            for value in self.image_shape_hw
        ):
            raise ValueError("image_shape_hw must contain two positive integers")
        score = float(self.detection_score)
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("detection_score must be finite and in [0, 1]")
        for name, value, shape in (
            ("bbox_xyxy", self.bbox_xyxy, (4,)),
            ("keypoints_xy", self.keypoints_xy, (12, 2)),
            ("keypoint_scores", self.keypoint_scores, (12,)),
        ):
            array = np.array(value, dtype=np.float64, copy=True)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"{name} must be finite with shape {shape}")
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        x1, y1, x2, y2 = self.bbox_xyxy
        height, width = self.image_shape_hw
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("bbox_xyxy must be a nonempty box within the image")
        object.__setattr__(self, "detection_score", score)
        object.__setattr__(self, "image_shape_hw", (int(height), int(width)))
        object.__setattr__(self, "keypoint_names", KEYPOINT_NAMES)


@dataclass(frozen=True, slots=True)
class BodyPose:
    """Independent body fit. Failed fits never expose a usable transform.

    ``reprojection_errors_px`` aligns with ``used_names``. The acceptance gate
    uses ``inlier_mean_error_px``; ``all_points_mean_error_px`` also includes
    observations rejected by RANSAC. A planar fit does not establish uniqueness.
    """

    transform: FloatArray | None
    failure_reason: str | None
    used_names: tuple[str, ...] = ()
    inlier_names: tuple[str, ...] = ()
    inlier_mean_error_px: float | None = None
    all_points_mean_error_px: float | None = None
    reprojection_errors_px: tuple[float, ...] = ()
    fit_rank: int = 0
    planar_geometry: bool = False
    candidate_count: int = 0
    pose_uniqueness_verified: bool = False

    @property
    def success(self) -> bool:
        return self.transform is not None and self.failure_reason is None


@dataclass(frozen=True, slots=True)
class PoseResult:
    """Camera-from-body transforms in metres, using OpenCV right/down/forward."""

    station: BodyPose
    pillar: BodyPose

    @property
    def T_camera_from_station(self) -> FloatArray | None:
        return self.station.transform

    @property
    def T_camera_from_pillar(self) -> FloatArray | None:
        return self.pillar.transform
