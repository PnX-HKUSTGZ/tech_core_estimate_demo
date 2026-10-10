"""Shared data contracts for rendering, composition and annotation inspection.

Arrays carry their actual storage dtype. Pixel coordinates remain lists for
compatibility with existing JSON files; Blender-only types stay in the runtime
modules so image composition can also run in ordinary CPython.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
FloatImage: TypeAlias = NDArray[np.float32]
ImageArray: TypeAlias = NDArray[np.uint8 | np.uint16 | np.float32]
# OpenCV IMREAD_UNCHANGED preserves 8/16-bit images and float32 HDR inputs.
MaskArray: TypeAlias = NDArray[np.bool_]
Points2D: TypeAlias = list[list[float]]
ProjectedKeypoints: TypeAlias = list[list[float]]  # u, v, visibility (0/1/2)
Metadata: TypeAlias = dict[str, object]
Color: TypeAlias = tuple[int, int, int]


class TargetAnnotation(TypedDict, total=False):
    bbox_2d: list[float] | None
    keypoints_2d: ProjectedKeypoints
    keypoint_names: list[str]


class ImageAnnotation(TypedDict, total=False):
    id: int
    file_name: str
    width: int
    height: int
    keypoint_names: list[str]
    keypoints_2d: ProjectedKeypoints
    keypoints_3d: list[list[float]]
    targets: dict[str, TargetAnnotation]
    bbox_2d: list[float] | None
    bbox_2d_exchange: list[float] | None
    camera_K: list[list[float]]
    camera_R: list[list[float]]
    camera_t: list[float]
    camera_pos_world: list[float]
    meta: Metadata


class TargetMetadata(TypedDict, total=False):
    name: str
    pass_index: int
    keypoint_names: list[str]
    keypoints_3d: list[list[float]]


class _OptionalDatasetAnnotations(TypedDict, total=False):
    targets: list[TargetMetadata] | dict[str, TargetMetadata]
    keypoint_3d_world: list[list[float]]


class DatasetAnnotations(_OptionalDatasetAnnotations):
    keypoint_names: list[str]
    images: list[ImageAnnotation]


class CropGeometry(TypedDict, total=False):
    kps_2d: Points2D
    kps_vis: list[int]
    bbox_2d: list[float] | None
    exc_kps_2d: Points2D
    exc_kps_vis: list[int]
    exc_bbox_2d: list[float] | None
    bbox_2d_exchange: list[float] | None


class Crop(CropGeometry):
    rgba: ImageArray
    file_name: str
    material_luma: float


class _OptionalPlacedInstance(TypedDict, total=False):
    exc_kps_2d: Points2D
    exc_kps_vis: list[int]
    exc_bbox: list[float]


class PlacedInstance(_OptionalPlacedInstance):
    kps_2d: Points2D
    kps_vis: list[int]
    bbox: list[float]
    material_luma: float
    file_name: str


class PlacedMask(TypedDict):
    mask: MaskArray
    offset: tuple[int, int]


class Background(TypedDict):
    path: Path
    luma: float


class FrameParameters(TypedDict):
    frame_idx: int
    selected: list[Crop]
    split: str
    worker_seed: int


class _OptionalRenderConfig(TypedDict, total=False):
    device: str
    engine: str
    use_denoising: bool
    sensor_width_mm: float


class RenderConfig(_OptionalRenderConfig):
    width: int
    height: int
    samples: int


class CameraConfig(TypedDict):
    position: list[float]
    look_at: list[float]
    fov_h_deg: float


class SceneConfig(TypedDict, total=False):
    sun_object: str
    material_map: dict[str, str]
    emission_node: str
    light_objects: list[str]


class StripLightConfig(TypedDict, total=False):
    objects: list[str]
    color: str
    color_choices: list[str]
    strength: float
    strength_range: list[float]
    on_prob: float
    material_map: dict[str, str]
    emission_node: str


class ArcLightConfig(TypedDict, total=False):
    objects: list[str]
    strength: float
    color: str
    color_choices: list[str]
    off: bool
    blink: bool
    on_prob: float
    per_object: bool
    min_on_count: int
    max_on_count: int
    base_material: str
    off_material: str
    emission_node: str


class MainLightConfig(TypedDict, total=False):
    objects: list[str]
    area_lights: list[str]
    energy: float
    diffuse_node: str
    base_material: str


class AuxLightConfig(TypedDict, total=False):
    count_range: list[int]
    dist: float
    energy_range: list[float]


class GlareConfig(TypedDict, total=False):
    glare_type: str
    threshold: float
    size: float
    size_range: list[float]
    mix: float


class LightingConfig(TypedDict, total=False):
    light_type: str
    light_type_choices: list[str]
    strips_strength: float
    ambient_color: list[float]
    ambient_strength: float
    sun_energy: float
    strip_lights: StripLightConfig
    arc_lights: ArcLightConfig
    main_lights: MainLightConfig
    aux_lights: AuxLightConfig
    glare: GlareConfig


class SlideConfig(TypedDict, total=False):
    objects: list[str]
    target: str
    skip_prob: float
    y_range: list[float]


class _OptionalTransformConfig(TypedDict, total=False):
    max_retries: int
    framing_margin: int
    pillar_slide: SlideConfig


class TransformConfig(_OptionalTransformConfig):
    pivot: list[float]
    yaw_range: list[float]
    pitch_range: list[float]
    roll_range: list[float]
    distance_range: list[float]
    offset_x_range: list[float]
    offset_y_range: list[float]


class MeshSelection(TypedDict, total=False):
    include: list[str]
    exclude: list[str]


class _OptionalTargetConfig(TypedDict, total=False):
    keypoints: dict[str, list[float]]
    mesh_objects: list[str] | MeshSelection
    bbox_from_targets: list[str]
    min_visible: int


class TargetConfig(_OptionalTargetConfig):
    name: str
    pass_index: int


class _OptionalPipelineConfig(TypedDict, total=False):
    scene: SceneConfig
    targets: list[TargetConfig]
    lighting: LightingConfig
    min_visible: dict[str, int]
    occlusion_offset_m: float


class PipelineConfig(_OptionalPipelineConfig):
    render: RenderConfig
    camera: CameraConfig
    object_transform: TransformConfig
