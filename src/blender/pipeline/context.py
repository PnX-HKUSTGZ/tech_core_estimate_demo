"""RenderContext — 帧级数据容器 (固定相机 + 物体变换模式)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .schema import FloatArray, Metadata, PipelineConfig, ProjectedKeypoints

if TYPE_CHECKING:
    import bpy


class RenderContext:
    """每一帧渲染过程中在各 Op 之间传递的共享状态."""

    __slots__ = (
        "scene",
        "cam_obj",
        "cam_data",
        "cfg",
        "light_type",
        "frame_idx",
        "img_path",
        "render_w",
        "render_h",
        # 相机参数 (FixedCameraOp 一次性写入)
        "K",
        "R_w2c",
        "t_w2c",
        "cam_pos",
        # 关键点 & bbox
        "kps_2d",
        "bbox_2d",
        "crop_w",
        "crop_h",
        # 物体变换参数 (ObjectTransformOp 每帧写入)
        "yaw_deg",
        "pitch_deg",
        "roll_deg",
        "distance_m",
        "offset_x_m",
        "offset_y_m",
        # 当前帧变换后的关键点 3D 坐标 (ObjectTransformOp 写入)
        "kp_3d_frame",
        "meta",
    )

    def __init__(
        self,
        scene: bpy.types.Scene,
        cam_obj: bpy.types.Object,
        cam_data: bpy.types.Camera,
        cfg: PipelineConfig,
        light_type: str = "red",
    ) -> None:
        self.scene = scene
        self.cam_obj = cam_obj
        self.cam_data = cam_data
        self.cfg = cfg
        self.light_type = light_type

        self.render_w: int = cfg["render"]["width"]
        self.render_h: int = cfg["render"]["height"]

        self.frame_idx: int = 0
        self.img_path: str = ""

        self.K: FloatArray | None = None
        self.R_w2c: FloatArray | None = None
        self.t_w2c: FloatArray | None = None
        self.cam_pos: FloatArray | None = None

        self.kps_2d: ProjectedKeypoints | None = None

        self.bbox_2d: list[float] | None = None
        self.crop_w: int = 0
        self.crop_h: int = 0

        self.yaw_deg: float = 0.0
        self.pitch_deg: float = 0.0
        self.roll_deg: float = 0.0
        self.distance_m: float = 0.0
        self.offset_x_m: float = 0.0
        self.offset_y_m: float = 0.0

        self.kp_3d_frame: dict[str, list[FloatArray]] | None = None

        self.meta: Metadata = {}
