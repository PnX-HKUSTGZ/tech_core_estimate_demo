"""固定相机与物体变换的 Blender 合成数据管线。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .context import RenderContext
from .schema import PipelineConfig, SceneConfig

if TYPE_CHECKING:
    from .ops import Op


def build_ops(
    cfg: PipelineConfig,
) -> tuple[list[Op[object]], list[Op[object]], SceneConfig]:
    """Build Blender operations lazily so annotation tools also import in CPython."""
    from .ops import build_ops as build_blender_ops

    return build_blender_ops(cfg)


__all__ = ["RenderContext", "build_ops"]
