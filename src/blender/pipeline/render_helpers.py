"""Scene object selection and segmentation output helpers for the renderer."""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .schema import MeshSelection, TargetAnnotation, TargetConfig

if TYPE_CHECKING:
    import bpy


def _resolve_mesh_objects(
    mesh_cfg: list[str] | MeshSelection, all_objects: Sequence[bpy.types.Object]
) -> set[bpy.types.Object]:
    if isinstance(mesh_cfg, dict):
        includes = mesh_cfg.get("include", [])
        excludes = mesh_cfg.get("exclude", [])
    else:
        includes = mesh_cfg if mesh_cfg else []
        excludes = []
    matched = set()
    for pat in includes:
        for obj in all_objects:
            if obj.type == "MESH" and fnmatch.fnmatch(obj.name, pat):
                matched.add(obj)
    for pat in excludes:
        for obj in list(matched):
            if fnmatch.fnmatch(obj.name, pat):
                matched.discard(obj)
    return matched


def _add_target_mask_output(
    tree: bpy.types.NodeTree,
    rl_node: bpy.types.Node,
    pass_index_val: int,
    output_dir: str,
    slot_prefix: str,
) -> bpy.types.CompositorNodeOutputFile:
    gt = tree.nodes.new("CompositorNodeMath")
    gt.operation = "GREATER_THAN"
    gt.inputs[1].default_value = pass_index_val - 0.5
    tree.links.new(rl_node.outputs["IndexOB"], gt.inputs[0])
    lt = tree.nodes.new("CompositorNodeMath")
    lt.operation = "LESS_THAN"
    lt.inputs[1].default_value = pass_index_val + 0.5
    tree.links.new(rl_node.outputs["IndexOB"], lt.inputs[0])
    mul = tree.nodes.new("CompositorNodeMath")
    mul.operation = "MULTIPLY"
    tree.links.new(gt.outputs[0], mul.inputs[0])
    tree.links.new(lt.outputs[0], mul.inputs[1])
    fo = tree.nodes.new("CompositorNodeOutputFile")
    fo.base_path = output_dir
    fo.file_slots[0].path = slot_prefix
    fo.format.file_format = "PNG"
    fo.format.color_mode = "BW"
    fo.format.color_depth = "8"
    tree.links.new(mul.outputs[0], fo.inputs[0])
    return fo


def _process_segmaps(
    seg_frame_str: str,
    target_seg_nodes: Mapping[str, tuple[bpy.types.CompositorNodeOutputFile, str]],
    targets_cfg: Sequence[TargetConfig],
    segmap_dir: str,
) -> dict[str, TargetAnnotation] | None:
    """从 segmap 提取每个 target 的 bbox. 返回 dict 或 None (有 target 缺 bbox 则返回 None)."""
    target_bboxes: dict[str, list[float]] = {}
    for tgt_name, (_fo_node, prefix) in target_seg_nodes.items():
        seg_path = os.path.join(segmap_dir, f"{prefix}{seg_frame_str}.png")
        if os.path.exists(seg_path):
            seg_img = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
            if seg_img is not None:
                ys, xs = np.where(seg_img > 0)
                if len(xs) > 0:
                    target_bboxes[tgt_name] = [
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()),
                        int(ys.max()),
                    ]
            os.remove(seg_path)

    for tgt in targets_cfg:
        union_names = tgt.get("bbox_from_targets")
        if not union_names:
            continue
        constituent = [target_bboxes[n] for n in union_names if n in target_bboxes]
        if constituent:
            target_bboxes[tgt["name"]] = [
                min(b[0] for b in constituent),
                min(b[1] for b in constituent),
                max(b[2] for b in constituent),
                max(b[3] for b in constituent),
            ]

    targets_ann: dict[str, TargetAnnotation] = {}
    for tgt in targets_cfg:
        bb = target_bboxes.get(tgt["name"])
        if bb is None:
            return None
        targets_ann[tgt["name"]] = {"bbox_2d": bb}
    return targets_ann
