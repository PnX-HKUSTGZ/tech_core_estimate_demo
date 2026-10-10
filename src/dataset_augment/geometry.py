"""Label-aware crop, affine, flip and letterbox operations.

Crop sampling is adapted from ``nn/yolo/train/augmentations.py`` at upstream
``cb5ef2f3``. Coordinates use continuous pixel edges (0..width, 0..height).
OpenCV warps receive the corresponding pixel-center transform, so an edge-space
horizontal flip x'=width-x agrees with reversing the image's columns.
"""
from __future__ import annotations

import cv2
import numpy as np

from .types import (
    FloatArray, Image, IntArray, JsonObject, JsonValue, Labels, boolean, integer, interval,
    json_value, number, rng_for, section, string, values,
)


def filter_labels(labels: Labels, settings: JsonObject) -> Labels:
    """Apply final pixel-size and visible-keypoint thresholds without mutation."""
    sizes = labels.boxes[:, 2:4] - labels.boxes[:, :2]
    keep = ((sizes[:, 0] > 0) & (sizes[:, 1] > 0)
            & (sizes[:, 0] + 1e-9 >= number(settings, "min_bbox_width_px"))
            & (sizes[:, 1] + 1e-9 >= number(settings, "min_bbox_height_px")))
    if labels.keypoints.shape[1]:
        keep &= np.count_nonzero(labels.keypoints[:, :, 2] > 0, axis=1) >= integer(settings, "min_keypoints")
    return labels.select(keep)


def _area(polygon: FloatArray) -> float:
    if len(polygon) < 3:
        return 0.0
    return float(abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1))
                     - np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1))) * 0.5)


def _clip(polygon: FloatArray, width: int, height: int) -> FloatArray:
    """Sutherland-Hodgman clipping; retain the actual polygon area."""
    result = polygon
    for axis, boundary, direction in ((0, 0.0, 1), (0, float(width), -1),
                                      (1, 0.0, 1), (1, float(height), -1)):
        if len(result) == 0:
            break
        points: list[FloatArray] = []
        previous = result[-1]
        previous_inside = direction * (float(previous[axis]) - boundary) >= 0
        for current in result:
            current_inside = direction * (float(current[axis]) - boundary) >= 0
            if previous_inside != current_inside:
                fraction = (boundary - float(previous[axis])) / float(current[axis] - previous[axis])
                points.append(previous + fraction * (current - previous))
            if current_inside:
                points.append(current)
            previous, previous_inside = current, current_inside
        result = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return result


def _transform_labels(labels: Labels, matrix: FloatArray, size: tuple[int, int],
                      min_area_ratio: float) -> Labels:
    width, height = size
    output = labels.copy()
    keep = np.zeros(len(labels.boxes), dtype=np.bool_)
    for index, box in enumerate(labels.boxes):
        x1, y1, x2, y2 = box
        corners = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
        transformed = corners @ matrix[:2, :2].T + matrix[:2, 2]
        original_area = _area(transformed)
        clipped = _clip(transformed, width, height)
        clipped_area = _area(clipped)
        if (original_area > 0 and clipped_area > 0
                and clipped_area / original_area + 1e-12 >= min_area_ratio):
            output.boxes[index, :2] = clipped.min(axis=0)
            output.boxes[index, 2:] = clipped.max(axis=0)
            keep[index] = True
    points = output.keypoints
    visible = points[:, :, 2] > 0
    points[:, :, :2] = points[:, :, :2] @ matrix[:2, :2].T + matrix[:2, 2]
    # Boundaries are inclusive in the edge-coordinate contract. This preserves
    # points exactly on an edge under two flips while rejecting all outside points.
    tolerance = 1e-12 * max(width, height, 1)
    inside = ((points[:, :, 0] >= -tolerance) & (points[:, :, 0] <= width + tolerance)
              & (points[:, :, 1] >= -tolerance) & (points[:, :, 1] <= height + tolerance))
    points[:, :, 0] = np.clip(points[:, :, 0], 0, width)
    points[:, :, 1] = np.clip(points[:, :, 1], 0, height)
    points[~(visible & inside)] = 0
    return output.select(keep)


def _enabled(settings: JsonObject, rng: np.random.Generator) -> bool:
    return boolean(settings, "enabled") and float(rng.random()) < number(settings, "p")


def _sample(settings: JsonObject, key: str, rng: np.random.Generator) -> float:
    return float(rng.uniform(*interval(settings, key)))


def _identity() -> FloatArray:
    return np.eye(3, dtype=np.float64)


def _crop(image: Image, labels: Labels, settings: JsonObject, seed: int,
          attempt: int, min_area_ratio: float) -> tuple[Image, Labels, FloatArray, JsonObject]:
    rng = rng_for(seed, "crop", attempt)
    matrix = _identity()
    audit: JsonObject = {"operation": "crop", "applied": False}
    if not _enabled(settings, rng):
        return image, labels, matrix, audit
    height, width = image.shape[:2]
    mode = string(settings, "mode")
    if mode == "choice":
        weights = section(settings, "choice_weights")
        random_weight, box_weight = number(weights, "random"), number(weights, "box")
        mode = "random" if float(rng.random()) < random_weight / (random_weight + box_weight) else "box"
    audit["mode"] = mode
    if mode == "random":
        crop_settings = section(settings, "random")
        crop_width = max(1, int(width * _sample(crop_settings, "width_ratio", rng)))
        crop_height = max(1, int(height * _sample(crop_settings, "height_ratio", rng)))
        x1 = int(rng.integers(0, width - crop_width + 1))
        y1 = int(rng.integers(0, height - crop_height + 1))
    elif mode == "box":
        crop_settings = section(settings, "box")
        eligible: IntArray = np.arange(len(labels.boxes), dtype=np.int64)
        targets = crop_settings["target_classes"]
        if targets is not None:
            target_values = [integer({"class": value}, "class")
                             for value in values(crop_settings, "target_classes")]
            eligible = eligible[np.isin(labels.class_ids, target_values)]
        sizes = labels.boxes[:, 2:] - labels.boxes[:, :2]
        eligible = eligible[(sizes[eligible, 0] > 1) & (sizes[eligible, 1] > 1)]
        if len(eligible) == 0:
            audit["skip_reason"] = "no_eligible_box"
            return image, labels, matrix, audit
        index = int(rng.choice(eligible))
        box = labels.boxes[index]
        box_width, box_height = float(box[2] - box[0]), float(box[3] - box[1])
        factor = _sample(crop_settings, "scale", rng)
        minimum = number(crop_settings, "min_crop_size_ratio")
        float_width = min(width, max(box_width * factor, width * minimum))
        float_height = min(height, max(box_height * factor, height * minimum))
        jitter = number(crop_settings, "center_jitter")
        center_x = float((box[0] + box[2]) / 2) + float(rng.uniform(-jitter, jitter)) * box_width
        center_y = float((box[1] + box[3]) / 2) + float(rng.uniform(-jitter, jitter)) * box_height
        x1 = max(0, min(int(round(center_x - float_width / 2)), int(width - float_width)))
        y1 = max(0, min(int(round(center_y - float_height / 2)), int(height - float_height)))
        crop_width = max(1, min(width - x1, int(round(float_width))))
        crop_height = max(1, min(height - y1, int(round(float_height))))
        audit.update({"target_row_id": int(labels.row_ids[index]), "scale": factor})
    else:
        raise ValueError(f"Unsupported crop mode: {mode}")
    matrix[:2, 2] = [-x1, -y1]
    output = image[y1:y1 + crop_height, x1:x1 + crop_width].copy()
    transformed = _transform_labels(labels, matrix, (crop_width, crop_height), min_area_ratio)
    audit.update({"applied": True, "xyxy": [x1, y1, x1 + crop_width, y1 + crop_height]})
    return output, transformed, matrix, audit


def _interpolation(settings: JsonObject) -> int:
    return {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC}[string(settings, "interpolation")]


def _fill_bgr(settings: JsonObject) -> tuple[int, int, int]:
    rgb = values(settings, "fill_rgb")
    proxy: JsonObject = {"r": rgb[0], "g": rgb[1], "b": rgb[2]}
    return integer(proxy, "b"), integer(proxy, "g"), integer(proxy, "r")


def _affine(image: Image, labels: Labels, settings: JsonObject, geometry: JsonObject,
            seed: int, attempt: int, min_area_ratio: float) -> tuple[Image, Labels, FloatArray, JsonObject]:
    rng = rng_for(seed, "affine", attempt)
    matrix = _identity()
    audit: JsonObject = {"operation": "affine", "applied": False}
    if not _enabled(settings, rng):
        return image, labels, matrix, audit
    height, width = image.shape[:2]
    degrees = _sample(settings, "degrees", rng)
    scale = _sample(settings, "scale", rng)
    translation_x = _sample(settings, "translate_x", rng) * width
    translation_y = _sample(settings, "translate_y", rng) * height
    matrix[:2] = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, scale)
    matrix[:2, 2] += [translation_x, translation_y]
    edge_from_center = _identity()
    edge_from_center[:2, 2] = 0.5
    center_from_edge = _identity()
    center_from_edge[:2, 2] = -0.5
    pixel_matrix = center_from_edge @ matrix @ edge_from_center
    output = np.asarray(cv2.warpAffine(image, pixel_matrix[:2], (width, height),
                                      flags=_interpolation(geometry), borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=_fill_bgr(geometry)), dtype=np.uint8)
    transformed = _transform_labels(labels, matrix, (width, height), min_area_ratio)
    audit.update({"applied": True, "degrees": degrees, "scale": scale,
                  "translation_px": [translation_x, translation_y]})
    return output, transformed, matrix, audit


def _flip(image: Image, labels: Labels, settings: JsonObject, horizontal: bool,
          seed: int, attempt: int) -> tuple[Image, Labels, FloatArray, JsonObject]:
    operation = "horizontal_flip" if horizontal else "vertical_flip"
    rng = rng_for(seed, operation, attempt)
    matrix = _identity()
    audit: JsonObject = {"operation": operation, "applied": False}
    if not _enabled(settings, rng):
        return image, labels, matrix, audit
    height, width = image.shape[:2]
    axis = 0 if horizontal else 1
    matrix[axis, axis] = -1
    matrix[axis, 2] = width if horizontal else height
    output = np.ascontiguousarray(image[:, ::-1] if horizontal else image[::-1])
    transformed = _transform_labels(labels, matrix, (width, height), 0)
    if labels.keypoints.shape[1]:
        raw_indices = values(settings, "keypoint_indices")
        indices = [integer({"index": item}, "index") for item in raw_indices]
        if (sorted(indices) != list(range(labels.keypoints.shape[1]))
                or any(indices[indices[index]] != index for index in range(len(indices)))):
            raise ValueError(f"{operation} requires an involutive complete keypoint permutation")
        transformed.keypoints = transformed.keypoints[:, indices, :].copy()
        audit["keypoint_indices"] = json_value(indices)
    audit["applied"] = True
    return output, transformed, matrix, audit


def _resize(image: Image, labels: Labels, settings: JsonObject, geometry: JsonObject,
            source_size: tuple[int, int]) -> tuple[Image, Labels, FloatArray, JsonObject]:
    size = settings["size"]
    if size == "source":
        width, height = source_size
    else:
        raw_size = values(settings, "size")
        width = integer({"width": raw_size[0]}, "width")
        height = integer({"height": raw_size[1]}, "height")
    current_height, current_width = image.shape[:2]
    matrix = _identity()
    audit: JsonObject = {"operation": "resize", "applied": False,
                         "output_size": [width, height]}
    if (width, height) == (current_width, current_height):
        return image, labels, matrix, audit
    scale = min(width / current_width, height / current_height)
    resized_width = max(1, min(width, int(round(current_width * scale))))
    resized_height = max(1, min(height, int(round(current_height * scale))))
    left, top = (width - resized_width) // 2, (height - resized_height) // 2
    matrix[0, 0], matrix[1, 1] = resized_width / current_width, resized_height / current_height
    matrix[:2, 2] = [left, top]
    output = np.empty((height, width, 3), dtype=np.uint8)
    output[:] = _fill_bgr(geometry)
    output[top:top + resized_height, left:left + resized_width] = cv2.resize(
        image, (resized_width, resized_height), interpolation=_interpolation(geometry))
    transformed = _transform_labels(labels, matrix, (width, height), 0)
    audit.update({"applied": True, "resized_size": [resized_width, resized_height],
                  "padding_left_top": [left, top]})
    return output, transformed, matrix, audit


def _visibility_changes(before: Labels, after: Labels, point_indices: list[int]) -> list[JsonValue]:
    changes: list[JsonValue] = []
    originals = {int(row): index for index, row in enumerate(before.row_ids)}
    for row_index, row in enumerate(after.row_ids):
        previous = before.keypoints[originals[int(row)], point_indices, 2]
        current = after.keypoints[row_index, :, 2]
        for point_index in np.flatnonzero(previous != current):
            changes.append({"row_id": int(row), "point_index": int(point_index),
                            "source_point_index": point_indices[int(point_index)],
                            "before": int(previous[point_index]), "after": int(current[point_index])})
    return changes


def _attempt(image: Image, labels: Labels, transforms: JsonObject, settings: JsonObject,
             seed: int, attempt: int, source_size: tuple[int, int],
             fallback: bool) -> tuple[Image, Labels, JsonObject]:
    original = labels
    output, transformed = image.copy(), labels.copy()
    transformed.keypoints[transformed.keypoints[:, :, 2] == 0] = 0
    geometry = section(transforms, "geometry")
    min_ratio = number(settings, "min_bbox_area_ratio")
    total = _identity()
    operations: list[JsonValue] = []
    point_indices = list(range(labels.keypoints.shape[1]))
    if not fallback:
        output, transformed, matrix, audit = _crop(
            output, transformed, section(transforms, "crop"), seed, attempt, min_ratio)
        total = matrix @ total
        audit["matrix"] = json_value(matrix.tolist())
        operations.append(audit)
        output, transformed, matrix, audit = _affine(
            output, transformed, section(transforms, "affine"), geometry, seed, attempt, min_ratio)
        total = matrix @ total
        audit["matrix"] = json_value(matrix.tolist())
        operations.append(audit)
        for operation, horizontal in (("horizontal_flip", True), ("vertical_flip", False)):
            output, transformed, matrix, audit = _flip(
                output, transformed, section(transforms, operation), horizontal, seed, attempt)
            total = matrix @ total
            audit["matrix"] = json_value(matrix.tolist())
            if audit["applied"] and point_indices:
                permutation = values(audit, "keypoint_indices")
                point_indices = [point_indices[integer({"index": value}, "index")]
                                 for value in permutation]
            operations.append(audit)
    output, transformed, matrix, audit = _resize(
        output, transformed, section(transforms, "resize"), geometry, source_size)
    total = matrix @ total
    audit["matrix"] = json_value(matrix.tolist())
    operations.append(audit)
    transformed = filter_labels(transformed, settings)
    after_ids = set(int(row) for row in transformed.row_ids)
    return output, transformed, {
        "operations": operations, "matrix": json_value(total.tolist()),
        "removed_row_ids": [int(row) for row in original.row_ids if int(row) not in after_ids],
        "visibility_changes": _visibility_changes(original, transformed, point_indices),
        "fallback": fallback,
    }


def apply_geometry(image: Image, labels: Labels, transforms: JsonObject,
                   label_settings: JsonObject, seed: int,
                   source_size: tuple[int, int]) -> tuple[Image, Labels, JsonObject]:
    """Apply deterministic geometry, retrying total positive-target loss.

    ``source_size`` is (width, height), used by ``resize.size: source``. Empty
    positive policy ``retry`` permits at most ``max_geometry_attempts`` attempts;
    exhaustion restores original geometry before requested output letterboxing.
    Original negative samples always execute exactly one attempt.
    """
    retry = len(labels.boxes) > 0 and string(label_settings, "empty_positive_policy") == "retry"
    maximum = integer(label_settings, "max_geometry_attempts") if retry else 1
    rejected: list[JsonValue] = []
    for attempt in range(maximum):
        output, transformed, audit = _attempt(image, labels, transforms, label_settings,
                                               seed, attempt, source_size, False)
        if len(transformed.boxes) > 0 or not retry:
            audit["attempts"] = attempt + 1
            audit["rejected_attempts"] = rejected
            return output, transformed, audit
        rejected.append(audit)
    output, transformed, audit = _attempt(image, labels, transforms, label_settings,
                                           seed, maximum, source_size, True)
    if len(transformed.boxes) == 0:
        raise ValueError("Original positive sample fails label thresholds even after geometry fallback")
    audit["attempts"] = maximum
    audit["rejected_attempts"] = rejected
    return output, transformed, audit
