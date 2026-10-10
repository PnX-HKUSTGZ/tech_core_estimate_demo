"""Independent RGB camera degradation and label-aware rectangular occlusion.

The camera effects adapt ``tech_core_hkust@cb5ef2f3``. Images entering and
leaving this module are BGR, as in OpenCV; all configured colors are RGB.
"""
from __future__ import annotations

import math
from typing import TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

from .types import (
    Image, JsonObject, JsonValue, Labels, boolean, integer, interval, number,
    rng_for, section, string, values,
)

FloatImage: TypeAlias = NDArray[np.float32]
Rectangle: TypeAlias = tuple[int, int, int, int]


def _sample(config: JsonObject, key: str, rng: np.random.Generator) -> float:
    low, high = interval(config, key)
    return float(rng.uniform(low, high))


def _active(config: JsonObject, rng: np.random.Generator) -> bool:
    return boolean(config, "enabled") and float(rng.random()) < number(config, "p")


def _number_list(config: JsonObject, key: str) -> list[float]:
    return [number({"value": item}, "value") for item in values(config, key)]


def _int_list(config: JsonObject, key: str) -> list[int]:
    return [integer({"value": item}, "value") for item in values(config, key)]


def _union_area(rectangles: list[tuple[float, float, float, float]]) -> float:
    """Area of a rectangle union, including overlap only once."""
    xs = sorted({edge for x0, _, x1, _ in rectangles for edge in (x0, x1)})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((y0, y1) for x0, y0, x1, y1 in rectangles
                           if x0 < right and x1 > left)
        if not intervals:
            continue
        bottom, top = intervals[0]
        height = 0.0
        for low, high in intervals[1:]:
            if low > top:
                height += top - bottom
                bottom, top = low, high
            else:
                top = max(top, high)
        height += top - bottom
        area += (right - left) * height
    return area


def _bbox_coverage(labels: Labels, rectangles: list[Rectangle]) -> list[JsonValue]:
    result: list[JsonValue] = []
    for index, box in enumerate(labels.boxes):
        bx0, by0, bx1, by1 = (float(value) for value in box)
        intersections: list[tuple[float, float, float, float]] = []
        for x0, y0, x1, y1 in rectangles:
            left, right = max(bx0, x0), min(bx1, x1)
            top, bottom = max(by0, y0), min(by1, y1)
            if right > left and bottom > top:
                intersections.append((left, top, right, bottom))
        area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
        result.append({"row_id": int(labels.row_ids[index]),
                       "class_id": int(labels.class_ids[index]),
                       "occluded_area_px": _union_area(intersections),
                       "bbox_area_px": area,
                       "coverage_fraction": _union_area(intersections) / area if area > 0 else 0.0})
    return result


def _allowed(rectangle: Rectangle, labels: Labels, maximum: float) -> bool:
    if maximum >= 1.0 or len(labels.boxes) == 0:
        return True
    x0, y0, x1, y1 = rectangle
    boxes = labels.boxes
    intersection = (np.maximum(0.0, np.minimum(x1, boxes[:, 2]) - np.maximum(x0, boxes[:, 0]))
                    * np.maximum(0.0, np.minimum(y1, boxes[:, 3]) - np.maximum(y0, boxes[:, 1])))
    area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-12)
    return bool(np.all(intersection / area <= maximum))


def _occlusion(
    rgb: Image, labels: Labels, config: JsonObject, seed: int,
) -> JsonObject:
    rng = rng_for(seed, "occlusion")
    if not _active(config, rng):
        return {"applied": False, "rectangles": [], "keypoint_visibility_changes": []}
    height, width = rgb.shape[:2]
    count_low, count_high = _int_list(config, "count")
    requested_count = int(rng.integers(count_low, count_high + 1))
    visible = labels.keypoints[:, :, :2][labels.keypoints[:, :, 2] == 2]
    rectangles: list[Rectangle] = []
    records: list[JsonValue] = []
    changes: list[JsonValue] = []
    skipped = 0
    for _ in range(requested_count):
        for attempt in range(integer(config, "max_attempts")):
            rw = max(1, min(width, round(_sample(config, "width_ratio", rng) * width)))
            rh = max(1, min(height, round(_sample(config, "height_ratio", rng) * height)))
            if len(visible) > 0 and float(rng.random()) < number(config, "keypoint_anchor_p"):
                point = visible[int(rng.integers(len(visible)))]
                jitter = number(config, "anchor_jitter")
                cx = float(point[0]) + float(rng.uniform(-jitter, jitter)) * rw
                cy = float(point[1]) + float(rng.uniform(-jitter, jitter)) * rh
                anchor = "visible_keypoint"
            else:
                cx, cy = float(rng.uniform(0, width)), float(rng.uniform(0, height))
                anchor = "random_canvas"
            x0 = max(0, min(width - rw, round(cx - rw / 2)))
            y0 = max(0, min(height - rh, round(cy - rh / 2)))
            rectangle = (x0, y0, x0 + rw, y0 + rh)
            if _allowed(rectangle, labels, number(config, "max_bbox_overlap")):
                break
        else:
            skipped += 1
            continue
        x0, y0, x1, y1 = rectangle
        mode = string(config, "fill_mode")
        color: list[JsonValue] | None = None
        if mode == "noise":
            rgb[y0:y1, x0:x1] = rng.integers(0, 256, (rh, rw, 3), dtype=np.uint8)
        elif mode == "constant":
            color = [int(item) for item in _int_list(config, "fill_rgb")]
            rgb[y0:y1, x0:x1] = np.asarray(color, dtype=np.uint8)
        elif mode == "random_color":
            low, high = _int_list(config, "color_base")
            base = int(rng.integers(low, high + 1))
            jitter_low, jitter_high = _int_list(config, "color_jitter")
            sampled = np.clip(base + rng.integers(jitter_low, jitter_high + 1, 3), 0, 255)
            color = [int(item) for item in sampled]
            rgb[y0:y1, x0:x1] = np.asarray(color, dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported occlusion fill: {mode}")
        rectangles.append(rectangle)
        records.append({"xyxy_pixels": list(rectangle), "right_bottom_exclusive": True,
                        "anchor": anchor, "fill": mode, "fill_rgb": color, "attempts": attempt + 1})
    for row_index, points in enumerate(labels.keypoints):
        for point_index, point in enumerate(points):
            if point[2] != 2:
                continue
            x, y = float(point[0]), float(point[1])
            covering = [index for index, (x0, y0, x1, y1) in enumerate(rectangles)
                        if x0 <= x < x1 and y0 <= y < y1]
            if covering:
                point[2] = 1
                covering_json: list[JsonValue] = [index for index in covering]
                changes.append({"row_id": int(labels.row_ids[row_index]),
                                "class_id": int(labels.class_ids[row_index]),
                                "point_index": point_index, "before": 2, "after": 1,
                                "occluder_indices": covering_json})
    return {"applied": bool(rectangles), "requested_count": requested_count,
            "skipped_count": skipped, "rectangles": records,
            "keypoint_visibility_changes": changes,
            "cumulative_bbox_coverage": _bbox_coverage(labels, rectangles)}


def _blur(pixels: FloatImage, config: JsonObject, seed: int) -> tuple[FloatImage, JsonObject]:
    if not boolean(config, "enabled"):
        return pixels, {"applied": False, "kind": "none"}
    rng = rng_for(seed, "blur")
    weights = section(config, "weights")
    kinds = ["none", "gaussian", "motion"]
    probabilities = np.asarray([number(weights, kind) for kind in kinds], dtype=np.float64)
    kind = str(rng.choice(kinds, p=probabilities / probabilities.sum()))
    if kind == "none":
        return pixels, {"applied": False, "kind": kind}
    kernel_weights = np.asarray(_number_list(config, "kernel_weights"), dtype=np.float64)
    size = int(rng.choice(_int_list(config, "kernel_sizes"), p=kernel_weights / kernel_weights.sum()))
    record: JsonObject = {"applied": True, "kind": kind, "kernel_size": size}
    if kind == "gaussian":
        sigma = _sample(config, "gaussian_sigma", rng)
        record["sigma_px"] = sigma
        return np.asarray(cv2.GaussianBlur(pixels, (size, size), sigma,
                                          borderType=cv2.BORDER_REFLECT_101), dtype=np.float32), record
    angle_deg = _sample(config, "motion_angle", rng)
    angle = math.radians(angle_deg)
    center = (size - 1) / 2
    start = (round(center - center * math.cos(angle)), round(center - center * math.sin(angle)))
    end = (round(center + center * math.cos(angle)), round(center + center * math.sin(angle)))
    kernel = np.zeros((size, size), dtype=np.float32)
    cv2.line(kernel, start, end, (1.0,), thickness=1)
    kernel /= kernel.sum()
    record.update(angle_deg=angle_deg, raster_endpoints_xy=[list(start), list(end)])
    return np.asarray(cv2.filter2D(pixels, -1, kernel, borderType=cv2.BORDER_REFLECT_101),
                      dtype=np.float32), record


def apply_photometric(
    image: Image, labels: Labels, transforms: JsonObject, seed: int,
) -> tuple[Image, Labels, JsonObject, bytes | None]:
    """Return new image/labels, actual parameters, and reusable JPEG bytes.

    Each effect uses its own seed, so changing one effect does not perturb the
    random parameters of other effects. A returned JPEG payload encodes the
    returned (decoded) image and must not be JPEG-encoded a second time.
    """
    rgb = image[:, :, ::-1].copy()
    result_labels = labels.copy()
    record: JsonObject = {"occlusion": _occlusion(rgb, result_labels, section(transforms, "occlusion"), seed)}
    pixels = rgb.astype(np.float32) / 255.0
    exposure = section(transforms, "exposure")
    exposure_rng = rng_for(seed, "exposure")
    exposure_on = _active(exposure, exposure_rng)
    ev = _sample(exposure, "ev", exposure_rng) if exposure_on else 0.0
    record["exposure"] = {"applied": exposure_on, "ev": ev}
    balance = section(transforms, "white_balance")
    balance_rng = rng_for(seed, "white_balance")
    balance_on = _active(balance, balance_rng)
    gains = [_sample(balance, key, balance_rng) for key in ("r_gain", "g_gain", "b_gain")] if balance_on else [1.0, 1.0, 1.0]
    record["white_balance"] = {"applied": balance_on, "rgb_gains": [float(gain) for gain in gains]}
    pixels *= np.asarray(gains, dtype=np.float32)[None, None, :] * (2.0 ** ev)
    contrast = section(transforms, "contrast")
    contrast_rng = rng_for(seed, "contrast")
    contrast_on = _active(contrast, contrast_rng)
    factor = _sample(contrast, "factor", contrast_rng) if contrast_on else 1.0
    center = number(contrast, "center")
    if contrast_on:
        pixels = (pixels - center) * factor + center
    record["contrast"] = {"applied": contrast_on, "factor": factor, "center": center}
    gamma = section(transforms, "gamma")
    gamma_rng = rng_for(seed, "gamma")
    gamma_on = _active(gamma, gamma_rng)
    exponent = _sample(gamma, "gamma", gamma_rng) if gamma_on else 1.0
    if gamma_on:
        pixels = np.power(np.clip(pixels, 0.0, 1.0), exponent)
    record["gamma"] = {"applied": gamma_on, "gamma": exponent, "definition": "output=clip(input,0,1)**gamma"}
    vignette = section(transforms, "vignette")
    vignette_rng = rng_for(seed, "vignette")
    vignette_on = _active(vignette, vignette_rng)
    vignette_record: JsonObject = {"applied": vignette_on}
    if vignette_on:
        strength = _sample(vignette, "strength", vignette_rng)
        cx, cy = _sample(vignette, "center_x", vignette_rng), _sample(vignette, "center_y", vignette_rng)
        height, width = pixels.shape[:2]
        x = np.linspace(0, 1, width, dtype=np.float32) - cx
        y = np.linspace(0, 1, height, dtype=np.float32) - cy
        radius = y[:, None] ** 2 + x[None, :] ** 2
        radius /= max(float(radius.max()), 1e-9)
        pixels *= (1.0 - strength * radius)[:, :, None]
        vignette_record.update(strength=strength, center_normalized=[cx, cy])
    record["vignette"] = vignette_record
    pixels, blur_record = _blur(pixels, section(transforms, "blur"), seed)
    record["blur"] = blur_record
    noise = section(transforms, "noise")
    noise_rng = rng_for(seed, "noise")
    noise_on = _active(noise, noise_rng)
    sigma = _sample(noise, "sigma_uint8", noise_rng) if noise_on else 0.0
    if noise_on:
        pixels += noise_rng.normal(0.0, sigma / 255.0, pixels.shape).astype(np.float32)
    record["noise"] = {"applied": noise_on, "sigma_uint8": sigma}
    output = np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8)[:, :, ::-1].copy()
    jpeg = section(transforms, "jpeg")
    jpeg_rng = rng_for(seed, "jpeg")
    jpeg_on = _active(jpeg, jpeg_rng)
    payload: bytes | None = None
    jpeg_record: JsonObject = {"applied": jpeg_on}
    if jpeg_on:
        low, high = _int_list(jpeg, "quality")
        quality = int(jpeg_rng.integers(low, high + 1))
        ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("JPEG augmentation encoding failed")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("JPEG augmentation decoding failed")
        payload = encoded.tobytes()
        output = np.asarray(decoded, dtype=np.uint8)
        jpeg_record["quality"] = quality
    record["jpeg"] = jpeg_record
    record["operation_order"] = ["occlusion", "white_balance/exposure", "contrast", "gamma",
                                 "vignette", "blur", "noise", "jpeg"]
    return output, result_labels, record, payload
