"""OpenVINO YOLO class-1 detection followed by twelve-point LiteHRNet inference.

Preprocessing and heatmap decoding follow the public HKUST detector at cb5ef2f3.
Model loading is explicit and performs no downloads. Instances may be reused for
successive frames; concurrent callers should use separate recognizer instances.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

from .types import FloatArray, ImageArray, KeypointDetection

if TYPE_CHECKING:
    from openvino import CompiledModel, Core, Model, PartialShape

_Float32Array = NDArray[np.float32]
_MEAN = np.array((123.675, 116.28, 103.53), dtype=np.float32)
_STD = np.array((58.395, 57.12, 57.375), dtype=np.float32)


def _model_path(model_path: str | Path) -> Path:
    path = Path(model_path).expanduser().resolve()
    if path.is_dir():
        candidates = sorted(path.glob("*.xml"))
        if not candidates:
            raise FileNotFoundError(f"OpenVINO model directory contains no XML: {path}")
        if len(candidates) != 1:
            raise ValueError(f"OpenVINO model directory must contain exactly one XML: {path}")
        path = candidates[0]
    if not path.is_file():
        raise FileNotFoundError(f"OpenVINO model XML not found: {path}")
    if path.suffix.lower() != ".xml":
        raise ValueError(f"OpenVINO model must be an XML file: {path}")
    weights = path.with_suffix(".bin")
    if not weights.is_file():
        raise FileNotFoundError(f"OpenVINO model weights not found: {weights}")
    return path


def _device(core: Core, requested: str) -> str:
    if not isinstance(requested, str):
        raise ValueError("OpenVINO device must be a device name such as CPU")
    name = requested.strip().upper()
    available = tuple(str(device) for device in core.available_devices)
    if name not in available:
        raise ValueError(
            f"OpenVINO device {requested!r} is unavailable; available devices: {available}. "
            "Use an explicit OpenVINO device name, not a CUDA device alias."
        )
    return name


def _input_shape(model: Model, label: str) -> tuple[int, int]:
    from openvino import Type

    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise ValueError(f"{label} model must have exactly one input and one output")
    shape = model.input().partial_shape
    if shape.rank.is_static and shape.rank.get_length() == 4 and shape[0].is_dynamic:
        if not all(shape[index].is_static for index in (1, 2, 3)):
            raise ValueError(f"{label} input channels and spatial sizes must be fixed, got {shape}")
        model.reshape([1, *(shape[index].get_length() for index in (1, 2, 3))])
        shape = model.input().partial_shape
    if not shape.is_static:
        raise ValueError(f"{label} model requires fixed NCHW channels and spatial sizes, got {shape}")
    dimensions = tuple(int(dimension.get_length()) for dimension in shape)
    if len(dimensions) != 4 or dimensions[:2] != (1, 3) or min(dimensions[2:]) <= 0:
        raise ValueError(f"{label} model requires NCHW batch-1 RGB input, got {shape}")
    if model.input().element_type != Type.f32:
        raise ValueError(f"{label} model input must be float32")
    return dimensions[3], dimensions[2]


def _validate_output_shape(shape: PartialShape, label: str) -> None:
    if not shape.rank.is_static:
        raise ValueError(f"{label} output rank must be known, got {shape}")
    rank = shape.rank.get_length()
    if label == "YOLO":
        if rank not in (2, 3):
            raise ValueError(f"YOLO output must be Nx6 or 1xNx6, got {shape}")
        if not shape[-1].is_static or shape[-1].get_length() != 6:
            raise ValueError(f"YOLO output must have six columns (xyxy, score, class), got {shape}")
        if rank == 3 and (not shape[0].is_static or shape[0].get_length() != 1):
            raise ValueError(f"YOLO output batch must be 1, got {shape}")
    elif (
        rank != 4
        or not shape.is_static
        or tuple(shape.to_shape())[:2] != (1, 12)
        or min(tuple(shape.to_shape())[2:]) <= 0
    ):
        raise ValueError(f"LiteHRNet output must be 1x12xHxW with positive heatmap size, got {shape}")


def _letterbox(image: ImageArray, size: tuple[int, int]) -> tuple[_Float32Array, float, int, int]:
    height, width = image.shape[:2]
    target_w, target_h = size
    scale = min(target_w / width, target_h / height)
    resized_w, resized_h = int(round(width * scale)), int(round(height * scale))
    if resized_w < 1 or resized_h < 1:
        raise ValueError("Image aspect ratio produces an empty YOLO letterbox dimension")
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (target_w - resized_w) // 2, (target_h - resized_h) // 2
    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
    blob = canvas[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / np.float32(255.0)
    return blob, scale, pad_x, pad_y


def _affine(bbox: FloatArray, size: tuple[int, int], padding: float) -> tuple[FloatArray, FloatArray]:
    x1, y1, x2, y2 = bbox
    scale = np.array((max(x2 - x1, 1.0), max(y2 - y1, 1.0)), dtype=np.float32)
    scale *= padding
    input_w, input_h = size
    aspect = input_w / input_h
    if scale[0] > scale[1] * aspect:
        scale[1] = scale[0] / aspect
    else:
        scale[0] = scale[1] * aspect
    center = np.array(((x1 + x2) * 0.5, (y1 + y2) * 0.5), dtype=np.float32)
    src = np.stack((center, center + np.array((-0.5 * scale[0], 0), dtype=np.float32), center))
    dst_center = np.array((input_w * 0.5, input_h * 0.5), dtype=np.float32)
    dst = np.stack((dst_center, dst_center + np.array((-0.5 * input_w, 0), dtype=np.float32), dst_center))
    for points in (src, dst):
        direction = points[0] - points[1]
        points[2] = points[1] + np.array((-direction[1], direction[0]), dtype=np.float32)
    return cv2.getAffineTransform(src, dst), cv2.getAffineTransform(dst, src)


def _decode(heatmaps: _Float32Array, size: tuple[int, int], inverse: FloatArray) -> tuple[FloatArray, FloatArray]:
    _, heatmap_h, heatmap_w = heatmaps.shape
    input_w, input_h = size
    coordinates = np.empty((12, 2), dtype=np.float64)
    scores = np.empty(12, dtype=np.float64)
    for index, heatmap in enumerate(heatmaps):
        y, x = divmod(int(np.argmax(heatmap)), heatmap_w)
        px, py = float(x), float(y)
        if 1 <= x < heatmap_w - 1 and 1 <= y < heatmap_h - 1:
            px += float(np.sign(heatmap[y, x + 1] - heatmap[y, x - 1])) * 0.25
            py += float(np.sign(heatmap[y + 1, x] - heatmap[y - 1, x])) * 0.25
        point = np.array(((px + 0.5) * input_w / heatmap_w, (py + 0.5) * input_h / heatmap_h, 1.0))
        coordinates[index] = inverse @ point
        scores[index] = float(heatmap[y, x])
    return coordinates, scores


class OpenVINORecognizer:
    """Reusable YOLO + LiteHRNet inference with original-image coordinates.

    Models must accept float32 NCHW inputs with fixed channels and spatial sizes. Dynamic
    batches are specialized to one. YOLO emits pixel-space
    ``[x1, y1, x2, y2, score, class_id]`` rows, already postprocessed by its
    export. LiteHRNet emits the twelve heatmaps in ``KEYPOINT_NAMES`` order.
    ``crop_margin`` expands only the keypoint crop; returned boxes stay the
    clipped detector boxes. Heatmap scores are raw peaks and are not filtered.
    """

    def __init__(
        self,
        *,
        yolo_model: str | Path,
        hrnet_model: str | Path,
        yolo_device: str = "CPU",
        hrnet_device: str = "CPU",
        hrnet_input_size: tuple[int, int] = (256, 256),
        yolo_conf: float = 0.30,
        crop_margin: float = 0.0,
        bbox_padding: float = 1.10,
    ) -> None:
        from openvino import Core

        if len(hrnet_input_size) != 2 or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0
            for value in hrnet_input_size
        ):
            raise ValueError("hrnet_input_size must contain positive integer width and height")
        for name, value in (("yolo_conf", yolo_conf), ("crop_margin", crop_margin), ("bbox_padding", bbox_padding)):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 <= yolo_conf <= 1 or crop_margin < 0 or bbox_padding <= 0:
            raise ValueError("Require yolo_conf in [0, 1], crop_margin >= 0, and bbox_padding > 0")
        self._input_size = hrnet_input_size
        self._confidence = float(yolo_conf)
        self._crop_margin = float(crop_margin)
        self._padding = float(bbox_padding)
        yolo_path, hrnet_path = _model_path(yolo_model), _model_path(hrnet_model)
        core = Core()
        yolo_device, hrnet_device = _device(core, yolo_device), _device(core, hrnet_device)
        yolo, hrnet = core.read_model(str(yolo_path)), core.read_model(str(hrnet_path))
        self._yolo_size = _input_shape(yolo, "YOLO")
        hrnet_size = _input_shape(hrnet, "LiteHRNet")
        if hrnet_size != hrnet_input_size:
            raise ValueError(f"LiteHRNet input size mismatch: model={hrnet_size}, configured={hrnet_input_size}")
        _validate_output_shape(yolo.output().partial_shape, "YOLO")
        _validate_output_shape(hrnet.output().partial_shape, "LiteHRNet")
        self._yolo: CompiledModel = core.compile_model(yolo, yolo_device)
        self._hrnet: CompiledModel = core.compile_model(hrnet, hrnet_device)

    def predict(self, image_bgr: ImageArray) -> list[KeypointDetection]:
        """Return every class-1 detection in descending detector-score order."""
        if not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8:
            raise ValueError("image_bgr must be a uint8 numpy array")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or min(image_bgr.shape[:2]) <= 0:
            raise ValueError(f"image_bgr must be nonempty HxWx3 BGR, got {image_bgr.shape}")
        height, width = image_bgr.shape[:2]
        blob, scale, pad_x, pad_y = _letterbox(image_bgr, self._yolo_size)
        output = np.asarray(self._yolo([blob])[self._yolo.output()], dtype=np.float64)
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]
        if output.ndim != 2 or output.shape[1] != 6:
            raise ValueError(f"YOLO output must be Nx6 or 1xNx6, got {output.shape}")
        if not np.isfinite(output).all():
            raise ValueError("YOLO output contains nonfinite values")
        if np.any((output[:, 4] < 0) | (output[:, 4] > 1)):
            raise ValueError("YOLO scores must be in [0, 1]")
        if np.any((output[:, 5] < 0) | (output[:, 5] != np.floor(output[:, 5]))):
            raise ValueError("YOLO class IDs must be nonnegative integers")
        selected = output[(output[:, 4] >= self._confidence) & (output[:, 5] == 1)]
        selected = selected[np.argsort(-selected[:, 4], kind="stable")]
        detections: list[KeypointDetection] = []
        for row in selected:
            bbox = row[:4].copy()
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                raise ValueError("YOLO target bbox must have increasing x/y coordinates")
            bbox[[0, 2]] = np.clip((bbox[[0, 2]] - pad_x) / scale, 0, width)
            bbox[[1, 3]] = np.clip((bbox[[1, 3]] - pad_y) / scale, 0, height)
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                raise ValueError("YOLO target bbox has no area inside the original image")
            crop = bbox.copy()
            margin_x = (bbox[2] - bbox[0]) * self._crop_margin
            margin_y = (bbox[3] - bbox[1]) * self._crop_margin
            crop[[0, 2]] = np.clip(crop[[0, 2]] + (-margin_x, margin_x), 0, width)
            crop[[1, 3]] = np.clip(crop[[1, 3]] + (-margin_y, margin_y), 0, height)
            points, scores = self._predict_keypoints(image_bgr, crop)
            detections.append(KeypointDetection(bbox, float(row[4]), points, scores, (height, width)))
        return detections

    def _predict_keypoints(self, image: ImageArray, bbox: FloatArray) -> tuple[FloatArray, FloatArray]:
        affine, inverse = _affine(bbox, self._input_size, self._padding)
        crop = cv2.warpAffine(
            image, affine, self._input_size, flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        normalized = (crop[..., ::-1].astype(np.float32) - _MEAN) / _STD
        blob = normalized.transpose(2, 0, 1)[None].astype(np.float32)
        output = np.asarray(self._hrnet([blob])[self._hrnet.output()], dtype=np.float32)
        if output.ndim != 4 or output.shape[:2] != (1, 12) or min(output.shape[2:]) <= 0:
            raise ValueError(f"LiteHRNet output must be 1x12xHxW, got {output.shape}")
        if not np.isfinite(output).all():
            raise ValueError("LiteHRNet heatmaps contain nonfinite values")
        return _decode(output[0], self._input_size, inverse)
