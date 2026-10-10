"""Read-only YOLO discovery and full image/label preflight."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from .config import AugmentConfig
from .types import (Image, JsonObject, JsonValue, Labels, boolean, derive_seed,
                    integer, json_value, obj, section, string, values)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class InputImage:
    split: str
    relative: str
    image: Path
    label: Path | None
    image_sha256: str
    label_sha256: str
    selected: bool
    width: int
    height: int


@dataclass
class DatasetPlan:
    source_yaml: Path
    source_yaml_sha256: str
    output: Path
    items: list[InputImage]
    metadata: JsonObject
    split_settings: dict[str, JsonObject]
    counts: JsonObject
    keypoint_count: int
    class_ids: set[int]


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def decode_image(content: bytes, path: Path) -> Image:
    pixels = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if pixels is None or pixels.dtype != np.uint8:
        raise ValueError(f"Cannot decode an 8-bit image: {path}")
    if pixels.ndim == 2:
        pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    if pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) < 1:
        raise ValueError(f"Expected RGB or grayscale image (no alpha): {path}")
    return np.asarray(pixels, dtype=np.uint8)


def parse_labels(content: bytes, path: Path, width: int, height: int,
                 keypoint_count: int, class_ids: set[int]) -> Labels:
    rows: list[list[float]] = []
    ids: list[int] = []
    for line_number, line in enumerate(content.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = [float(token) for token in line.split()]
        except ValueError as error:
            raise ValueError(f"Non-numeric label {path}:{line_number}") from error
        if len(row) != 5 + 3 * keypoint_count or not np.isfinite(row).all():
            raise ValueError(f"Wrong label columns/nonfinite values: {path}:{line_number}")
        if not row[0].is_integer() or int(row[0]) not in class_ids:
            raise ValueError(f"Unknown class: {path}:{line_number}")
        cx, cy, bw, bh = row[1:5]
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError(f"Invalid normalized box: {path}:{line_number}")
        if min(cx-bw/2, cy-bh/2) < -1e-6 or max(cx+bw/2, cy+bh/2) > 1+1e-6:
            raise ValueError(f"Box outside image: {path}:{line_number}")
        for index in range(keypoint_count):
            x, y, visibility = row[5+3*index:8+3*index]
            if not (0 <= x <= 1 and 0 <= y <= 1 and visibility in (0, 1, 2)):
                raise ValueError(f"Invalid keypoint: {path}:{line_number}")
        rows.append(row)
        ids.append(line_number)
    array = np.asarray(rows, dtype=np.float64).reshape(-1, 5+3*keypoint_count)
    boxes = array[:, 1:5].copy()
    boxes[:, 0] = (array[:, 1]-array[:, 3]/2)*width
    boxes[:, 1] = (array[:, 2]-array[:, 4]/2)*height
    boxes[:, 2] = (array[:, 1]+array[:, 3]/2)*width
    boxes[:, 3] = (array[:, 2]+array[:, 4]/2)*height
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    points = array[:, 5:].copy().reshape(len(rows), keypoint_count, 3)
    points[:, :, 0] *= width
    points[:, :, 1] *= height
    points[points[:, :, 2] == 0] = 0
    return Labels(array[:, 0].astype(np.int64), boxes, points, np.array(ids, dtype=np.int64))


def label_bytes(labels: Labels, width: int, height: int, precision: int) -> bytes:
    lines: list[str] = []
    for index, class_id in enumerate(labels.class_ids):
        x0, y0, x1, y1 = labels.boxes[index]
        coordinates = [(x0+x1)/(2*width), (y0+y1)/(2*height),
                       (x1-x0)/width, (y1-y0)/height]
        fields = [str(int(class_id)), *(f"{value:.{precision}f}" for value in coordinates)]
        # Avoid rounded zero-size boxes at a user-requested low precision.
        if float(fields[3]) <= 0 or float(fields[4]) <= 0:
            raise ValueError("Label precision rounds a positive box to zero; increase labels.precision")
        for x, y, visibility in labels.keypoints[index]:
            fields.extend((f"{x/width:.{precision}f}", f"{y/height:.{precision}f}", str(int(visibility))))
        lines.append(" ".join(fields))
    return (("\n".join(lines)+"\n") if lines else "").encode()


def _mapping(value: JsonValue, count: int, label: str) -> list[JsonValue]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f"{label} requires an explicit {count}-point mapping")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in value):
        raise ValueError(f"{label} mapping must contain integer indices")
    indices = [int(index) for index in value if isinstance(index, int)]
    if sorted(indices) != list(range(count)) or any(indices[indices[i]] != i for i in range(count)):
        raise ValueError(f"{label} must be an involutive permutation")
    return list(value)


def _output_suffix(path: Path, encoding: JsonObject) -> str:
    mode = string(encoding, "format")
    return path.suffix.lower() if mode == "source" else "."+mode


def variant_name(item: InputImage, index: int, encoding: JsonObject) -> str:
    path = Path(item.relative)
    identity = hashlib.sha256(item.relative.encode()).hexdigest()[:12]
    return (path.parent / f"{path.stem}__{identity}__aug{index:04d}{_output_suffix(path, encoding)}").as_posix()


def _overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def preflight(config: AugmentConfig) -> DatasetPlan:
    from .geometry import filter_labels

    settings = config.settings
    dataset = section(settings, "dataset")
    source = Path(string(dataset, "source")).expanduser().resolve()
    output = Path(string(dataset, "output")).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output must not exist: {output}")
    yaml_path = source / "dataset.yaml" if source.is_dir() else source
    if not yaml_path.is_file() or yaml_path.suffix.lower() not in (".yaml", ".yml"):
        raise ValueError(f"Expected dataset directory or local YAML: {source}")
    yaml_content = yaml_path.read_bytes()
    metadata = obj(json_value(yaml.safe_load(yaml_content.decode("utf-8-sig"))))
    task = string(dataset, "task")
    declared = metadata.get("task")
    if declared is not None and declared not in ("detect", "pose"):
        raise ValueError(f"Unsupported dataset task: {declared}")
    shape = metadata.get("kpt_shape")
    if task == "auto":
        task = "pose" if shape is not None else "detect"
    if declared is not None and declared != task:
        raise ValueError("Dataset task declaration contradicts labels/kpt_shape")
    keypoint_count = 0
    if task == "pose":
        if (not isinstance(shape, list) or len(shape) != 2 or isinstance(shape[0], bool)
                or not isinstance(shape[0], int) or shape[0] < 1
                or not isinstance(shape[1], int) or shape[1] != 3):
            raise ValueError("Pose requires kpt_shape: [K, 3], K >= 1")
        keypoint_count = shape[0]
    elif shape is not None:
        raise ValueError("Detection input must not declare keypoints")
    names = metadata.get("names")
    if isinstance(names, list) and names and all(isinstance(name, str) for name in names):
        class_ids = set(range(len(names)))
    elif isinstance(names, dict) and names and all(isinstance(name, str) for name in names.values()):
        try:
            class_ids = {int(key) for key in names}
        except ValueError as error:
            raise ValueError("names keys must be class integers") from error
        if class_ids != set(range(len(names))) or any(str(int(key)) != key for key in names):
            raise ValueError("names class IDs must be contiguous from zero")
    else:
        raise ValueError("dataset.yaml requires nonempty class names")
    declared_count = metadata.get("nc", len(class_ids))
    if isinstance(declared_count, bool) or not isinstance(declared_count, int) or declared_count != len(class_ids):
        raise ValueError("Dataset nc differs from names")
    point_names = metadata.get("kpt_names")
    if point_names is not None:
        if not keypoint_count:
            raise ValueError("Detection input must not declare kpt_names")
        groups: list[JsonValue] = [point_names]
        if isinstance(point_names, dict):
            if any(key not in {str(index) for index in class_ids} for key in point_names):
                raise ValueError("kpt_names contains unknown class IDs")
            groups = list(point_names.values())
        for group in groups:
            if not isinstance(group, list) or len(group) != keypoint_count or not all(isinstance(name, str) and name for name in group):
                raise ValueError("kpt_names must contain K nonempty names per declared group")
    root_value = metadata.get("path", ".")
    if not isinstance(root_value, str):
        raise ValueError("Dataset path must be a local directory")
    root = (yaml_path.parent / root_value).resolve()
    if _overlap(output, root) or yaml_path.is_relative_to(output):
        raise ValueError("Output overlaps dataset input")
    # Resolve flip metadata across all configured splits before touching output.
    import copy
    split_settings = copy.deepcopy(config.split_settings)
    horizontal_mappings: list[list[JsonValue]] = []
    if keypoint_count and metadata.get("flip_idx") is not None:
        _mapping(metadata["flip_idx"], keypoint_count, "dataset flip_idx")
    for split_cfg in split_settings.values():
        transforms = section(split_cfg, "transforms")
        for axis in ("horizontal_flip", "vertical_flip"):
            flip = section(transforms, axis)
            mapping = flip.get("keypoint_indices")
            if keypoint_count and axis == "horizontal_flip" and mapping is not None:
                horizontal_mappings.append(_mapping(mapping, keypoint_count, axis))
            if keypoint_count and mapping is None and axis == "horizontal_flip":
                mapping = metadata.get("flip_idx")
            if keypoint_count and mapping is not None:
                flip["keypoint_indices"] = _mapping(mapping, keypoint_count, axis)
            elif keypoint_count and boolean(split_cfg, "enabled") and boolean(flip, "enabled"):
                raise ValueError(f"{axis} enabled without keypoint mapping")
        target_classes = section(section(transforms, "crop"), "box").get("target_classes")
        if target_classes is not None and any(value not in class_ids for value in values(section(section(transforms, "crop"), "box"), "target_classes")):
            raise ValueError("crop.box.target_classes contains unknown class")
    if horizontal_mappings:
        if any(mapping != horizontal_mappings[0] for mapping in horizontal_mappings):
            raise ValueError("Horizontal keypoint mapping must be consistent across splits")
        metadata["flip_idx"] = horizontal_mappings[0]
        for split_cfg in split_settings.values():
            section(section(split_cfg, "transforms"), "horizontal_flip")["keypoint_indices"] = horizontal_mappings[0]
    generation = section(settings, "generation")
    encoding = section(settings, "encoding")
    items: list[InputImage] = []
    counts: JsonObject = {}
    seen_sources: set[Path] = set()
    for split in ("train", "val", "test"):
        entry = metadata.get(split)
        if entry is None:
            continue
        if not isinstance(entry, str):
            raise ValueError(f"{split} must be a directory, not a list or image manifest")
        image_dir = (root / entry).resolve()
        if not image_dir.is_dir() or image_dir.name != split or image_dir.parent.name != "images":
            raise ValueError(f"Expected images/{split} directory: {image_dir}")
        label_dir = image_dir.parent.parent / "labels" / split
        if _overlap(output, image_dir) or _overlap(output, label_dir.resolve()):
            raise ValueError("Output overlaps a split input directory")
        cfg = split_settings[split]
        paths = sorted(path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            raise ValueError(f"Empty declared split: {split}")
        stems = [path.relative_to(image_dir).with_suffix("").as_posix() for path in paths]
        if len(stems) != len(set(stems)):
            raise ValueError(f"Duplicate image stems in {split}")
        extras = {p.relative_to(label_dir).with_suffix("").as_posix() for p in label_dir.rglob("*.txt")} - set(stems)
        if extras:
            raise ValueError(f"Labels without images in {split}: {sorted(extras)[:5]}")
        fraction = cfg["fraction"]
        assert isinstance(fraction, (int, float))
        selected_count = math.floor(len(paths)*fraction+0.5) if boolean(cfg, "enabled") else 0
        base_seed = integer(cfg, "seed")
        ranking = sorted(paths, key=lambda p: (derive_seed(base_seed, split, p.relative_to(image_dir).as_posix(), "select"), str(p)))
        selected = set(ranking[:selected_count])
        expected_names = set(path.relative_to(image_dir).with_suffix("").as_posix() for path in paths)
        split_items: list[InputImage] = []
        missing_count = 0
        for path in paths:
            resolved = path.resolve()
            if resolved in seen_sources:
                raise ValueError(f"Input image shared between splits: {path}")
            seen_sources.add(resolved)
            if _overlap(output, resolved.parent):
                raise ValueError("Output overlaps resolved input image")
            relative = path.relative_to(image_dir)
            label = label_dir / relative.with_suffix(".txt")
            if not label.is_file():
                if not boolean(dataset, "allow_missing_labels"):
                    raise FileNotFoundError(f"Missing label: {label}")
                missing_count += 1
                label_path = None
            else:
                label_path = label
                if _overlap(output, label.resolve().parent):
                    raise ValueError("Output overlaps resolved label")
            content = path.read_bytes()
            pixels = decode_image(content, path)
            height, width = pixels.shape[:2]
            label_content = label.read_bytes() if label_path else b""
            labels = parse_labels(label_content, label, width, height, keypoint_count, class_ids)
            if path in selected and len(labels.class_ids) and not len(filter_labels(labels, section(cfg, "labels")).class_ids):
                raise ValueError(f"Positive input fails baseline label thresholds: {label}")
            resize = section(section(cfg, "transforms"), "resize")["size"]
            if path in selected and len(labels.class_ids) and isinstance(resize, list):
                target_width = integer({"v": resize[0]}, "v")
                target_height = integer({"v": resize[1]}, "v")
                factor = min(target_width/width, target_height/height)
                resized = labels.copy()
                resized.boxes[:, [0, 2]] *= max(1, round(width*factor))/width
                resized.boxes[:, [1, 3]] *= max(1, round(height*factor))/height
                if not len(filter_labels(resized, section(cfg, "labels")).class_ids):
                    raise ValueError(f"Positive fallback fails configured resize thresholds: {label}")
            item = InputImage(split, relative.as_posix(), path, label_path, digest(content), digest(label_content), path in selected, width, height)
            if item.selected:
                for index in range(integer(generation, "copies_per_image")):
                    name = Path(variant_name(item, index, encoding)).with_suffix("").as_posix()
                    if name in expected_names:
                        raise ValueError(f"Output filename collision: {name}")
                    expected_names.add(name)
            split_items.append(item)
        items.extend(split_items)
        count = len(paths)+selected_count*integer(generation, "copies_per_image")
        if not boolean(generation, "keep_originals"):
            count -= selected_count
        counts[split] = {"input": len(paths), "selected": selected_count,
                         "output": count, "missing_labels": missing_count}
    if not items:
        raise ValueError("No declared train/val/test image directories")
    return DatasetPlan(yaml_path, digest(yaml_content), output, items, metadata, split_settings, counts, keypoint_count, class_ids)


def output_metadata(plan: DatasetPlan) -> JsonObject:
    allowed = ("names", "nc", "kpt_shape", "kpt_names", "flip_idx")
    result: JsonObject = {key: value for key, value in plan.metadata.items() if key in allowed}
    # Without `path`, Ultralytics resolves splits against the YAML directory.
    # A relative path='.' is instead resolved against its global datasets_dir.
    for split in plan.counts:
        result[split] = f"images/{split}"
    return result
