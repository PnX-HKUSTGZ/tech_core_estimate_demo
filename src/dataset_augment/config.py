"""Validated YAML and command-line configuration for offline augmentation.

Configuration precedence is preset, YAML globals, YAML split overrides, then
command-line overrides. Relative YAML paths refer to the configuration folder.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import yaml

from .types import JsonObject, JsonValue, boolean, derive_seed, integer, json_value, obj, section, string, values


@dataclass(frozen=True)
class AugmentConfig:
    """Validated global controls and independently resolved split settings."""

    settings: JsonObject
    split_settings: dict[str, JsonObject]


def _transforms(preset: str = "moderate") -> JsonObject:
    mild = preset == "mild"
    return {
        "crop": {"enabled": not mild, "p": 0.30, "mode": "choice", "choice_weights": {"random": 0.60, "box": 0.40},
                 "random": {"width_ratio": [0.65, 1.0], "height_ratio": [0.65, 1.0]},
                 "box": {"target_classes": None, "scale": [1.4, 3.2], "center_jitter": 0.15, "min_crop_size_ratio": 0.25}},
        "affine": {"enabled": not mild, "p": 0.50, "degrees": [-10.0, 10.0], "scale": [0.90, 1.10], "translate_x": [-0.05, 0.05], "translate_y": [-0.05, 0.05]},
        "horizontal_flip": {"enabled": False, "p": 0.30, "keypoint_indices": None},
        "vertical_flip": {"enabled": False, "p": 0.10, "keypoint_indices": None},
        "resize": {"size": "source"},
        "geometry": {"interpolation": "linear", "fill_rgb": [114, 114, 114]},
        "occlusion": {"enabled": not mild, "p": 0.20, "count": [1, 2], "width_ratio": [0.04, 0.14], "height_ratio": [0.04, 0.14],
                      "keypoint_anchor_p": 0.80, "anchor_jitter": 0.25, "fill_mode": "random_color", "fill_rgb": [0, 0, 0],
                      "color_base": [20, 230], "color_jitter": [-20, 20], "max_bbox_overlap": 1.0, "max_attempts": 20},
        "exposure": {"enabled": True, "p": 1.0, "ev": [-0.15, 0.15] if mild else [-0.35, 0.35]},
        "white_balance": {"enabled": True, "p": 1.0, **{f"{c}_gain": [0.96, 1.04] if mild else [0.90, 1.12] for c in ("r", "g", "b")}},
        "contrast": {"enabled": True, "p": 1.0, "factor": [0.95, 1.06] if mild else [0.86, 1.18], "center": 0.5},
        "gamma": {"enabled": True, "p": 1.0, "gamma": [0.94, 1.06] if mild else [0.85, 1.18]},
        "vignette": {"enabled": True, "p": 0.65, "strength": [0.0, 0.10] if mild else [0.0, 0.28],
                     "center_x": [0.47, 0.53] if mild else [0.42, 0.58], "center_y": [0.47, 0.53] if mild else [0.42, 0.58]},
        "blur": {"enabled": True, "weights": {"none": 0.70, "gaussian": 0.30, "motion": 0.0} if mild else {"none": 0.40, "gaussian": 0.35, "motion": 0.25},
                 "kernel_sizes": [3] if mild else [3, 5, 7], "kernel_weights": [1.0] if mild else [0.55, 0.35, 0.10],
                 "gaussian_sigma": [0.35, 0.75] if mild else [0.5, 1.6], "motion_angle": [0.0, 180.0]},
        "noise": {"enabled": True, "p": 0.65, "sigma_uint8": [0.0, 3.0] if mild else [0.0, 7.0]},
        "jpeg": {"enabled": True, "p": 1.0, "quality": [85, 100] if mild else [70, 100]},
    }


def _labels() -> JsonObject:
    return {"min_bbox_area_ratio": 0.10, "min_bbox_width_px": 2.0, "min_bbox_height_px": 2.0,
            "min_keypoints": 0, "empty_positive_policy": "retry", "max_geometry_attempts": 10, "precision": 8}


def _split(name: str) -> JsonObject:
    return {"enabled": name == "train", "fraction": 1.0, "preset": "moderate" if name == "train" else "mild", "seed": None, "overrides": {}}


def _defaults() -> JsonObject:
    return {
        "dataset": {"source": None, "output": None, "task": "auto", "allow_missing_labels": False},
        "generation": {"copies_per_image": 1, "keep_originals": True, "seed": 20261003},
        "transforms": _transforms(), "labels": _labels(),
        "splits": {name: _split(name) for name in ("train", "val", "test")},
        "encoding": {"format": "png", "jpeg_quality": 95, "png_compression": 3},
        "runtime": {"workers": 4, "opencv_threads": 1, "max_in_flight": None, "progress_every": 100},
        "logging": {"level": "INFO", "max_bytes": 10 * 1024 * 1024, "backup_count": 3},
        "preview": {"count_per_split": 8, "overlay_labels": True, "tile_width": 640},
    }


def _merge(base: JsonObject, patch: JsonObject) -> JsonObject:
    result = copy.deepcopy(base)
    for key, value in patch.items():
        previous = result.get(key)
        if isinstance(previous, dict) and isinstance(value, dict):
            result[key] = _merge(previous, value)
        else:
            detached: JsonValue = copy.deepcopy(value)
            result[key] = detached
    return result


def _check_partial(layer: JsonObject, schema: JsonObject, prefix: str = "") -> None:
    for key, value in layer.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in schema:
            raise ValueError(f"Unknown configuration field: {path}")
        expected = schema[key]
        if path == "splits":
            for name, spec in obj(value).items():
                if name not in ("train", "val", "test"):
                    raise ValueError(f"Unsupported split name: {name}; expected train, val, or test")
                split_schema = _split(name)
                split_schema["overrides"] = {"transforms": _transforms(), "labels": _labels()}
                _check_partial(obj(spec), split_schema, path + "." + name)
        elif isinstance(expected, dict):
            _check_partial(obj(value), expected, path)
        elif isinstance(value, dict):
            raise ValueError(f"{path} is not a mapping field")


def _numeric(value: JsonValue, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{path} must be a finite number")
    return float(value)


def _bounded(mapping: JsonObject, key: str, low: float | None = None, high: float | None = None, *, strict_low: bool = False, integral: bool = False) -> float:
    value = mapping[key]
    if integral and (isinstance(value, bool) or not isinstance(value, int)):
        raise ValueError(f"{key} must be an integer")
    number_ = _numeric(value, key)
    if low is not None and (number_ <= low if strict_low else number_ < low):
        raise ValueError(f"{key} must be {'greater than' if strict_low else 'at least'} {low}")
    if high is not None and number_ > high:
        raise ValueError(f"{key} must be at most {high}")
    return number_


def _range(mapping: JsonObject, key: str, low: float | None = None, high: float | None = None, *, strict_low: bool = False, integral: bool = False) -> None:
    items = values(mapping, key)
    if len(items) != 2:
        raise ValueError(f"{key} must contain exactly two bounds")
    endpoints = [_bounded({key: item}, key, low, high, strict_low=strict_low, integral=integral) for item in items]
    if endpoints[0] > endpoints[1]:
        raise ValueError(f"{key} range must be ordered")


def _choice(mapping: JsonObject, key: str, options: tuple[str, ...]) -> None:
    if string(mapping, key) not in options:
        raise ValueError(f"{key} must be one of {', '.join(options)}")


def _rgb(mapping: JsonObject, key: str) -> None:
    channels = values(mapping, key)
    if len(channels) != 3:
        raise ValueError(f"{key} requires RGB triplet")
    for value in channels:
        _bounded({key: value}, key, 0, 255, integral=True)


def _weights(mapping: JsonObject) -> None:
    total = sum(_bounded(mapping, key, 0) for key in mapping)
    if total <= 0 or not math.isfinite(total):
        raise ValueError("Weights must have a finite positive sum")


def _validate_transforms(transforms: JsonObject) -> None:
    for name, value in transforms.items():
        spec = obj(value)
        if name not in ("resize", "geometry"):
            boolean(spec, "enabled")
            if name != "blur":
                _bounded(spec, "p", 0, 1)
    crop = section(transforms, "crop")
    _choice(crop, "mode", ("random", "box", "choice"))
    _weights(section(crop, "choice_weights"))
    random = section(crop, "random")
    for key in ("width_ratio", "height_ratio"):
        _range(random, key, 0, 1, strict_low=True)
    box = section(crop, "box")
    _range(box, "scale", 0, strict_low=True)
    _bounded(box, "center_jitter", 0)
    _bounded(box, "min_crop_size_ratio", 0, 1, strict_low=True)
    if box["target_classes"] is not None:
        classes = values(box, "target_classes")
        for cls in classes:
            _bounded({"target_classes": cls}, "target_classes", 0, integral=True)
        if len(set(classes)) != len(classes):
            raise ValueError("target_classes must be unique")
    affine = section(transforms, "affine")
    _range(affine, "degrees")
    _range(affine, "scale", 0, strict_low=True)
    for key in ("translate_x", "translate_y"):
        _range(affine, key)
    for name in ("horizontal_flip", "vertical_flip"):
        flip = section(transforms, name)
        if flip["keypoint_indices"] is not None:
            mapping = values(flip, "keypoint_indices")
            for value in mapping:
                _bounded({"keypoint_indices": value}, "keypoint_indices", 0, integral=True)
            indices = [integer({"i": value}, "i") for value in mapping]
            if sorted(indices) != list(range(len(indices))) or any(indices[index] != i for i, index in enumerate(indices)):
                raise ValueError(f"{name}.keypoint_indices must be an involutive permutation")
    size = section(transforms, "resize")["size"]
    if size != "source":
        if not isinstance(size, list) or len(size) != 2:
            raise ValueError("resize.size must be source or [width,height]")
        for item in size:
            _bounded({"size": item}, "size", 1, integral=True)
    geometry = section(transforms, "geometry")
    _choice(geometry, "interpolation", ("nearest", "linear", "cubic"))
    _rgb(geometry, "fill_rgb")
    occ = section(transforms, "occlusion")
    _range(occ, "count", 1, integral=True)
    for key in ("width_ratio", "height_ratio"):
        _range(occ, key, 0, 1, strict_low=True)
    _bounded(occ, "keypoint_anchor_p", 0, 1)
    _bounded(occ, "anchor_jitter", 0)
    _choice(occ, "fill_mode", ("random_color", "constant", "noise"))
    _rgb(occ, "fill_rgb")
    _range(occ, "color_base", 0, 255, integral=True)
    _range(occ, "color_jitter", -255, 255, integral=True)
    _bounded(occ, "max_bbox_overlap", 0, 1)
    _bounded(occ, "max_attempts", 1, integral=True)
    _range(section(transforms, "exposure"), "ev")
    for key in ("r_gain", "g_gain", "b_gain"):
        _range(section(transforms, "white_balance"), key, 0, strict_low=True)
    contrast = section(transforms, "contrast")
    _range(contrast, "factor", 0, strict_low=True)
    _bounded(contrast, "center", 0, 1)
    _range(section(transforms, "gamma"), "gamma", 0, strict_low=True)
    vignette = section(transforms, "vignette")
    for key in ("strength", "center_x", "center_y"):
        _range(vignette, key, 0, 1)
    blur = section(transforms, "blur")
    _weights(section(blur, "weights"))
    kernels, weights = values(blur, "kernel_sizes"), values(blur, "kernel_weights")
    if not kernels or len(kernels) != len(weights):
        raise ValueError("blur kernel_sizes and kernel_weights need equal nonzero lengths")
    for value in kernels:
        k = _bounded({"kernel_sizes": value}, "kernel_sizes", 1, integral=True)
        if k % 2 == 0:
            raise ValueError("Blur kernels must be odd")
    _weights({str(i): weight for i, weight in enumerate(weights)})
    _range(blur, "gaussian_sigma", 0, strict_low=True)
    _range(blur, "motion_angle")
    _range(section(transforms, "noise"), "sigma_uint8", 0)
    _range(section(transforms, "jpeg"), "quality", 1, 100, integral=True)


def _validate_labels(labels: JsonObject) -> None:
    _bounded(labels, "min_bbox_area_ratio", 0, 1)
    for key in ("min_bbox_width_px", "min_bbox_height_px"):
        _bounded(labels, key, 0, strict_low=True)
    _bounded(labels, "min_keypoints", 0, integral=True)
    _choice(labels, "empty_positive_policy", ("retry", "allow"))
    _bounded(labels, "max_geometry_attempts", 1, integral=True)
    _bounded(labels, "precision", 1, 17, integral=True)


def _validate(settings: JsonObject) -> None:
    dataset = section(settings, "dataset")
    for key in ("source", "output"):
        if dataset[key] is not None and (not isinstance(dataset[key], str) or not dataset[key]):
            raise ValueError(f"dataset.{key} must be a nonempty path or null")
    _choice(dataset, "task", ("auto", "detect", "pose"))
    boolean(dataset, "allow_missing_labels")
    generation = section(settings, "generation")
    _bounded(generation, "copies_per_image", 1, integral=True)
    boolean(generation, "keep_originals")
    _seed(generation, "seed")
    encoding = section(settings, "encoding")
    _choice(encoding, "format", ("png", "jpg", "source"))
    _bounded(encoding, "jpeg_quality", 1, 100, integral=True)
    _bounded(encoding, "png_compression", 0, 9, integral=True)
    runtime = section(settings, "runtime")
    _bounded(runtime, "workers", 1, 64, integral=True)
    for key in ("opencv_threads", "max_in_flight", "progress_every"):
        _bounded(runtime, key, 1, integral=True)
    logging = section(settings, "logging")
    _choice(logging, "level", ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"))
    _bounded(logging, "max_bytes", 1, integral=True)
    _bounded(logging, "backup_count", 1, integral=True)
    preview = section(settings, "preview")
    _bounded(preview, "count_per_split", 0, integral=True)
    _bounded(preview, "tile_width", 1, integral=True)
    boolean(preview, "overlay_labels")
    _validate_transforms(section(settings, "transforms"))
    _validate_labels(section(settings, "labels"))


def _seed(mapping: JsonObject, key: str) -> int:
    # Check integer endpoints directly: float conversion loses the uint64 boundary.
    result = integer(mapping, key)
    if not 0 <= result <= 2**64 - 1:
        raise ValueError(f"{key} must be an unsigned 64-bit integer")
    return result


def _override_layer(overrides: Sequence[str]) -> JsonObject:
    result: JsonObject = {}
    seen: set[str] = set()
    for expression in overrides:
        if "=" not in expression:
            raise ValueError(f"Override must have key=YAML-value form: {expression}")
        path, raw = expression.split("=", 1)
        components = path.split(".")
        if not all(components):
            raise ValueError(f"Invalid override path: {path}")
        if any(path == previous or path.startswith(previous + ".") or previous.startswith(path + ".") for previous in seen):
            raise ValueError(f"Duplicate or overlapping override: {path}")
        seen.add(path)
        cursor = result
        for component in components[:-1]:
            cursor = obj(cursor.setdefault(component, {}))
        cursor[components[-1]] = json_value(yaml.safe_load(raw))
    return result


def _resolve_paths(layer: JsonObject, base: Path) -> None:
    if "dataset" not in layer:
        return
    dataset = section(layer, "dataset")
    for key in ("source", "output"):
        value = dataset.get(key)
        if isinstance(value, str) and value:
            path = Path(value).expanduser()
            dataset[key] = str((base / path).resolve() if not path.is_absolute() else path.resolve())


def load_config(path: str | Path | None = None, *, overrides: Sequence[str] = (), source: str | Path | None = None,
                output: str | Path | None = None, workers: int | None = None, seed: int | None = None) -> AugmentConfig:
    """Load a YAML file and apply CLI-equivalent dotted overrides.

    Source/output may remain null for inspecting defaults. Dataset execution
    requires both paths. All other settings are fully validated here.
    """
    schema = _defaults()
    yaml_layer: JsonObject = {}
    if path is not None:
        config_path = Path(path).expanduser().resolve()
        loaded: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        yaml_layer = obj(json_value(loaded)) if loaded is not None else {}
        _resolve_paths(yaml_layer, config_path.parent)
    cli_layer = _override_layer(overrides)
    aliases: tuple[tuple[str, str, str | int | None], ...] = (
        ("dataset", "source", str(source) if source is not None else None),
        ("dataset", "output", str(output) if output is not None else None),
        ("runtime", "workers", workers), ("generation", "seed", seed),
    )
    for group, key, value in aliases:
        if value is not None:
            group_values = obj(cli_layer.setdefault(group, {}))
            if key in group_values:
                raise ValueError(f"Conflicting alias and --set for {group}.{key}")
            group_values[key] = value
    _resolve_paths(cli_layer, Path.cwd())
    for layer in (yaml_layer, cli_layer):
        _check_partial(layer, schema)
    settings = _merge(_merge(schema, yaml_layer), cli_layer)
    runtime = section(settings, "runtime")
    if runtime["max_in_flight"] is None:
        runtime["max_in_flight"] = 2 * integer(runtime, "workers")
    _validate(settings)
    resolved: dict[str, JsonObject] = {}
    splits = section(settings, "splits")
    for name, split_value in tuple(splits.items()):
        spec = _merge(_split(name), obj(split_value))
        splits[name] = spec
        boolean(spec, "enabled")
        _bounded(spec, "fraction", 0, 1)
        _choice(spec, "preset", ("moderate", "mild"))
        split_seed = derive_seed(_seed(section(settings, "generation"), "seed"), name) if spec["seed"] is None else _seed(spec, "seed")
        body: JsonObject = {"transforms": _transforms(string(spec, "preset")), "labels": _labels()}
        for key in ("transforms", "labels"):
            if key in yaml_layer:
                body[key] = _merge(section(body, key), section(yaml_layer, key))
        # YAML split overrides follow YAML globals. CLI global values follow both.
        yaml_spec = obj(section(yaml_layer, "splits").get(name, {})) if "splits" in yaml_layer else {}
        body = _merge(body, obj(yaml_spec.get("overrides", {})))
        for key in ("transforms", "labels"):
            if key in cli_layer:
                body[key] = _merge(section(body, key), section(cli_layer, key))
        cli_spec = obj(section(cli_layer, "splits").get(name, {})) if "splits" in cli_layer else {}
        body = _merge(body, obj(cli_spec.get("overrides", {})))
        _validate_transforms(section(body, "transforms"))
        _validate_labels(section(body, "labels"))
        resolved[name] = {"enabled": spec["enabled"], "fraction": spec["fraction"], "preset": spec["preset"], "seed": split_seed, **body}
    return AugmentConfig(settings=settings, split_settings=resolved)


def config_to_dict(config: AugmentConfig) -> JsonObject:
    """Return a detached JSON/YAML-serializable effective configuration."""
    result = copy.deepcopy(config.settings)
    result["resolved_splits"] = {name: copy.deepcopy(value) for name, value in config.split_settings.items()}
    return result
