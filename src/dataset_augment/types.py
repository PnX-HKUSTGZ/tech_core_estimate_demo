"""Small shared numerical and JSON contracts for standalone augmentation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

JsonValue: TypeAlias = "None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]"
JsonObject: TypeAlias = dict[str, JsonValue]
Image: TypeAlias = NDArray[np.uint8]
FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


@dataclass
class Labels:
    """Pixel-edge xyxy boxes and Kx(x,y,v) points; row IDs refer to input lines."""

    class_ids: IntArray
    boxes: FloatArray
    keypoints: FloatArray
    row_ids: IntArray

    def copy(self) -> Labels:
        return Labels(self.class_ids.copy(), self.boxes.copy(), self.keypoints.copy(), self.row_ids.copy())

    def select(self, mask: NDArray[np.bool_]) -> Labels:
        return Labels(self.class_ids[mask].copy(), self.boxes[mask].copy(),
                      self.keypoints[mask].copy(), self.row_ids[mask].copy())


def json_value(value: object) -> JsonValue:
    """Validate an untyped YAML/JSON boundary without propagating Any."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Nonfinite configuration/metadata value")
        return value
    if isinstance(value, list):
        return [json_value(item) for item in cast(list[object], value)]
    if isinstance(value, dict):
        result: JsonObject = {}
        for key, item in cast(dict[object, object], value).items():
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise ValueError("Mapping keys must be strings or integers")
            name = str(key)
            if name in result:
                raise ValueError(f"Duplicate normalized mapping key: {name}")
            result[name] = json_value(item)
        return result
    raise ValueError(f"Unsupported metadata value: {type(value).__name__}")


def obj(value: JsonValue) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError("Expected a configuration mapping")
    return value


def section(mapping: JsonObject, key: str) -> JsonObject:
    return obj(mapping[key])


def number(mapping: JsonObject, key: str) -> float:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{key} must be finite numeric")
    return float(value)


def integer(mapping: JsonObject, key: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def boolean(mapping: JsonObject, key: str) -> bool:
    value = mapping[key]
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be boolean")
    return value


def string(mapping: JsonObject, key: str) -> str:
    value = mapping[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def values(mapping: JsonObject, key: str) -> list[JsonValue]:
    value = mapping[key]
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def interval(mapping: JsonObject, key: str) -> tuple[float, float]:
    value = values(mapping, key)
    if len(value) != 2:
        raise ValueError(f"{key} needs a two-element range")
    proxy: JsonObject = {"lo": value[0], "hi": value[1]}
    return number(proxy, "lo"), number(proxy, "hi")


def derive_seed(base: int, *parts: str | int) -> int:
    payload = json.dumps([1, base, *parts], ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def rng_for(base: int, *parts: str | int) -> np.random.Generator:
    return np.random.default_rng(derive_seed(base, *parts))
