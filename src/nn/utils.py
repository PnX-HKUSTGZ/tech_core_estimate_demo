"""Configuration types, validation and dynamic factories shared by NN tools."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import TypeAlias, cast

ConfigScalar: TypeAlias = str | int | float | bool | None
ConfigValue: TypeAlias = ConfigScalar | list["ConfigValue"] | dict[str, "ConfigValue"]
ConfigMap: TypeAlias = dict[str, ConfigValue]


def _as_config_value(value: object, context: str) -> ConfigValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_as_config_value(item, context) for item in cast(list[object], value)]
    if isinstance(value, dict):
        return as_config_mapping(value, context)
    raise ValueError(f"{context} contains an unsupported value: {type(value).__name__}")


def as_config_mapping(value: object, context: str = "Config") -> ConfigMap:
    """Validate a YAML/JSON mapping before it crosses into application code."""
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    result: ConfigMap = {}
    for key, item in cast(dict[object, object], value).items():
        if not isinstance(key, str):
            raise ValueError(f"{context} keys must be strings: {key!r}")
        result[key] = _as_config_value(item, f"{context}.{key}")
    return result


def config_section(config: Mapping[str, ConfigValue], name: str) -> ConfigMap:
    """Return a mapping section, treating an omitted or empty section as empty."""
    value = config.get(name) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a mapping")
    return value


def import_and_get(object_path: str) -> object:
    """Import an arbitrary object; callers narrow its type at the use site."""
    if "." not in object_path:
        raise ValueError(f"Expected a fully qualified object path, got: {object_path!r}")
    module_name, object_name = object_path.rsplit(".", 1)
    module = importlib.import_module(module_name, package=None)
    try:
        return cast(object, getattr(module, object_name))
    except AttributeError as exc:
        raise ImportError(f"{object_path!r} does not exist") from exc


def make_object(class_path: str, args: Mapping[str, object] | None = None) -> object:
    """Call a dynamically selected factory using keyword arguments."""
    factory = import_and_get(class_path)
    if not callable(factory):
        raise TypeError(f"Configured factory is not callable: {class_path!r}")
    constructor = cast(Callable[..., object], factory)
    return constructor() if args is None else constructor(**args)


def make_object_from_config(config: object) -> object:
    """Recursively instantiate `_class_name` entries in an open config tree.

    Unlike serialized configuration, the result may contain arbitrary imported
    objects, so its concrete type must be checked by the consumer.
    """
    if isinstance(config, list):
        return [make_object_from_config(item) for item in cast(list[object], config)]
    if isinstance(config, dict):
        mapping = cast(dict[str, object], config)
        if "_function_name" in mapping:
            if len(mapping) != 1:
                raise ValueError("_function_name cannot be combined with other keys")
            function_path = mapping["_function_name"]
            if not isinstance(function_path, str):
                raise ValueError("_function_name must be a fully qualified string")
            return import_and_get(function_path)
        if "_class_name" in mapping:
            class_path = mapping["_class_name"]
            if not isinstance(class_path, str):
                raise ValueError("_class_name must be a fully qualified string")
            return make_object(
                class_path,
                {
                    key: make_object_from_config(value)
                    for key, value in mapping.items()
                    if key != "_class_name"
                },
            )
        return {key: make_object_from_config(value) for key, value in mapping.items()}
    return config
