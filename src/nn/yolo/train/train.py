#!/usr/bin/env python3
"""Unified Ultralytics training entry point for pose and detection models."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

from nn.utils import ConfigMap, ConfigValue, as_config_mapping, config_section

if TYPE_CHECKING:
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.models.yolo.pose import PoseTrainer
    from ultralytics.utils.metrics import DetMetrics


TRAIN_DIR = Path(__file__).resolve().parent
YOLO_ROOT = TRAIN_DIR.parent
DEFAULT_CONFIG = YOLO_ROOT / "configs" / "train_pose.yaml"

CLI_OVERRIDES: dict[str, tuple[str, ...]] = {
    "task": ("task",),
    "model": ("model",),
    "data": ("data",),
    "resume": ("resume",),
    "epochs": ("train", "epochs"),
    "batch": ("train", "batch"),
    "imgsz": ("train", "imgsz"),
    "device": ("train", "device"),
    "workers": ("train", "workers"),
    "project": ("train", "project"),
    "name": ("train", "name"),
    "optimizer": ("train", "optimizer"),
    "export_openvino": ("export_openvino",),
    "random_crop": ("random_crop", "enabled"),
    "random_occlusion": ("random_occlusion", "enabled"),
    "albumentations": ("albumentations", "enabled"),
}


def _merge_config(base: ConfigMap, override: ConfigMap) -> ConfigMap:
    merged = dict(base)
    for key, value in override.items():
        previous = merged.get(key)
        if isinstance(value, dict) and isinstance(previous, dict):
            merged[key] = _merge_config(previous, value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path, stack: tuple[Path, ...] = ()) -> ConfigMap:
    import yaml

    path = path.resolve()
    if path in stack:
        raise ValueError(f"Circular config inheritance: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing training config: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = as_config_mapping(
            yaml.safe_load(file) or {}, f"Training config {path}"
        )

    base_name = config.pop("_base_", None)
    if not base_name:
        return config
    base_path = Path(_string(base_name, "_base_")).expanduser()
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    return _merge_config(_read_yaml(base_path, stack + (path,)), config)


def load_config(path: str | Path) -> ConfigMap:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    config = _read_yaml(config_path)
    config["_config_path"] = str(config_path)

    for key in ("data", "resume"):
        value = config.get(key)
        if value:
            filename = _string(value, key)
            if not Path(filename).expanduser().is_absolute():
                config[key] = str((config_path.parent / filename).resolve())

    training = _section(config, "train")
    project = training.get("project")
    if project:
        project_name = _string(project, "train.project")
        if not Path(project_name).expanduser().is_absolute():
            training["project"] = str((config_path.parent / project_name).resolve())
    return config


def _set_nested(config: ConfigMap, path: tuple[str, ...], value: ConfigValue) -> None:
    current = config
    for key in path[:-1]:
        section = current.setdefault(key, {})
        if not isinstance(section, dict):
            raise ValueError(f"Config section {key!r} must be a mapping")
        current = section
    current[path[-1]] = value


def apply_cli_overrides(config: ConfigMap, args: argparse.Namespace) -> ConfigMap:
    for argument, path in CLI_OVERRIDES.items():
        value = cast(ConfigValue, getattr(args, argument, None))
        if value is not None:
            _set_nested(config, path, value)
    return config


def _add_boolean_override(
    parser: argparse.ArgumentParser, name: str, help_text: str
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        f"--{name.replace('_', '-')}",
        dest=name,
        action="store_true",
        default=None,
        help=f"Enable {help_text}",
    )
    group.add_argument(
        f"--no-{name.replace('_', '-')}",
        dest=name,
        action="store_false",
        help=f"Disable {help_text}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--task", choices=("pose", "detect"))
    parser.add_argument("--model")
    parser.add_argument("--data")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--project")
    parser.add_argument("--name")
    parser.add_argument("--optimizer")
    _add_boolean_override(parser, "export_openvino", "OpenVINO export")
    _add_boolean_override(parser, "random_crop", "random crop augmentation")
    _add_boolean_override(
        parser, "random_occlusion", "random occlusion augmentation"
    )
    _add_boolean_override(parser, "albumentations", "Albumentations")
    return parser.parse_args()


# Keep the historical helper name for callers while sharing section validation.
_section = config_section


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Config field {name!r} must be a string")
    return value


def _model_path(config: ConfigMap) -> str:
    if config.get("resume"):
        return _string(config["resume"], "resume")
    default = "yolo26n-pose.pt" if config["task"] == "pose" else "yolo26n.pt"
    model = str(config.get("model", default))
    return model if model.endswith((".pt", ".yaml", ".yml")) else f"{model}.pt"


def _training_arguments(config: ConfigMap) -> dict[str, object]:
    from nn.yolo.train.augmentations import build_image_augmentations

    arguments: dict[str, object] = {
        "data": config.get("data"),
        **_section(config, "train"),
        **_section(config, "yolo_aug"),
        **_section(config, "loss"),
    }
    image_augmentations = build_image_augmentations(
        _section(config, "albumentations")
    )
    if image_augmentations is not None:
        arguments["augmentations"] = image_augmentations
    if config.get("resume"):
        arguments["resume"] = True
    return {key: value for key, value in arguments.items() if value is not None}


def _pose_trainer(config: ConfigMap) -> type[PoseTrainer] | None:
    augmentation_config: ConfigMap = {
        "random_crop": _section(config, "random_crop"),
        "random_occlusion": _section(config, "random_occlusion"),
    }
    if not any(
        _section(augmentation_config, name).get("enabled")
        for name in augmentation_config
    ):
        return None

    from nn.yolo.train.trainers import create_pose_trainer

    return create_pose_trainer(augmentation_config)


def _detect_trainer(config: ConfigMap) -> type[DetectionTrainer] | None:
    augmentation_config: ConfigMap = {
        name: _section(config, name)
        for name in (
            "random_crop",
            "box_crop",
            "crop_choice",
            "random_occlusion",
        )
    }
    if not any(
        _section(augmentation_config, name).get("enabled")
        for name in ("random_crop", "box_crop", "random_occlusion")
    ):
        return None

    from nn.yolo.train.trainers import create_detection_trainer

    return create_detection_trainer(augmentation_config)


def _configure_runtime() -> None:
    import cv2

    cv2.setNumThreads(0)
    try:
        cv2.ocl.setUseOpenCL(False)
    except AttributeError:
        pass


def train(config: ConfigMap) -> DetMetrics | DetectionTrainer | None:
    from ultralytics.models.yolo.model import YOLO

    task = _string(config.get("task", "pose"), "task")
    if task not in {"pose", "detect"}:
        raise ValueError("Config field 'task' must be 'pose' or 'detect'")
    config["task"] = task

    model_path = _model_path(config)
    arguments = _training_arguments(config)
    trainer_class = _pose_trainer(config) if task == "pose" else _detect_trainer(config)

    print(f"[train] task={task}, model={model_path}")
    print(
        f"[train] epochs={arguments.get('epochs')}, batch={arguments.get('batch')}, "
        f"imgsz={arguments.get('imgsz')}"
    )
    if trainer_class is None:
        result: DetMetrics | DetectionTrainer | None = cast(
            "DetMetrics | None", YOLO(model_path).train(**arguments)
        )
    else:
        arguments["model"] = model_path
        trainer = trainer_class(overrides=arguments)
        cast(Callable[[], None], trainer.train)()
        result = trainer

    project = Path(_string(arguments.get("project", "runs"), "train.project"))
    name = _string(arguments.get("name", task), "train.name")
    weights_dir = project / name / "weights"
    best_weights = weights_dir / "best.pt"
    if not best_weights.exists():
        best_weights = weights_dir / "last.pt"

    if best_weights.exists():
        print(f"[train] best checkpoint: {best_weights}")
    else:
        print(f"[train] warning: no checkpoint found in {weights_dir}")

    if config.get("export_openvino") and best_weights.exists():
        YOLO(str(best_weights)).export(
            format="openvino",
            imgsz=arguments.get("imgsz", 640),
            half=True,
            dynamic=False,
            nms=True,
        )
    return result


def main() -> None:
    _configure_runtime()
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    print(f"[train] config={config['_config_path']}")
    train(config)


if __name__ == "__main__":
    main()
