"""Ultralytics trainer adapters and DDP configuration handoff."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import cast

from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.pose import PoseTrainer

from nn.utils import ConfigMap, as_config_mapping
from nn.yolo.train.augmentations import (
    apply_detection_augmentations,
    apply_pose_augmentations,
)


def _store_worker_config(variable: str, config: ConfigMap) -> None:
    os.environ[variable] = json.dumps(config)


def _load_worker_config(variable: str) -> ConfigMap:
    value = os.environ.get(variable)
    return as_config_mapping(json.loads(value), variable) if value else {}


def _augment_training_dataset(
    dataset: YOLODataset,
    mode: str,
    variable: str,
    apply_augmentations: Callable[[YOLODataset, ConfigMap], YOLODataset],
) -> YOLODataset:
    """Apply the same DDP handoff for both concrete trainer adapters."""
    if mode == "train":
        apply_augmentations(dataset, _load_worker_config(variable))
    return dataset


class AugmentedPoseTrainer(PoseTrainer):
    augmentation_config_env = "RM26_POSE_AUGMENTATIONS"
    apply_augmentations = staticmethod(apply_pose_augmentations)

    def build_dataset(
        self, img_path: str, mode: str = "train", batch: int | None = None
    ) -> YOLODataset:
        dataset = cast(YOLODataset, super().build_dataset(img_path, mode, batch))
        return _augment_training_dataset(
            dataset, mode, self.augmentation_config_env, self.apply_augmentations
        )


class AugmentedDetectionTrainer(DetectionTrainer):
    augmentation_config_env = "RM26_DETECT_AUGMENTATIONS"
    apply_augmentations = staticmethod(apply_detection_augmentations)

    def build_dataset(
        self, img_path: str, mode: str = "train", batch: int | None = None
    ) -> YOLODataset:
        dataset = cast(YOLODataset, super().build_dataset(img_path, mode, batch))
        return _augment_training_dataset(
            dataset, mode, self.augmentation_config_env, self.apply_augmentations
        )


def create_pose_trainer(config: ConfigMap) -> type[AugmentedPoseTrainer]:
    """Configure DDP workers and return the pose trainer class."""
    _store_worker_config(AugmentedPoseTrainer.augmentation_config_env, config)
    return AugmentedPoseTrainer


def create_detection_trainer(config: ConfigMap) -> type[AugmentedDetectionTrainer]:
    """Configure DDP workers and return the detection trainer class."""
    _store_worker_config(AugmentedDetectionTrainer.augmentation_config_env, config)
    return AugmentedDetectionTrainer
