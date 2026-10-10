from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import TypeAlias, TypedDict, cast

import albumentations as A
import numpy as np
from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS
from numpy.typing import NDArray

from nn.utils import ConfigMap, ConfigValue

Augmentation: TypeAlias = A.BasicTransform | A.BaseCompose


class ImageResults(TypedDict):
    """The image field used here; MMPose may carry additional untouched fields."""

    img: NDArray[np.uint8]


@TRANSFORMS.register_module()
class PixelAlbumentation(BaseTransform):
    """Apply Albumentations 2 pixel transforms to an MMPose image."""

    def __init__(self, transforms: list[ConfigMap]) -> None:
        self.transforms = deepcopy(transforms)
        self.augmentation = A.Compose(
            [self._build_transform(config) for config in self.transforms]
        )

    def _build_transform(self, config: ConfigMap) -> Augmentation:
        args: dict[str, ConfigValue | list[Augmentation]] = deepcopy(dict(config))
        transform_type = args.pop("type")
        if not isinstance(transform_type, str):
            raise TypeError("Albumentations transform type must be a string")
        if "transforms" in args:
            children = cast(list[ConfigMap], args["transforms"])
            args["transforms"] = [self._build_transform(child) for child in children]
        constructor = cast(Callable[..., Augmentation], getattr(A, transform_type))
        return constructor(**args)

    def transform(self, results: ImageResults) -> ImageResults:
        results["img"] = self.augmentation(image=results["img"])["image"]
        return results
