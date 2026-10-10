"""COCO keypoint records shared by conversion and visualization tools."""
from __future__ import annotations

from typing import TypedDict
from typing_extensions import NotRequired


class CocoInstance(TypedDict):
    bbox: list[float]
    area: float
    num_keypoints: int
    keypoints: list[float]
    num_keypoints_in_frame: NotRequired[int]
    keypoints_raw_visibility: NotRequired[list[int]]
    keypoints_in_frame: NotRequired[list[int]]


class CocoAnnotation(CocoInstance):
    id: int
    image_id: int
    category_id: int
    iscrowd: int


class CocoImage(TypedDict):
    id: int
    file_name: str
    width: int
    height: int


class CocoCategory(TypedDict):
    id: int
    name: str
    supercategory: str
    keypoints: list[str]
    skeleton: list[list[int]]


class CocoDataset(TypedDict):
    images: list[CocoImage]
    annotations: list[CocoAnnotation]
    categories: list[CocoCategory]
