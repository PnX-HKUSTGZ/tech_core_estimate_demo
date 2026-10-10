"""
Convert YOLO Pose dataset to COCO keypoint format for MMPose.

Supported modes:

1. pillar5
   Keep the pillar crop and only export the first 5 keypoints:
   ['TL', 'TR', 'BL', 'BR', 'ring'].

2. exchange12
   Keep the exchange crop and export all 12 keypoints:
   5 pillar keypoints + 7 exchange keypoints.
   This mode pairs the pillar row and exchange row from the same YOLO label file.

Expected dataset layout:
    dataset_root/
        images/
            train/
            val/
        labels/
            train/
            val/
        dataset.yaml

Usage:
    python prepare_data.py /path/to/dataset_root --output annotations --mode both
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob

import cv2

from nn.hrnet.data.convert_yolo_to_coco import (
    EXCHANGE_KPT_NAMES as EXCHANGE_KPT_NAMES,
    EXCHANGE_SKELETON as EXCHANGE_SKELETON,
    FULL_KPT_NAMES, FULL_SKELETON, PILLAR_KPT_NAMES, PILLAR_SKELETON,
    LabelRecord as LabelRecord, clamp_bbox_xywh as clamp_bbox_xywh,
    find_image as find_image, parse_label_line as parse_label_line,
    yolo_bbox_to_coco_xywh as yolo_bbox_to_coco_xywh,
)
from nn.hrnet.data.schema import CocoAnnotation, CocoCategory, CocoDataset, CocoImage, CocoInstance

PILLAR_FLIP_IDX = [1, 0, 3, 2, 4]
FULL_FLIP_IDX = [1, 0, 3, 2, 4, 11, 10, 9, 8, 7, 6, 5]


def keypoints_to_coco(
    keypoints_norm: list[tuple[float, float, int]],
    img_w: int,
    img_h: int,
) -> tuple[list[float], int]:
    coco_kpts: list[float] = []
    valid_count = 0

    for kx, ky, kv in keypoints_norm:
        px = kx * img_w
        py = ky * img_h
        v = int(kv)
        if v > 0:
            valid_count += 1
        coco_kpts.extend([float(px), float(py), v])

    return coco_kpts, valid_count


def build_pillar5_instances(
    records: list[LabelRecord],
    img_w: int,
    img_h: int,
    pillar_class_id: int,
    keep_kpts: int,
) -> list[CocoInstance]:
    out: list[CocoInstance] = []
    for rec in records:
        if rec.cls_id != pillar_class_id:
            continue
        x, y, w, h = yolo_bbox_to_coco_xywh(rec, img_w, img_h)
        if w < 2 or h < 2:
            continue
        coco_kpts, valid_count = keypoints_to_coco(rec.all_kpts[:keep_kpts], img_w, img_h)
        out.append(
            {
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "num_keypoints": valid_count,
                "keypoints": coco_kpts,
            }
        )
    return out


def build_exchange12_instances(
    records: list[LabelRecord],
    img_w: int,
    img_h: int,
    pillar_class_id: int,
    exchange_class_id: int,
    pillar_kpts: int,
) -> tuple[list[CocoInstance], int]:
    out: list[CocoInstance] = []
    pending_pillars: list[LabelRecord] = []
    skipped_exchange = 0

    for rec in records:
        if rec.cls_id == pillar_class_id:
            pending_pillars.append(rec)
            continue

        if rec.cls_id != exchange_class_id:
            continue

        if not pending_pillars:
            skipped_exchange += 1
            continue

        pillar_rec = pending_pillars.pop(0)
        merged_kpts = pillar_rec.all_kpts[:pillar_kpts] + rec.all_kpts[pillar_kpts:]
        x, y, w, h = yolo_bbox_to_coco_xywh(rec, img_w, img_h)
        if w < 2 or h < 2:
            continue
        coco_kpts, valid_count = keypoints_to_coco(merged_kpts, img_w, img_h)
        out.append(
            {
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "num_keypoints": valid_count,
                "keypoints": coco_kpts,
            }
        )

    skipped_exchange += len(pending_pillars)
    return out, skipped_exchange


def process_split(
    image_dir: str,
    label_dir: str,
    output_json: str,
    mode: str,
    total_kpts: int = 12,
    pillar_class_id: int = 0,
    exchange_class_id: int = 1,
    keep_kpts: int = 5,
) -> None:
    images: list[CocoImage] = []
    annotations: list[CocoAnnotation] = []
    skipped_pairs = 0

    categories: list[CocoCategory]
    if mode == "pillar5":
        categories = [
            {
                "id": 1,
                "name": "pillar",
                "supercategory": "object",
                "keypoints": PILLAR_KPT_NAMES,
                "skeleton": PILLAR_SKELETON,
            }
        ]
    elif mode == "exchange12":
        categories = [
            {
                "id": 1,
                "name": "exchange_with_pillar",
                "supercategory": "object",
                "keypoints": FULL_KPT_NAMES,
                "skeleton": FULL_SKELETON,
            }
        ]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    label_files = sorted(glob(os.path.join(label_dir, "*.txt")))
    img_id = 1
    ann_id = 1

    for label_path in label_files:
        stem = os.path.splitext(os.path.basename(label_path))[0]
        img_path = find_image(image_dir, stem)
        if img_path is None:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        images.append(
            {
                "id": img_id,
                "file_name": os.path.basename(img_path),
                "width": img_w,
                "height": img_h,
            }
        )

        with open(label_path, "r") as f:
            raw_lines = [line.strip() for line in f if line.strip()]

        records: list[LabelRecord] = []
        for line in raw_lines:
            try:
                records.append(parse_label_line(line, total_kpts=total_kpts))
            except ValueError as exc:
                print(f"Warning: {label_path}: {exc}")

        if mode == "pillar5":
            instances = build_pillar5_instances(
                records=records,
                img_w=img_w,
                img_h=img_h,
                pillar_class_id=pillar_class_id,
                keep_kpts=keep_kpts,
            )
        else:
            instances, skipped = build_exchange12_instances(
                records=records,
                img_w=img_w,
                img_h=img_h,
                pillar_class_id=pillar_class_id,
                exchange_class_id=exchange_class_id,
                pillar_kpts=keep_kpts,
            )
            skipped_pairs += skipped

        for inst in instances:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": inst["bbox"],
                    "area": inst["area"],
                    "iscrowd": 0,
                    "num_keypoints": inst["num_keypoints"],
                    "keypoints": inst["keypoints"],
                }
            )
            ann_id += 1

        img_id += 1

    coco: CocoDataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco, f)

    print(f"Saved: {output_json}")
    print(f"  mode: {mode}")
    print(f"  images: {len(images)}")
    print(f"  annotations: {len(annotations)}")
    if mode == "exchange12":
        print(f"  skipped_unpaired_rows: {skipped_pairs}")


def run_mode(
    source: str,
    out_dir: str,
    split: str,
    mode: str,
    total_kpts: int,
    pillar_class_id: int,
    exchange_class_id: int,
    keep_kpts: int,
) -> None:
    prefix = "pillar" if mode == "pillar5" else "exchange12"
    process_split(
        image_dir=os.path.join(source, "images", split),
        label_dir=os.path.join(source, "labels", split),
        output_json=os.path.join(out_dir, f"{prefix}_{split}.json"),
        mode=mode,
        total_kpts=total_kpts,
        pillar_class_id=pillar_class_id,
        exchange_class_id=exchange_class_id,
        keep_kpts=keep_kpts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO Pose -> COCO keypoint json for MMPose"
    )
    parser.add_argument("source", help="YOLO Pose dataset root")
    parser.add_argument(
        "--output",
        default="annotations",
        help="output directory relative to source, or absolute path",
    )
    parser.add_argument(
        "--mode",
        choices=["pillar5", "exchange12", "both"],
        default="pillar5",
        help="pillar5 keeps current 5-kpt pillar crop; exchange12 uses exchange crop and all 12 keypoints",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="pillar class id (default 0)",
    )
    parser.add_argument(
        "--exchange-class-id",
        type=int,
        default=1,
        help="exchange class id (default 1)",
    )
    parser.add_argument(
        "--total-kpts",
        type=int,
        default=12,
        help="total keypoints per YOLO label row",
    )
    parser.add_argument(
        "--keep-kpts",
        type=int,
        default=5,
        help="pillar keypoint count; used by pillar5 and as the split point in exchange12",
    )
    args = parser.parse_args()

    if args.keep_kpts != len(PILLAR_KPT_NAMES):
        raise ValueError(
            f"--keep-kpts must be {len(PILLAR_KPT_NAMES)} for the current pillar definition"
        )
    if args.total_kpts != len(FULL_KPT_NAMES):
        raise ValueError(
            f"--total-kpts must be {len(FULL_KPT_NAMES)} for pillar5/exchange12 conversion"
        )

    source = os.path.abspath(args.source)
    out_dir = args.output if os.path.isabs(args.output) else os.path.join(source, args.output)
    os.makedirs(out_dir, exist_ok=True)

    modes = ["pillar5", "exchange12"] if args.mode == "both" else [args.mode]
    for mode in modes:
        for split in ("train", "val"):
            run_mode(
                source=source,
                out_dir=out_dir,
                split=split,
                mode=mode,
                total_kpts=args.total_kpts,
                pillar_class_id=args.class_id,
                exchange_class_id=args.exchange_class_id,
                keep_kpts=args.keep_kpts,
            )

    print("\nDone.")
    if "pillar5" in modes:
        print(f"pillar5 flip_idx: {PILLAR_FLIP_IDX}")
    if "exchange12" in modes:
        print(f"exchange12 flip_idx: {FULL_FLIP_IDX}")


if __name__ == "__main__":
    main()
