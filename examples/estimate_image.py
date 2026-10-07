"""Run the callable perception library on one calibrated image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from perception import OpenVINORecognizer, PoseEstimator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--yolo-model", type=Path, required=True)
    parser.add_argument("--hrnet-model", type=Path, required=True)
    parser.add_argument("--camera-matrix", type=float, nargs=9, required=True,
                        help="Nine row-major entries of the original image's calibrated K")
    parser.add_argument("--distortion", type=float, nargs="+", default=None,
                        help="OpenCV distortion coefficients; omitted means zero distortion")
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"Cannot decode image: {args.image}")
    camera_matrix = np.asarray(args.camera_matrix, dtype=np.float64).reshape(3, 3)
    distortion = (None if args.distortion is None
                  else np.asarray(args.distortion, dtype=np.float64))
    recognizer = OpenVINORecognizer(yolo_model=args.yolo_model, hrnet_model=args.hrnet_model)
    solver = PoseEstimator()
    detections = recognizer.predict(image)
    print(f"detections: {len(detections)}")
    for index, detection in enumerate(detections):
        result = solver.estimate(detection, camera_matrix, distortion)
        for name, body in (("station", result.station), ("pillar", result.pillar)):
            print(f"target={index} body={name} success={body.success} "
                  f"reason={body.failure_reason} error_px={body.inlier_mean_error_px} "
                  f"planar={body.planar_geometry}")
            if body.success:
                print(body.transform)


if __name__ == "__main__":
    main()
