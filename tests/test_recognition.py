"""CPU integration tests with tiny generated OpenVINO graphs (no downloaded models)."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
from numpy.typing import NDArray
import openvino as ov
from openvino import opset13 as ops

from perception.recognition import OpenVINORecognizer
from perception.types import KEYPOINT_NAMES


class RecognitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(prefix="perception-recognition-")
        self.addCleanup(self._temporary.cleanup)
        self._directory = Path(self._temporary.name)
        self._counter = 0
        self._image = np.zeros((100, 200, 3), dtype=np.uint8)

    def _model(
        self,
        output: NDArray[np.float32],
        input_shape: tuple[int, ...],
        *,
        channel_means: bool = False,
    ) -> Path:
        self._counter += 1
        folder = self._directory / str(self._counter)
        folder.mkdir()
        path = folder / "model.xml"
        parameter = ops.parameter(input_shape, np.float32)
        result = ops.constant(output)
        if channel_means:
            means = ops.reduce_mean(parameter, [2, 3], keep_dims=True)
            repeated = ops.gather(means, np.arange(12, dtype=np.int64) % 3, 1)
            result = ops.broadcast(repeated, list(output.shape))
        model = ov.Model([result], [parameter])
        ov.save_model(model, path, compress_to_fp16=False)
        return path

    def _recognizer(
        self,
        rows: list[list[float]],
        *,
        heatmaps: NDArray[np.float32] | None = None,
        input_size: tuple[int, int] = (256, 256),
        bbox_padding: float = 1.10,
        crop_margin: float = 0.0,
        channel_means: bool = False,
        dynamic_batch: bool = False,
        yolo_rank_two: bool = False,
    ) -> OpenVINORecognizer:
        output = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
        yolo_path = self._model(output if yolo_rank_two else output[None], (1, 3, 64, 64))
        if heatmaps is None:
            heatmaps = np.zeros((1, 12, 4, 4), dtype=np.float32)
            heatmaps[:, :, 2, 1] = 0.9
            heatmaps[:, :, 2, 2] = 0.4  # Right neighbour: x shifts by +1/4 pixel.
            heatmaps[:, :, 1, 1] = 0.3  # Upper neighbour: y shifts by -1/4 pixel.
        hrnet_path = self._model(
            heatmaps, (-1 if dynamic_batch else 1, 3, input_size[1], input_size[0]),
            channel_means=channel_means,
        )
        return OpenVINORecognizer(
            yolo_model=yolo_path.parent, hrnet_model=hrnet_path,
            hrnet_input_size=input_size, bbox_padding=bbox_padding, crop_margin=crop_margin,
        )

    def test_no_class_one_detection_returns_empty(self) -> None:
        recognizer = self._recognizer([[1, 17, 32, 48, 0.99, 0], [1, 17, 32, 48, 0.2, 1]])
        self.assertEqual(recognizer.predict(self._image), [])

    def test_zero_rows_returns_empty(self) -> None:
        self.assertEqual(self._recognizer([], yolo_rank_two=True).predict(self._image), [])

    def test_multiple_targets_sorted_and_original_coordinates_restored(self) -> None:
        recognizer = self._recognizer([
            [9.6, 22.4, 35.2, 35.2, 0.60, 1],
            [35.2, 19.2, 60.8, 44.8, 0.95, 1],
            [1, 17, 32, 48, 0.99, 0],
        ])
        detections = recognizer.predict(self._image)
        self.assertEqual(len(detections), 2)
        self.assertGreater(detections[0].detection_score, detections[1].detection_score)
        detection = detections[1]
        self.assertEqual(detection.image_shape_hw, (100, 200))
        self.assertEqual(detection.keypoint_names, KEYPOINT_NAMES)
        np.testing.assert_allclose(detection.bbox_xyxy, (30, 20, 110, 60), atol=1e-5)
        # Square crop: centre=(70,40), span=88. Quarter-refined heatmap
        # position is (1.75/4,2.25/4), so the original pixel is (64.5,45.5).
        np.testing.assert_allclose(detection.keypoints_xy, np.tile((64.5, 45.5), (12, 1)), atol=1e-5)
        np.testing.assert_allclose(detection.keypoint_scores, 0.9)

    def test_edge_bbox_clipping_and_padded_affine(self) -> None:
        recognizer = self._recognizer([[-6.4, 12.8, 12.8, 25.6, 0.9, 1]])
        detection = recognizer.predict(self._image)[0]
        np.testing.assert_allclose(detection.bbox_xyxy, (0, 0, 40, 30), atol=1e-5)
        np.testing.assert_allclose(detection.keypoints_xy, np.tile((17.25, 17.75), (12, 1)), atol=1e-5)

    def test_non_square_hrnet_input_and_dynamic_batch(self) -> None:
        recognizer = self._recognizer(
            [[9.6, 22.4, 35.2, 35.2, 0.9, 1]], input_size=(128, 256), dynamic_batch=True,
        )
        detection = recognizer.predict(self._image)[0]
        # A 1:2 crop preserves aspect: span=(88,176) around (70,40).
        np.testing.assert_allclose(detection.keypoints_xy, np.tile((64.5, 51), (12, 1)), atol=1e-5)

    def test_crop_margin_does_not_change_returned_detector_box(self) -> None:
        recognizer = self._recognizer([[9.6, 22.4, 35.2, 35.2, 0.9, 1]], crop_margin=0.25)
        detection = recognizer.predict(self._image)[0]
        np.testing.assert_allclose(detection.bbox_xyxy, (30, 20, 110, 60), atol=1e-5)
        np.testing.assert_allclose(detection.keypoints_xy, np.tile((61.75, 48.25), (12, 1)), atol=1e-5)

    def test_hrnet_rgb_normalization_preserves_raw_peaks(self) -> None:
        image = np.full((256, 256, 3), (10, 20, 230), dtype=np.uint8)
        recognizer = self._recognizer(
            [[0, 0, 64, 64, 0.9, 1]], channel_means=True, bbox_padding=1.0,
        )
        scores = recognizer.predict(image)[0].keypoint_scores
        expected = (np.array((230, 20, 10)) - (123.675, 116.28, 103.53)) / (58.395, 57.12, 57.375)
        # CPU kernels may accumulate channel means with reduced precision.
        np.testing.assert_allclose(scores, np.tile(expected, 4), rtol=3e-4, atol=3e-4)

    def test_yolo_preprocessing_letterbox_rgb_and_range(self) -> None:
        parameter = ops.parameter([1, 3, 64, 64], np.float32)
        # A target's score is the red-channel image centre, proving RGB and /255.
        pixel = ops.gather(ops.reshape(parameter, [-1], False), [32 * 64 + 32], 0)
        row = ops.concat([ops.constant([0, 16, 64, 48], np.float32), pixel, ops.constant([1], np.float32)], 0)
        path = self._directory / "pixel.xml"
        ov.save_model(ov.Model([ops.reshape(row, [1, 6], False)], [parameter]), path, compress_to_fp16=False)
        heatmaps = np.zeros((1, 12, 4, 4), dtype=np.float32)
        hrnet_path = self._model(heatmaps, (1, 3, 256, 256))
        recognizer = OpenVINORecognizer(yolo_model=path, hrnet_model=hrnet_path)
        image = np.full((100, 200, 3), (10, 20, 230), dtype=np.uint8)
        self.assertAlmostEqual(recognizer.predict(image)[0].detection_score, 230 / 255, places=6)

    def test_malformed_model_output_shapes_rejected_at_initialization(self) -> None:
        correct_yolo = np.zeros((1, 1, 6), dtype=np.float32)
        correct_hrnet = np.zeros((1, 12, 4, 4), dtype=np.float32)
        cases = (
            (np.zeros((1, 6, 2), np.float32), correct_hrnet, "six columns"),
            (np.zeros((2, 1, 6), np.float32), correct_hrnet, "batch"),
            (np.zeros((6,), np.float32), correct_hrnet, "Nx6"),
            (correct_yolo, np.zeros((1, 5, 4, 4), np.float32), "1x12xHxW"),
        )
        for yolo, hrnet, message in cases:
            with self.subTest(message=message):
                yolo_path = self._model(yolo, (1, 3, 64, 64))
                hrnet_path = self._model(hrnet, (1, 3, 256, 256))
                with self.assertRaisesRegex(ValueError, message):
                    OpenVINORecognizer(yolo_model=yolo_path, hrnet_model=hrnet_path)

    def test_invalid_images_rejected(self) -> None:
        recognizer = self._recognizer([])
        for image in (
            np.zeros((10, 10, 3), np.float32), np.zeros((10, 10), np.uint8),
            np.zeros((0, 10, 3), np.uint8), np.zeros((10, 10, 4), np.uint8),
        ):
            with self.subTest(shape=image.shape, dtype=image.dtype), self.assertRaises(ValueError):
                recognizer.predict(image)  # type: ignore[arg-type]

    def test_invalid_prediction_values_rejected(self) -> None:
        for row, message in (
            ([20, 20, 10, 30, 0.9, 1], "increasing"),
            ([0, 0, 5, 5, 0.9, 1], "no area"),
            ([0, 16, 64, 48, float("nan"), 1], "nonfinite"),
            ([0, 16, 64, 48, 1.1, 1], "scores"),
            ([0, 16, 64, 48, 0.9, 1.5], "class IDs"),
        ):
            with self.subTest(row=row):
                recognizer = self._recognizer([row])
                with self.assertRaisesRegex(ValueError, message):
                    recognizer.predict(self._image)
        heatmaps = np.full((1, 12, 4, 4), np.nan, dtype=np.float32)
        recognizer = self._recognizer([[0, 16, 64, 48, 0.9, 1]], heatmaps=heatmaps)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            recognizer.predict(self._image)

    def test_targets_keep_per_detection_inference_and_error_order(self) -> None:
        # The first target reaches HRNet before the next target's box is read.
        # Eagerly collecting all boxes during refactoring changes this error.
        heatmaps = np.full((1, 12, 4, 4), np.nan, dtype=np.float32)
        recognizer = self._recognizer(
            [[0, 16, 64, 48, 0.9, 1], [20, 20, 10, 30, 0.8, 1]],
            heatmaps=heatmaps,
        )
        with self.assertRaisesRegex(ValueError, "LiteHRNet heatmaps contain nonfinite"):
            recognizer.predict(self._image)

    def test_missing_weights_ambiguous_paths_and_unavailable_devices(self) -> None:
        yolo_path = self._model(np.zeros((1, 1, 6), np.float32), (1, 3, 64, 64))
        hrnet_path = self._model(np.zeros((1, 12, 4, 4), np.float32), (1, 3, 256, 256))
        for device in ("cuda:0", "AUTO", "GPU.UNAVAILABLE"):
            with self.subTest(device=device), self.assertRaisesRegex(ValueError, "unavailable"):
                OpenVINORecognizer(yolo_model=yolo_path, hrnet_model=hrnet_path, yolo_device=device)
        with self.assertRaisesRegex(ValueError, "input size mismatch"):
            OpenVINORecognizer(yolo_model=yolo_path, hrnet_model=hrnet_path, hrnet_input_size=(128, 128))
        with self.assertRaisesRegex(ValueError, "float32"):
            parameter = ops.parameter([1, 3, 64, 64], np.uint8)
            typed_path = self._directory / "uint8.xml"
            ov.save_model(ov.Model([ops.constant(np.zeros((1, 6), np.float32))], [parameter]), typed_path)
            OpenVINORecognizer(yolo_model=typed_path, hrnet_model=hrnet_path)
        (yolo_path.parent / "second.xml").write_text("unused", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly one XML"):
            OpenVINORecognizer(yolo_model=yolo_path.parent, hrnet_model=hrnet_path)
        hrnet_path.with_suffix(".bin").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "weights not found"):
            OpenVINORecognizer(yolo_model=yolo_path, hrnet_model=hrnet_path)


if __name__ == "__main__":
    unittest.main()
