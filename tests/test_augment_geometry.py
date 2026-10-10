"""Analytic geometry checks independent of model or renderer outputs."""
from __future__ import annotations

import unittest

import numpy as np

from dataset_augment.config import load_config
from dataset_augment.geometry import _transform_labels, apply_geometry, filter_labels
from dataset_augment.photometric import apply_photometric
from dataset_augment.types import Image, JsonObject, Labels, obj, section, values


def settings() -> tuple[JsonObject, JsonObject]:
    config = load_config(overrides=("transforms.crop.enabled=false", "transforms.affine.enabled=false"))
    split = config.split_settings["train"]
    return section(split, "transforms"), section(split, "labels")


def labels(boxes: list[list[float]], points: list[list[list[float]]] | None = None,
           classes: list[int] | None = None) -> Labels:
    count = len(boxes)
    return Labels(np.asarray(classes if classes is not None else [0] * count, dtype=np.int64),
                  np.asarray(boxes, dtype=np.float64).reshape(count, 4),
                  np.asarray(points, dtype=np.float64) if points is not None else np.empty((count, 0, 3)),
                  np.arange(1, count + 1, dtype=np.int64))


def operation(audit: JsonObject, name: str) -> JsonObject:
    for item in values(audit, "operations"):
        entry = obj(item)
        if entry["operation"] == name:
            return entry
    raise AssertionError(f"Missing operation {name}")


class GeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image: Image = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
        self.transforms, self.filters = settings()

    def run_geometry(self, data: Labels, seed: int = 17) -> tuple[Image, Labels, JsonObject]:
        return apply_geometry(self.image, data, self.transforms, self.filters, seed, (100, 100))

    def test_disabled_operations_preserve_input_without_aliasing(self) -> None:
        data = labels([[10, 20, 40, 50]], [[[20, 30, 2], [0, 0, 0]]])
        before = data.copy()
        image, output, audit = self.run_geometry(data)
        np.testing.assert_array_equal(image, self.image)
        np.testing.assert_array_equal(output.boxes, before.boxes)
        np.testing.assert_array_equal(output.keypoints, before.keypoints)
        image[:] = 0
        output.boxes[:] = 0
        np.testing.assert_array_equal(data.boxes, before.boxes)
        self.assertTrue(np.any(self.image))
        self.assertFalse(any(obj(item)["applied"] for item in values(audit, "operations")))

    def test_visual_counterclockwise_warp_matches_points_and_pixels(self) -> None:
        image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        affine = section(self.transforms, "affine")
        affine.update({"enabled": True, "p": 1.0, "degrees": [90, 90], "scale": [1, 1],
                       "translate_x": [0, 0], "translate_y": [0, 0]})
        data = labels([[2, 1, 4, 3]], [[[3, 2, 2], [4, 0, 2], [0, 4, 1]]])
        output, transformed, _ = apply_geometry(image, data, self.transforms, self.filters, 1, (4, 4))
        np.testing.assert_array_equal(output, np.rot90(image))
        np.testing.assert_allclose(transformed.boxes, [[1, 0, 3, 2]], atol=1e-12)
        np.testing.assert_allclose(transformed.keypoints, [[[2, 1, 2], [0, 0, 2], [4, 4, 1]]], atol=1e-12)

    def test_joint_scale_translation_uses_original_center(self) -> None:
        section(self.transforms, "affine").update({"enabled": True, "p": 1.0, "degrees": [0, 0],
            "scale": [0.5, 0.5], "translate_x": [0.1, 0.1], "translate_y": [-0.1, -0.1]})
        data = labels([[20, 20, 80, 80]], [[[50, 50, 2], [0, 0, 0]]])
        _, transformed, audit = self.run_geometry(data)
        np.testing.assert_allclose(transformed.boxes, [[45, 25, 75, 55]])
        np.testing.assert_allclose(transformed.keypoints, [[[60, 40, 2], [0, 0, 0]]])
        self.assertEqual(operation(audit, "affine")["translation_px"], [10.0, -10.0])

    def test_polygon_clipping_uses_polygon_area_not_axis_aligned_box(self) -> None:
        data = labels([[0, 0, 10, 10]])
        c = np.sqrt(0.5)
        matrix = np.asarray([[c, c, 0], [-c, c, 0], [0, 0, 1]], dtype=np.float64)
        kept = _transform_labels(data, matrix, (10, 10), 0.41)
        removed = _transform_labels(data, matrix, (10, 10), 0.42)
        self.assertEqual(len(kept.boxes), 1)
        self.assertEqual(len(removed.boxes), 0)
        np.testing.assert_allclose(kept.boxes, [[0, 0, 10, np.sqrt(50)]])

    def test_box_crop_clips_points_immediately_and_hidden_points_do_not_revive(self) -> None:
        section(self.transforms, "crop").update({"enabled": True, "p": 1.0, "mode": "box"})
        section(section(self.transforms, "crop"), "box").update(
            {"scale": [1, 1], "center_jitter": 0, "min_crop_size_ratio": 0.1})
        section(self.transforms, "affine").update({"enabled": True, "p": 1.0, "degrees": [0, 0],
            "scale": [0.5, 0.5], "translate_x": [0, 0], "translate_y": [0, 0]})
        data = labels([[25, 25, 75, 75]], [[[20, 50, 2], [50, 50, 1], [0, 0, 0]]])
        _, output, audit = self.run_geometry(data)
        np.testing.assert_allclose(output.keypoints, [[[0, 0, 0], [50, 50, 1], [0, 0, 0]]])
        np.testing.assert_allclose(output.boxes, [[25, 25, 75, 75]])
        self.assertEqual(operation(audit, "crop")["xyxy"], [25, 25, 75, 75])
        self.assertEqual(len(values(audit, "visibility_changes")), 1)

    def test_random_crop_restores_source_canvas_and_reports_total_matrix(self) -> None:
        section(self.transforms, "crop").update({"enabled": True, "p": 1.0, "mode": "random"})
        section(section(self.transforms, "crop"), "random").update(
            {"width_ratio": [0.5, 0.5], "height_ratio": [0.5, 0.5]})
        data = labels([[0, 0, 100, 100]], [[[50, 50, 2]]])
        output, transformed, audit = self.run_geometry(data)
        self.assertEqual(output.shape, self.image.shape)
        rectangle = values(operation(audit, "crop"), "xyxy")
        x, y = float(str(rectangle[0])), float(str(rectangle[1]))
        np.testing.assert_allclose(transformed.boxes, [[0, 0, 100, 100]])
        expected_matrix = np.asarray([[2, 0, -2*x], [0, 2, -2*y], [0, 0, 1]], dtype=np.float64)
        np.testing.assert_allclose(np.asarray(audit["matrix"], dtype=np.float64), expected_matrix)
        second_image, second_labels, second_audit = self.run_geometry(data)
        np.testing.assert_array_equal(second_image, output)
        np.testing.assert_array_equal(second_labels.keypoints, transformed.keypoints)
        self.assertEqual(second_audit, audit)

    def test_box_crop_targets_class_and_skips_if_none_eligible(self) -> None:
        section(self.transforms, "crop").update({"enabled": True, "p": 1.0, "mode": "choice",
                                                  "choice_weights": {"random": 0, "box": 1}})
        section(section(self.transforms, "crop"), "box").update({"target_classes": [7]})
        data = labels([[10, 10, 30, 30], [70, 70, 90, 90]], classes=[0, 7])
        _, _, audit = self.run_geometry(data)
        self.assertEqual(operation(audit, "crop")["target_row_id"], 2)
        section(section(self.transforms, "crop"), "box")["target_classes"] = [8]
        output, transformed, audit = self.run_geometry(data)
        np.testing.assert_array_equal(output, self.image)
        np.testing.assert_array_equal(transformed.boxes, data.boxes)
        self.assertEqual(operation(audit, "crop")["skip_reason"], "no_eligible_box")

    def test_both_flips_twice_restore_boundaries_images_and_named_points(self) -> None:
        data = labels([[0, 0, 100, 100]], [[[0, 0, 2], [100, 100, 1], [50, 50, 2], [0, 0, 0]]])
        for name in ("horizontal_flip", "vertical_flip"):
            with self.subTest(name=name):
                transforms, filters = settings()
                section(transforms, name).update({"enabled": True, "p": 1.0,
                                                  "keypoint_indices": [1, 0, 3, 2]})
                once, once_labels, audit = apply_geometry(self.image, data, transforms, filters, 3, (100, 100))
                twice, twice_labels, _ = apply_geometry(once, once_labels, transforms, filters, 9, (100, 100))
                np.testing.assert_array_equal(twice, self.image)
                np.testing.assert_array_equal(twice_labels.boxes, data.boxes)
                np.testing.assert_array_equal(twice_labels.keypoints, data.keypoints)
                self.assertEqual(audit["visibility_changes"], [])

    def test_pose_flip_rejects_missing_or_noninvolutive_mapping(self) -> None:
        data = labels([[10, 10, 90, 90]], [[[20, 20, 2], [80, 20, 2], [50, 80, 2]]])
        spec = section(self.transforms, "horizontal_flip")
        spec.update({"enabled": True, "p": 1.0})
        with self.assertRaises(ValueError):
            self.run_geometry(data)
        spec["keypoint_indices"] = [1, 2, 0]
        with self.assertRaises(ValueError):
            self.run_geometry(data)
        spec["keypoint_indices"] = [0, 0, 2]
        with self.assertRaises(ValueError):
            self.run_geometry(data)

    def test_letterbox_matches_actual_rounded_resize_and_rgb_fill(self) -> None:
        section(self.transforms, "resize")["size"] = [80, 40]
        section(self.transforms, "geometry")["fill_rgb"] = [10, 20, 30]
        data = labels([[10, 20, 90, 80]], [[[25, 50, 2]]])
        output, transformed, audit = self.run_geometry(data)
        self.assertEqual(output.shape, (40, 80, 3))
        np.testing.assert_array_equal(output[0, 0], [30, 20, 10])
        np.testing.assert_allclose(transformed.boxes, [[24, 8, 56, 32]])
        np.testing.assert_allclose(transformed.keypoints, [[[30, 20, 2]]])
        self.assertEqual(operation(audit, "resize")["padding_left_top"], [20, 0])

    def test_minimum_points_and_box_size_delete_whole_rows(self) -> None:
        data = labels([[10, 10, 20, 20], [20, 20, 21, 40], [40, 40, 60, 60]],
                      [[[11, 11, 2], [12, 12, 1]], [[20, 20, 2], [21, 21, 2]], [[50, 50, 2], [0, 0, 0]]],
                      classes=[3, 7, 9])
        self.filters["min_keypoints"] = 2
        output = filter_labels(data, self.filters)
        self.assertEqual(output.class_ids.tolist(), [3])
        self.assertEqual(output.row_ids.tolist(), [1])
        self.assertEqual(output.keypoints.shape, (1, 2, 3))

    def test_empty_positive_retries_ten_then_fallback_but_negative_only_once(self) -> None:
        section(self.transforms, "affine").update({"enabled": True, "p": 1.0, "degrees": [0, 0],
            "scale": [1, 1], "translate_x": [2, 2], "translate_y": [0, 0]})
        data = labels([[20, 20, 80, 80]], [[[30, 30, 2]]])
        image, output, audit = self.run_geometry(data)
        self.assertTrue(audit["fallback"])
        self.assertEqual(audit["attempts"], 10)
        self.assertEqual(len(values(audit, "rejected_attempts")), 10)
        np.testing.assert_array_equal(image, self.image)
        np.testing.assert_array_equal(output.keypoints, data.keypoints)
        _, negative, audit = self.run_geometry(labels([]))
        self.assertEqual(len(negative.boxes), 0)
        self.assertFalse(audit["fallback"])
        self.assertEqual(audit["attempts"], 1)
        self.filters["empty_positive_policy"] = "allow"
        _, empty, audit = self.run_geometry(data)
        self.assertEqual(len(empty.boxes), 0)
        self.assertFalse(audit["fallback"])
        self.assertEqual(audit["removed_row_ids"], [1])

    def test_invalid_baseline_cannot_silently_publish_empty_fallback(self) -> None:
        self.filters["max_geometry_attempts"] = 1
        section(self.transforms, "resize")["size"] = [1, 1]
        with self.assertRaisesRegex(ValueError, "fallback"):
            self.run_geometry(labels([[0, 0, 100, 100]]))

    def test_each_operation_uses_independent_rng_and_probability_zero(self) -> None:
        section(self.transforms, "affine").update({"enabled": True, "p": 1.0})
        data = labels([[0, 0, 100, 100]])
        _, _, before = self.run_geometry(data)
        section(self.transforms, "crop").update({"enabled": True, "p": 1.0, "mode": "random"})
        _, _, after = self.run_geometry(data)
        for key in ("degrees", "scale"):
            self.assertEqual(operation(before, "affine")[key], operation(after, "affine")[key])
        section(self.transforms, "affine")["p"] = 0.0
        section(self.transforms, "crop")["p"] = 0.0
        image, _, audit = self.run_geometry(data)
        np.testing.assert_array_equal(image, self.image)
        self.assertFalse(operation(audit, "affine")["applied"])

    def test_crop_flip_letterbox_then_occlusion_preserves_point_names_and_loss(self) -> None:
        # Four arbitrary point names: swapping the first two and last two models
        # an explicit involutive semantic flip mapping, independent of any model.
        section(self.transforms, "crop").update({"enabled": True, "p": 1.0, "mode": "box"})
        section(section(self.transforms, "crop"), "box").update(
            {"scale": [1, 1], "center_jitter": 0, "min_crop_size_ratio": 0.1})
        section(self.transforms, "horizontal_flip").update(
            {"enabled": True, "p": 1.0, "keypoint_indices": [1, 0, 3, 2]})
        section(self.transforms, "resize")["size"] = [80, 40]
        for name in ("white_balance", "exposure", "contrast", "gamma", "vignette", "blur", "noise", "jpeg"):
            section(self.transforms, name)["enabled"] = False
        section(self.transforms, "occlusion").update({"enabled": True, "p": 1.0, "count": [1, 1],
            "width_ratio": [1, 1], "height_ratio": [1, 1], "fill_mode": "constant", "fill_rgb": [5, 25, 125]})
        data = labels([[25, 25, 75, 75]], [[[20, 50, 2], [50, 50, 2], [50, 60, 1], [0, 0, 0]]])
        original_points = data.keypoints.copy()
        geometric_image, geometric_labels, geometry_audit = self.run_geometry(data)
        output_image, output_labels, photo_audit, payload = apply_photometric(
            geometric_image, geometric_labels, self.transforms, 17)
        np.testing.assert_allclose(geometric_labels.keypoints,
                                   [[[40, 20, 2], [0, 0, 0], [0, 0, 0], [40, 28, 1]]])
        np.testing.assert_allclose(output_labels.keypoints,
                                   [[[40, 20, 1], [0, 0, 0], [0, 0, 0], [40, 28, 1]]])
        np.testing.assert_allclose(output_labels.boxes, [[20, 0, 60, 40]])
        np.testing.assert_array_equal(data.keypoints, original_points)
        np.testing.assert_array_equal(output_image, np.broadcast_to([125, 25, 5], (40, 80, 3)))
        self.assertIsNone(payload)
        changes = values(geometry_audit, "visibility_changes")
        self.assertEqual(len(changes), 1)
        self.assertEqual(obj(changes[0])["point_index"], 1)
        self.assertEqual(obj(changes[0])["source_point_index"], 0)
        occlusion = section(photo_audit, "occlusion")
        self.assertEqual(len(values(occlusion, "keypoint_visibility_changes")), 1)

    def test_fallback_geometry_still_allows_occlusion_to_run(self) -> None:
        section(self.transforms, "affine").update({"enabled": True, "p": 1.0, "degrees": [0, 0],
            "scale": [1, 1], "translate_x": [2, 2], "translate_y": [0, 0]})
        section(self.transforms, "occlusion").update({"enabled": True, "p": 1.0, "count": [1, 1],
            "width_ratio": [1, 1], "height_ratio": [1, 1]})
        data = labels([[20, 20, 80, 80]], [[[30, 30, 2], [50, 50, 1], [0, 0, 0]]])
        geometric_image, geometric_labels, geometry_audit = self.run_geometry(data)
        _, output_labels, _, _ = apply_photometric(geometric_image, geometric_labels, self.transforms, 2)
        self.assertTrue(geometry_audit["fallback"])
        np.testing.assert_array_equal(output_labels.boxes, data.boxes)
        np.testing.assert_array_equal(output_labels.keypoints[:, :, :2], data.keypoints[:, :, :2])
        np.testing.assert_array_equal(output_labels.keypoints[:, :, 2], [[1, 1, 0]])


if __name__ == "__main__":
    unittest.main()
