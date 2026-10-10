"""Numerical camera-effect and occlusion-label contracts, without NN imports."""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from dataset_augment.photometric import apply_photometric
from dataset_augment.types import Image, JsonObject, Labels, section


def _config() -> JsonObject:
    return {
        "occlusion": {"enabled": False, "p": 1.0, "count": [1, 1], "width_ratio": [1.0, 1.0],
                      "height_ratio": [1.0, 1.0], "keypoint_anchor_p": 1.0, "anchor_jitter": 0.0,
                      "fill_mode": "constant", "fill_rgb": [128, 64, 32], "color_base": [20, 230],
                      "color_jitter": [-20, 20], "max_bbox_overlap": 1.0, "max_attempts": 3},
        "exposure": {"enabled": False, "p": 1.0, "ev": [0.0, 0.0]},
        "white_balance": {"enabled": False, "p": 1.0, "r_gain": [1.0, 1.0],
                          "g_gain": [1.0, 1.0], "b_gain": [1.0, 1.0]},
        "contrast": {"enabled": False, "p": 1.0, "factor": [1.0, 1.0], "center": 0.5},
        "gamma": {"enabled": False, "p": 1.0, "gamma": [1.0, 1.0]},
        "vignette": {"enabled": False, "p": 1.0, "strength": [0.28, 0.28],
                     "center_x": [0.5, 0.5], "center_y": [0.5, 0.5]},
        "blur": {"enabled": False, "weights": {"none": 0.0, "gaussian": 1.0, "motion": 0.0},
                 "kernel_sizes": [3], "kernel_weights": [1.0], "gaussian_sigma": [0.75, 0.75],
                 "motion_angle": [0.0, 0.0]},
        "noise": {"enabled": False, "p": 1.0, "sigma_uint8": [7.0, 7.0]},
        "jpeg": {"enabled": False, "p": 1.0, "quality": [80, 80]},
    }


def _labels() -> Labels:
    return Labels(np.array([2], dtype=np.int64), np.array([[0, 0, 16, 12]], dtype=np.float64),
                  np.array([[[8, 6, 2], [3, 3, 1], [0, 0, 0], [16, 6, 2]]], dtype=np.float64),
                  np.array([7], dtype=np.int64))


def _image() -> Image:
    return np.random.default_rng(23).integers(0, 256, (12, 16, 3), dtype=np.uint8)


class PhotometricTests(unittest.TestCase):
    def test_disabled_is_exact_and_does_not_mutate_inputs(self) -> None:
        image, labels = _image(), _labels()
        before_image, before_labels = image.copy(), labels.copy()
        output, result, _, jpeg = apply_photometric(image, labels, _config(), 100)
        np.testing.assert_array_equal(output, image)
        np.testing.assert_array_equal(result.keypoints, labels.keypoints)
        self.assertIsNone(jpeg)
        output[:] = 0
        result.keypoints[:] = 0
        np.testing.assert_array_equal(image, before_image)
        np.testing.assert_array_equal(labels.keypoints, before_labels.keypoints)

    def test_source_rgb_formula_and_order(self) -> None:
        config = _config()
        section(config, "exposure").update(enabled=True, ev=[0.15, 0.15])
        section(config, "white_balance").update(enabled=True, r_gain=[1.12, 1.12],
                                                g_gain=[0.9, 0.9], b_gain=[1.03, 1.03])
        section(config, "contrast").update(enabled=True, factor=[0.86, 0.86])
        section(config, "gamma").update(enabled=True, gamma=[1.18, 1.18])
        image = _image()
        expected = image[:, :, ::-1].astype(np.float32) / 255.0
        expected *= np.array([1.12, 0.9, 1.03], dtype=np.float32)[None, None, :] * 2 ** 0.15
        expected = (expected - 0.5) * 0.86 + 0.5
        expected = np.power(np.clip(expected, 0, 1), 1.18)
        expected_bgr = np.rint(np.clip(expected, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
        output, result, audit, _ = apply_photometric(image, _labels(), config, 12)
        np.testing.assert_array_equal(output, expected_bgr)
        np.testing.assert_array_equal(result.boxes, _labels().boxes)
        self.assertEqual(section(audit, "white_balance")["rgb_gains"], [1.12, 0.9, 1.03])

    def test_occlusion_precedes_gamma_and_changes_only_visible_points(self) -> None:
        config = _config()
        section(config, "occlusion")["enabled"] = True
        section(config, "gamma").update(enabled=True, gamma=[2.0, 2.0])
        labels = _labels()
        output, result, audit, _ = apply_photometric(_image(), labels, config, 3)
        expected_pixel = np.rint((np.array([32, 64, 128], dtype=np.float32) / 255) ** 2 * 255).astype(np.uint8)
        np.testing.assert_array_equal(output, np.broadcast_to(expected_pixel, output.shape))
        np.testing.assert_array_equal(result.keypoints[0, :, 2], [1, 1, 0, 2])
        np.testing.assert_array_equal(result.keypoints[:, :, :2], labels.keypoints[:, :, :2])
        np.testing.assert_array_equal(result.boxes, labels.boxes)
        self.assertEqual(section(audit, "occlusion")["keypoint_visibility_changes"],
                         [{"row_id": 7, "class_id": 2, "point_index": 0, "before": 2,
                           "after": 1, "occluder_indices": [0]}])
        np.testing.assert_array_equal(labels.keypoints, _labels().keypoints)

    def test_occlusion_union_counts_overlap_once(self) -> None:
        config = _config()
        section(config, "occlusion").update(enabled=True, count=[2, 2])
        _, _, audit, _ = apply_photometric(_image(), _labels(), config, 3)
        self.assertEqual(section(audit, "occlusion")["cumulative_bbox_coverage"],
                         [{"row_id": 7, "class_id": 2, "occluded_area_px": 192.0,
                           "bbox_area_px": 192.0, "coverage_fraction": 1.0}])

    def test_occlusion_overlap_limit_exhausts_attempts_without_changes(self) -> None:
        config = _config()
        section(config, "occlusion").update(enabled=True, count=[2, 2], max_bbox_overlap=0.4)
        image = _image()
        output, result, audit, _ = apply_photometric(image, _labels(), config, 3)
        np.testing.assert_array_equal(output, image)
        np.testing.assert_array_equal(result.keypoints, _labels().keypoints)
        self.assertEqual(section(audit, "occlusion")["skipped_count"], 2)
        self.assertEqual(section(audit, "occlusion")["applied"], False)

    def test_detection_negative_and_both_random_fills(self) -> None:
        labels = Labels(np.empty(0, dtype=np.int64), np.empty((0, 4), dtype=np.float64),
                        np.empty((0, 0, 3), dtype=np.float64), np.empty(0, dtype=np.int64))
        for fill in ("random_color", "noise"):
            with self.subTest(fill=fill):
                config = _config()
                section(config, "occlusion").update(enabled=True, fill_mode=fill)
                output, result, first, _ = apply_photometric(_image(), labels, config, 5)
                repeated, _, second, _ = apply_photometric(_image(), labels, config, 5)
                np.testing.assert_array_equal(output, repeated)
                self.assertEqual(first, second)
                self.assertEqual(len(result.boxes), 0)
                self.assertEqual(section(first, "occlusion")["cumulative_bbox_coverage"], [])
                unique = np.unique(output.reshape(-1, 3), axis=0)
                self.assertEqual(len(unique) == 1, fill == "random_color")

    def test_vignette_matches_source_squared_radius(self) -> None:
        config = _config()
        section(config, "vignette")["enabled"] = True
        image = np.full((5, 7, 3), 200, dtype=np.uint8)
        output, _, _, _ = apply_photometric(image, _labels(), config, 8)
        xs = np.linspace(0, 1, 7, dtype=np.float32) - 0.5
        ys = np.linspace(0, 1, 5, dtype=np.float32) - 0.5
        radius = ys[:, None] ** 2 + xs[None, :] ** 2
        radius /= radius.max()
        expected = np.rint((np.float32(200) / np.float32(255)) * (1 - 0.28 * radius) * 255).astype(np.uint8)
        np.testing.assert_array_equal(output[:, :, 0], expected)
        self.assertEqual(int(output[2, 3, 0]), 200)
        self.assertEqual(int(output[0, 0, 0]), 144)

    def test_gaussian_and_motion_are_exclusive_source_kernels(self) -> None:
        image = _image()
        pixels = image[:, :, ::-1].astype(np.float32) / 255.0
        for kind in ("gaussian", "motion", "none"):
            with self.subTest(kind=kind):
                config = _config()
                weights: JsonObject = {"none": 0.0, "gaussian": 0.0, "motion": 0.0}
                weights[kind] = 1.0
                section(config, "blur").update(enabled=True, weights=weights)
                if kind == "gaussian":
                    expected = cv2.GaussianBlur(pixels, (3, 3), 0.75, borderType=cv2.BORDER_REFLECT_101)
                elif kind == "motion":
                    kernel = np.zeros((3, 3), dtype=np.float32)
                    kernel[1] = np.float32(1.0 / 3.0)
                    expected = cv2.filter2D(pixels, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
                else:
                    expected = pixels
                expected_bgr = np.rint(np.clip(expected, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]
                output, _, audit, _ = apply_photometric(image, _labels(), config, 23)
                np.testing.assert_array_equal(output, expected_bgr)
                self.assertEqual(section(audit, "blur")["kind"], kind)

    def test_noise_is_deterministic_and_independent_of_other_operations(self) -> None:
        config = _config()
        section(config, "noise")["enabled"] = True
        image = np.full((12, 16, 3), 128, dtype=np.uint8)
        output, _, audit, _ = apply_photometric(image, _labels(), config, 21)
        section(config, "exposure")["enabled"] = True  # Identity, but consumes its own RNG.
        repeated, _, second_audit, _ = apply_photometric(image, _labels(), config, 21)
        np.testing.assert_array_equal(output, repeated)
        self.assertEqual(audit["noise"], second_audit["noise"])
        different, _, _, _ = apply_photometric(image, _labels(), config, 22)
        self.assertFalse(np.array_equal(output, different))
        self.assertGreater(float(output.std()), 3.0)

    def test_jpeg_payload_encodes_returned_image_once(self) -> None:
        config = _config()
        section(config, "jpeg")["enabled"] = True
        image = _image()
        output, _, audit, payload = apply_photometric(image, _labels(), config, 4)
        self.assertIsNotNone(payload)
        assert payload is not None
        ok, expected = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        self.assertTrue(ok)
        self.assertEqual(payload, expected.tobytes())
        np.testing.assert_array_equal(output, cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR))
        self.assertEqual(audit["jpeg"], {"applied": True, "quality": 80})

    def test_zero_probability_disables_effects(self) -> None:
        config = _config()
        for name in ("occlusion", "exposure", "white_balance", "contrast", "gamma", "vignette", "noise", "jpeg"):
            section(config, name).update(enabled=True, p=0.0)
        output, result, _, payload = apply_photometric(_image(), _labels(), config, 11)
        np.testing.assert_array_equal(output, _image())
        np.testing.assert_array_equal(result.keypoints, _labels().keypoints)
        self.assertIsNone(payload)


if __name__ == "__main__":
    unittest.main()
