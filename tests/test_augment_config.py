"""Configuration contracts independent of augmentation execution."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import yaml

from dataset_augment.cli import main
from dataset_augment.config import config_to_dict, load_config
from dataset_augment.types import section


class ConfigTests(unittest.TestCase):
    def test_presets_and_derived_defaults(self) -> None:
        config = load_config()
        self.assertEqual(section(config.settings, "runtime")["max_in_flight"], 8)
        self.assertEqual(section(config.settings, "generation")["copies_per_image"], 1)
        train = config.split_settings["train"]
        val = config.split_settings["val"]
        self.assertTrue(train["enabled"])
        self.assertFalse(val["enabled"])
        self.assertTrue(section(section(train, "transforms"), "affine")["enabled"])
        self.assertFalse(section(section(val, "transforms"), "affine")["enabled"])
        self.assertEqual(section(section(val, "transforms"), "exposure")["ev"], [-0.15, 0.15])
        self.assertNotEqual(train["seed"], val["seed"])
        self.assertEqual(train["seed"], load_config().split_settings["train"]["seed"])
        self.assertEqual(section(load_config(workers=1).settings, "runtime")["max_in_flight"], 2)

    def test_yaml_cli_precedence_paths_and_list_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.yaml"
            path.write_text("""dataset:
  source: inputs
  output: outputs
transforms:
  affine:
    degrees: [-20, 20]
splits:
  val:
    enabled: true
    overrides:
      transforms:
        affine:
          enabled: true
          degrees: [-5, 5]
""", encoding="utf-8")
            loaded = load_config(path)
            self.assertEqual(section(loaded.settings, "dataset")["source"], str(Path(temp) / "inputs"))
            self.assertEqual(section(section(loaded.split_settings["val"], "transforms"), "affine")["degrees"], [-5, 5])
            config = load_config(path, overrides=("transforms.affine.degrees=[-1,1]",), output="relative-output")
            self.assertEqual(section(config.settings, "dataset")["output"], str(Path.cwd() / "relative-output"))
            for name in ("train", "val"):
                self.assertEqual(section(section(config.split_settings[name], "transforms"), "affine")["degrees"], [-1, 1])
            specific = load_config(path, overrides=("transforms.affine.degrees=[-1,1]", "splits.val.overrides.transforms.affine.degrees=[0,0]"))
            self.assertEqual(section(section(specific.split_settings["val"], "transforms"), "affine")["degrees"], [0, 0])

    def test_unknown_fields_types_and_ranges_rejected(self) -> None:
        invalid = [
            "unexpected=true", "transforms.affine.degree=[0,0]", "splits.val.overrides.labels.unknown=1",
            "dataset.task=segment", "dataset.source=[]", "generation.copies_per_image=0", "generation.keep_originals=1",
            "generation.seed=-1", f"generation.seed={2**64}", "generation.seed=true", "runtime.workers=65", "runtime.workers=1.5",
            "runtime.opencv_threads=0", "runtime.max_in_flight=0", "runtime.progress_every=0", "logging.backup_count=0", "logging.max_bytes=0",
            "preview.count_per_split=-1", "preview.tile_width=0", "preview.overlay_labels=1", "logging.level=VERBOSE",
            "splits.train.fraction=1.1", "splits.train.seed=false", "splits.train.preset=strong", "splits.train.enabled=1",
            "encoding.format=bmp", "encoding.jpeg_quality=101", "encoding.png_compression=10",
            "transforms.crop.mode=all", "transforms.crop.random.width_ratio=[0,1]", "transforms.crop.box.scale=[2,1]",
            "transforms.crop.box.target_classes=[0,0]", "transforms.crop.box.target_classes=[true]", "transforms.crop.box.center_jitter=-1",
            "transforms.crop.box.min_crop_size_ratio=2", "transforms.crop.choice_weights={random: 0, box: 0}",
            "transforms.affine.p=-1", "transforms.affine.enabled=0", "transforms.affine.degrees=[0]", "transforms.affine.scale=[0,1]",
            "transforms.horizontal_flip.keypoint_indices=[1,2,0]", "transforms.vertical_flip.keypoint_indices=[1,1]",
            "transforms.resize.size=[640,0]", "transforms.resize.size=640", "transforms.geometry.interpolation=area",
            "transforms.geometry.fill_rgb=[0,0,256]", "transforms.geometry.fill_rgb=[1,2]",
            "transforms.occlusion.count=[0,2]", "transforms.occlusion.width_ratio=[0,1]", "transforms.occlusion.height_ratio=[0,2]",
            "transforms.occlusion.keypoint_anchor_p=2", "transforms.occlusion.anchor_jitter=-1", "transforms.occlusion.fill_mode=blur",
            "transforms.occlusion.color_base=[-1,2]", "transforms.occlusion.color_jitter=[-256,0]", "transforms.occlusion.max_bbox_overlap=-1",
            "transforms.occlusion.max_attempts=0", "transforms.white_balance.r_gain=[0,1]", "transforms.contrast.center=2",
            "transforms.gamma.gamma=[0,1]", "transforms.vignette.strength=[0,2]", "transforms.vignette.center_x=[-1,1]",
            "transforms.blur.weights={none: 0, gaussian: 0, motion: 0}", "transforms.blur.kernel_sizes=[2,4,6]",
            "transforms.blur.kernel_weights=[1]", "transforms.blur.gaussian_sigma=[0,1]", "transforms.noise.sigma_uint8=[-1,2]",
            "transforms.jpeg.quality=[70.5,90]", "transforms.exposure.ev=[.nan,0]", "labels.min_bbox_area_ratio=2",
            "labels.min_bbox_width_px=0", "labels.min_bbox_height_px=-1", "labels.min_keypoints=-1", "labels.empty_positive_policy=drop",
            "labels.max_geometry_attempts=0", "labels.precision=0", "transforms={crop: false}",
        ]
        for expression in invalid:
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                load_config(overrides=(expression,))

    def test_valid_boundary_overrides_and_independent_copy(self) -> None:
        config = load_config(overrides=(
            f"generation.seed={2**64 - 1}", "generation.keep_originals=false", "splits.test.fraction=0",
            "transforms.horizontal_flip.keypoint_indices=[1,0,2]", "transforms.vertical_flip.keypoint_indices=[0,1,2]",
            "transforms.resize.size=[640,480]", "transforms.occlusion.max_bbox_overlap=0", "transforms.jpeg.quality=[1,100]",
            "transforms.blur.kernel_sizes=[1]", "transforms.blur.kernel_weights=[1]", "preview.count_per_split=0",
        ))
        exported = config_to_dict(config)
        section(exported, "generation")["seed"] = 0
        self.assertEqual(section(config.settings, "generation")["seed"], 2**64 - 1)
        self.assertIn("resolved_splits", exported)

    def test_alias_conflicts_and_malformed_overrides(self) -> None:
        with self.assertRaises(ValueError):
            load_config(source="x", overrides=("dataset.source=y",))
        with self.assertRaises(ValueError):
            load_config(workers=2, overrides=("runtime={workers: 3}",))
        for overrides in (("runtime.workers",), ("runtime..workers=1",), ("runtime.workers=1", "runtime.workers=2")):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                load_config(overrides=overrides)

    def test_known_splits_and_explicit_seed(self) -> None:
        config = load_config(overrides=("splits.test.enabled=true", "splits.test.seed=12"))
        self.assertEqual(config.split_settings["test"]["seed"], 12)
        self.assertEqual(config.split_settings["test"]["preset"], "mild")
        with self.assertRaises(ValueError):
            load_config(overrides=("splits.trian.enabled=true",))

    def test_cli_dry_run_reads_all_inputs_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / "input", root / "new-parent" / "out"
            image_dir, label_dir = source / "images/train", source / "labels/train"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            ok, encoded = cv2.imencode(".png", image)
            self.assertTrue(ok)
            (image_dir / "sample.png").write_bytes(encoded.tobytes())
            label = label_dir / "sample.txt"
            label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            (source / "dataset.yaml").write_text("names: [target]\ntrain: images/train\n", encoding="utf-8")
            stdout, stderr = io.StringIO(), io.StringIO()
            args = ["--source", str(source), "--output", str(output), "--dry-run"]
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(main(args), 0)
            result: object = json.loads(stdout.getvalue())
            self.assertIsInstance(result, dict)
            self.assertIn('"output": 2', stdout.getvalue())
            self.assertFalse(output.parent.exists())
            label.write_text("invalid label", encoding="utf-8")
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(main(args), 2)
            self.assertIn("Non-numeric label", stderr.getvalue())
            self.assertFalse(output.parent.exists())
            self.assertEqual(list(root.iterdir()), [source])

    def test_cli_print_does_not_read_dataset_and_modes_are_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(main(["--source", str(root / "missing"), "--output", str(root / "out"), "--print-config"]), 0)
                with self.assertRaises(SystemExit) as raised:
                    main(["--print-config", "--dry-run"])
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(list(root.iterdir()), [])

    def test_cli_print_and_errors_without_dataset(self) -> None:
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(main(["--print-config", "--workers", "1", "--set", "transforms.affine.p=0"]), 0)
        document: object = yaml.safe_load(stdout.getvalue())
        self.assertIsInstance(document, dict)
        self.assertEqual(stderr.getvalue(), "")
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(main(["--print-config", "--set", "unknown=1"]), 2)
        self.assertIn("Unknown configuration field", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
