"""Independent YOLO fixtures exercise output, safety and reproducibility contracts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from dataset_augment import augment_dataset, load_config
from dataset_augment.dataset import preflight
from dataset_augment.types import JsonObject, JsonValue, json_value, obj, section


def fixture(root: Path, *, pose: bool = True, count: int = 4) -> Path:
    dataset = root / "input"
    metadata: JsonObject = {"names": ["first", "second", "third"]}
    if pose:
        metadata.update(kpt_shape=[3, 3], flip_idx=[1, 0, 2], kpt_names=["left", "right", "center"])
    for split in ("train", "val", "test"):
        metadata[split] = f"images/{split}"
        for index in range(count if split == "train" else 1):
            relative = Path("nested" if index % 2 else ".") / f"image_{index}.png"
            image_path = dataset / "images" / split / relative
            label_path = dataset / "labels" / split / relative.with_suffix(".txt")
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            yy, xx = np.indices((48, 64))
            image = np.stack(((xx*3+index*7)%256, (yy*4+index)%256, (xx+yy*2)%256), axis=2).astype(np.uint8)
            cv2.rectangle(image, (16, 12), (48, 36), (70, 200, 240), 1)
            assert cv2.imwrite(str(image_path), image)
            if index == 1:
                label = ""
            else:
                label = "2 0.5 0.5 0.5 0.5"
                if pose:
                    label += " 0.3 0.4 2 0.7 0.6 1 0 0 0"
                label += "\n"
                if index == 2:
                    label += "0 0.5 0.5 0.3 0.3" + (" 0.4 0.4 2 0.6 0.6 2 0 0 0" if pose else "") + "\n"
            label_path.write_text(label)
    (dataset / "dataset.yaml").write_text(yaml.safe_dump(metadata))
    return dataset


def hashes(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def read_records(output: Path) -> list[JsonObject]:
    return [obj(json_value(json.loads(line))) for line in (output / "augmentation/images.jsonl").read_text().splitlines()]


class DatasetTests(unittest.TestCase):
    def test_pose_reproducible_one_four_workers_and_standalone_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = fixture(root)
            before = hashes(source)
            output1, output4 = root / "one", root / "four"
            overrides = ["preview.count_per_split=1", "preview.tile_width=100",
                         "transforms.horizontal_flip.enabled=true", "transforms.horizontal_flip.p=1"]
            for workers, output in ((1, output1), (4, output4)):
                result = augment_dataset(load_config(source=source, output=output, workers=workers, overrides=overrides))
                self.assertEqual(result["status"], "complete")
                self.assertEqual(len(list((output / "images/train").rglob("*.png"))), 8)
                self.assertEqual(len(read_records(output)), 10)
                self.assertTrue(list((output / "augmentation/previews").rglob("*.jpg")))
                for split in ("val", "test"):
                    self.assertEqual(hashes(source / "images" / split), hashes(output / "images" / split))
                    self.assertEqual(hashes(source / "labels" / split), hashes(output / "labels" / split))
                self.assertNotEqual((source / "images/train/image_0.png").stat().st_ino,
                                    (output / "images/train/image_0.png").stat().st_ino)
            self.assertEqual(read_records(output1), read_records(output4))
            manifest = obj(json_value(json.loads((output1 / "augmentation/manifest.json").read_text())))
            self.assertEqual(manifest["source_yaml_sha256"], before["dataset.yaml"])
            self.assertEqual(manifest["output_yaml_sha256"], hashlib.sha256((output1 / "dataset.yaml").read_bytes()).hexdigest())
            self.assertIn("pyyaml", section(manifest, "runtime"))
            self.assertEqual(hashes(output1 / "images"), hashes(output4 / "images"))
            self.assertEqual(hashes(output1 / "labels"), hashes(output4 / "labels"))
            self.assertEqual(before, hashes(source))
            source.rename(root / "source_moved")
            portable = preflight(load_config(source=output1, output=root / "next", overrides=["splits.train.enabled=false"]))
            self.assertEqual(len(portable.items), 10)
            self.assertFalse(any(path.is_symlink() for path in output1.rglob("*")))

    def test_detection_half_up_rounding_replace_and_negative_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = fixture(root, pose=False, count=5)
            overrides = ["generation.keep_originals=false", "generation.copies_per_image=3",
                         "splits.train.fraction=0.5", "preview.count_per_split=0"]
            config = load_config(source=source, output=root / "out", workers=1, overrides=overrides)
            dry = augment_dataset(config, dry_run=True)
            counts = section(section(dry, "counts"), "train")
            self.assertEqual(counts["selected"], 3)
            self.assertEqual(counts["output"], 11)
            self.assertFalse((root / "out").exists())
            augment_dataset(config)
            self.assertEqual(len(list((root / "out/images/train").rglob("*png"))), 11)
            for path in (root / "out/labels").rglob("*.txt"):
                for line in path.read_text().splitlines():
                    self.assertEqual(len(line.split()), 5)

    def test_yaml_and_cli_overrides_produce_same_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = fixture(root, count=1)
            cfg = root / "settings.yaml"
            cfg.write_text(yaml.safe_dump({"dataset": {"source": "input", "output": "yaml_out"},
                "runtime": {"workers": 1}, "preview": {"count_per_split": 0},
                "transforms": {"crop": {"enabled": False}, "affine": {"degrees": [4, 4]}}}))
            augment_dataset(load_config(cfg))
            command = [sys.executable, "-m", "dataset_augment", "--source", str(source), "--output", str(root / "cli_out"),
                       "--workers", "1", "--set", "preview.count_per_split=0", "--set", "transforms.crop.enabled=false",
                       "--set", "transforms.affine.degrees=[4,4]"]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=45)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(read_records(root / "yaml_out"), read_records(root / "cli_out"))

    def test_missing_label_explicit_policy_and_bad_inputs_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = fixture(root, count=1)
            label = source / "labels/train/image_0.txt"
            original = label.read_bytes()
            label.unlink()
            with self.assertRaises(FileNotFoundError):
                augment_dataset(load_config(source=source, output=root / "out"), dry_run=True)
            config = load_config(source=source, output=root / "out", overrides=["dataset.allow_missing_labels=true"])
            self.assertEqual(section(section(augment_dataset(config, dry_run=True), "counts"), "train")["missing_labels"], 1)
            label.write_text("0 0.2 0.2\n")
            with self.assertRaisesRegex(ValueError, "columns"):
                augment_dataset(config, dry_run=True)
            label.write_bytes(original)
            (source / "images/train/image_0.png").write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "decode"):
                augment_dataset(config, dry_run=True)
            self.assertFalse((root / "out").exists())
            self.assertFalse(list(root.glob(".out.augmentation-*")))

    def test_pose_mapping_and_metadata_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = fixture(root, count=1)
            path = source / "dataset.yaml"
            metadata = obj(json_value(yaml.safe_load(path.read_text())))
            metadata.pop("flip_idx")
            path.write_text(yaml.safe_dump(metadata))
            for option in ("horizontal_flip", "vertical_flip"):
                with self.assertRaisesRegex(ValueError, "mapping"):
                    preflight(load_config(source=source, output=root / "out", overrides=[f"transforms.{option}.enabled=true"]))
            config = load_config(source=source, output=root / "out", overrides=["transforms.horizontal_flip.enabled=true", "transforms.horizontal_flip.keypoint_indices=[0,1,2]"])
            self.assertEqual(preflight(config).metadata["flip_idx"], [0, 1, 2])
            for shape in ([3, 2], [0, 3], [3, 3.0]):
                metadata["kpt_shape"] = json_value(shape)
                path.write_text(yaml.safe_dump(metadata))
                with self.assertRaisesRegex(ValueError, "kpt_shape"):
                    preflight(load_config(source=source, output=root / "out"))
            metadata["kpt_shape"] = [3, 3]
            metadata["nc"] = 100
            path.write_text(yaml.safe_dump(metadata))
            with self.assertRaisesRegex(ValueError, "nc"):
                preflight(load_config(source=source, output=root / "out"))
            metadata.pop("nc")
            bad_point_names: list[JsonValue] = [["short"], [1, 2, 3], {"3": ["a", "b", "c"]}]
            for point_names in bad_point_names:
                metadata["kpt_names"] = point_names
                path.write_text(yaml.safe_dump(metadata))
                with self.assertRaisesRegex(ValueError, "kpt_names"):
                    preflight(load_config(source=source, output=root / "out"))

    def test_optional_splits_external_yaml_and_overlap_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = fixture(root, count=1)
            metadata: JsonObject = {"path": "input", "train": "images/train", "names": ["a", "b", "c"], "kpt_shape": [3, 3]}
            yaml_file = root / "entry.yaml"
            yaml_file.write_text(yaml.safe_dump(metadata))
            plan = preflight(load_config(source=yaml_file, output=root / "out"))
            self.assertEqual(set(plan.counts), {"train"})
            for output in (source, source / "nested_out", root):
                with self.assertRaises((ValueError, FileExistsError)):
                    preflight(load_config(source=source, output=output))
            with self.assertRaisesRegex(ValueError, "resize"):
                preflight(load_config(source=source, output=root / "tiny", overrides=["transforms.resize.size=[1,1]"]))

    def test_midrun_failure_does_not_publish_and_cleans_temporary_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = fixture(root, count=1)
            before = hashes(source)
            output = root / "out"
            with patch("dataset_augment.engine._preview", side_effect=RuntimeError("injected preview failure")):
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    augment_dataset(load_config(source=source, output=output, workers=1))
            self.assertFalse(output.exists())
            diagnostics = list(root.glob(".out.augmentation-*"))
            self.assertEqual(len(diagnostics), 1)
            directory = diagnostics[0]
            state = obj(json_value(json.loads((directory / "augmentation/status.json").read_text())))
            self.assertEqual(state["status"], "failed")
            for name in ("images", "labels", "dataset.yaml", "augmentation/manifest.json"):
                self.assertFalse((directory / name).exists())
            self.assertFalse((root / ".out.augmentation.lock").exists())
            self.assertEqual(hashes(source), before)

    def test_low_precision_cannot_publish_invalid_boxes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = fixture(root, pose=False, count=1)
            (source / "labels/train/image_0.txt").write_text("0 0.15 0.15 0.3 0.3\n")
            overrides = ["labels.precision=1", "preview.count_per_split=0",
                         "transforms.crop.enabled=false", "transforms.affine.enabled=false"]
            with self.assertRaisesRegex(ValueError, "precision"):
                augment_dataset(load_config(source=source, output=root / "out", workers=1, overrides=overrides))
            self.assertFalse((root / "out").exists())

    def test_source_yaml_change_prevents_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = fixture(root, count=1)

            def change_metadata(*args: object) -> None:
                path = source / "dataset.yaml"
                path.write_text(path.read_text() + "# changed during processing\n")

            with patch("dataset_augment.engine._preview", side_effect=change_metadata):
                with self.assertRaisesRegex(RuntimeError, "Source metadata changed"):
                    augment_dataset(load_config(source=source, output=root / "out", workers=1))
            self.assertFalse((root / "out").exists())
            self.assertFalse(list(root.glob(".out.augmentation-*/augmentation/manifest.json")))

    def test_variant_name_collision_is_rejected_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = fixture(root, pose=False, count=1)
            identity = hashlib.sha256(b"image_0.png").hexdigest()[:12]
            name = f"image_0__{identity}__aug0000"
            original = source / "images/train/image_0.png"
            original.with_name(name+".png").write_bytes(original.read_bytes())
            (source / "labels/train" / (name+".txt")).write_text("")
            with self.assertRaisesRegex(ValueError, "collision"):
                augment_dataset(load_config(source=source, output=root / "out"))
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
