"""Bounded CPU execution, deterministic audit records and atomic publication."""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
import logging
from logging.handlers import RotatingFileHandler
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile
from collections.abc import Iterator

import cv2
import numpy as np
import yaml

from .config import AugmentConfig, config_to_dict
from .dataset import (DatasetPlan, InputImage, decode_image, digest, label_bytes,
                      output_metadata, parse_labels, preflight, variant_name)
from .geometry import apply_geometry
from .photometric import apply_photometric
from .types import (Image, JsonObject, JsonValue, Labels, boolean, derive_seed,
                    integer, json_value, obj, section, string)

SOURCE_COMMIT = "cb5ef2f3dea1223c0fedbd358c4129b2c586cb33"


@dataclass(frozen=True)
class Job:
    item: InputImage
    variant: int | None
    relative: str
    temporary: Path
    split_config: JsonObject
    encoding: JsonObject
    keypoint_count: int
    class_ids: set[int]


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(value: JsonValue) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)+"\n").encode()


def _file_digest(path: Path) -> str:
    import hashlib
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def _count_operations(counts: JsonObject, record: JsonObject) -> None:
    operations = section(counts, "operations")
    raw_operations = section(record, "geometry").get("operations", [])
    assert isinstance(raw_operations, list)
    for value in raw_operations:
        if isinstance(value, dict) and value.get("applied") is True:
            name = string(value, "operation")
            operations[name] = integer(operations, name)+1 if name in operations else 1
    for name, value in section(record, "photometric").items():
        if isinstance(value, dict) and value.get("applied") is True:
            operations[name] = integer(operations, name)+1 if name in operations else 1
    geometry = section(record, "geometry")
    if geometry.get("fallback") is True:
        counts["geometry_fallbacks"] = integer(counts, "geometry_fallbacks") + 1


def _encode(image: Image, suffix: str, encoding: JsonObject, jpeg: bytes | None) -> bytes:
    if suffix.lower() in (".jpg", ".jpeg") and jpeg is not None:
        return jpeg
    options: list[int] = []
    if suffix.lower() in (".jpg", ".jpeg"):
        options = [cv2.IMWRITE_JPEG_QUALITY, integer(encoding, "jpeg_quality")]
    elif suffix.lower() == ".png":
        options = [cv2.IMWRITE_PNG_COMPRESSION, integer(encoding, "png_compression")]
    ok, encoded = cv2.imencode(suffix, image, options)
    if not ok:
        raise RuntimeError(f"Failed image encoding: {suffix}")
    return encoded.tobytes()


def _worker_init(threads: int) -> None:
    cv2.setNumThreads(threads)


def _process(job: Job) -> JsonObject:
    item = job.item
    before_image = item.image.read_bytes()
    before_label = item.label.read_bytes() if item.label else b""
    if digest(before_image) != item.image_sha256 or digest(before_label) != item.label_sha256:
        raise RuntimeError(f"Input changed after preflight: {item.image}")
    geometry: JsonObject = {}
    photo: JsonObject = {}
    seed: int | None = None
    width, height = item.width, item.height
    image_content, label_content = before_image, before_label
    original_count = len([line for line in before_label.decode("utf-8-sig").splitlines() if line.strip()])
    final_count = original_count
    if job.variant is not None:
        image = decode_image(before_image, item.image)
        labels = parse_labels(before_label, item.label or item.image.with_suffix(".txt"), width, height,
                              job.keypoint_count, job.class_ids)
        seed = derive_seed(integer(job.split_config, "seed"), item.split, item.relative, job.variant)
        transforms = section(job.split_config, "transforms")
        label_settings = section(job.split_config, "labels")
        image, labels, geometry = apply_geometry(image, labels, transforms, label_settings, seed, (width, height))
        image, labels, photo, jpeg = apply_photometric(image, labels, transforms, seed)
        height, width = image.shape[:2]
        image_content = _encode(image, Path(job.relative).suffix, job.encoding, jpeg)
        label_content = label_bytes(labels, width, height, integer(label_settings, "precision"))
        try:
            parse_labels(label_content, Path(job.relative), width, height, job.keypoint_count, job.class_ids)
        except ValueError as error:
            raise ValueError("Output label rounding is invalid; increase labels.precision") from error
        final_count = len(labels.class_ids)
    image_relative = Path("images") / item.split / job.relative
    label_relative = Path("labels") / item.split / Path(job.relative).with_suffix(".txt")
    atomic_bytes(job.temporary / image_relative, image_content)
    atomic_bytes(job.temporary / label_relative, label_content)
    return {"split": item.split, "source": item.relative, "variant": job.variant,
            "image": image_relative.as_posix(), "label": label_relative.as_posix(),
            "seed": seed, "width": width, "height": height,
            "original_instances": original_count, "output_instances": final_count,
            "missing_label": item.label is None,
            "sha256_before": {"image": item.image_sha256, "label": item.label_sha256},
            "sha256_after": {"image": digest(image_content), "label": digest(label_content)},
            "original_bytes_retained": image_content == before_image and label_content == before_label,
            "geometry": geometry, "photometric": photo}


def _jobs(plan: DatasetPlan, settings: JsonObject, temporary: Path) -> Iterator[Job]:
    generation = section(settings, "generation")
    encoding = section(settings, "encoding")
    for item in plan.items:
        common = (temporary, plan.split_settings[item.split], encoding, plan.keypoint_count, plan.class_ids)
        if not item.selected or boolean(generation, "keep_originals"):
            yield Job(item, None, item.relative, *common)
        if item.selected:
            for index in range(integer(generation, "copies_per_image")):
                yield Job(item, index, variant_name(item, index, encoding), *common)


def _draw(image: Image, labels: Labels) -> Image:
    rendered = image.copy()
    for index, box in enumerate(labels.boxes):
        x0, y0, x1, y1 = (round(float(value)) for value in box)
        color = ((30, 210, 255), (255, 200, 50), (80, 230, 90))[int(labels.class_ids[index]) % 3]
        cv2.rectangle(rendered, (x0, y0), (x1, y1), color, 1)
        cv2.putText(rendered, str(int(labels.class_ids[index])), (x0, max(y0-3, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
        for x, y, visible in labels.keypoints[index]:
            if visible:
                cv2.circle(rendered, (round(float(x)), round(float(y))), 3, color,
                           -1 if visible == 2 else 1, cv2.LINE_AA)
    return rendered


def _preview(job: Job, record: JsonObject, settings: JsonObject) -> None:
    before = decode_image(job.item.image.read_bytes(), job.item.image)
    after_path = job.temporary / string(record, "image")
    after = decode_image(after_path.read_bytes(), after_path)
    if boolean(settings, "overlay_labels"):
        original_label = job.item.label.read_bytes() if job.item.label else b""
        label_path = job.temporary / string(record, "label")
        before = _draw(before, parse_labels(original_label, job.item.image, before.shape[1], before.shape[0], job.keypoint_count, job.class_ids))
        after = _draw(after, parse_labels(label_path.read_bytes(), label_path, after.shape[1], after.shape[0], job.keypoint_count, job.class_ids))
    tile_width = integer(settings, "tile_width")
    heights = [max(1, round(image.shape[0]*tile_width/image.shape[1])) for image in (before, after)]
    canvas = np.full((max(heights)+35, tile_width*2, 3), 28, dtype=np.uint8)
    for column, (image, height) in enumerate(zip((before, after), heights, strict=True)):
        thumb = cv2.resize(image, (tile_width, height), interpolation=cv2.INTER_AREA)
        canvas[35:35+height, column*tile_width:(column+1)*tile_width] = thumb
        cv2.putText(canvas, "INPUT" if column == 0 else "AUGMENTED", (column*tile_width+10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (240, 240, 240), 1, cv2.LINE_AA)
    target = job.temporary / "augmentation/previews" / job.item.split / Path(job.relative).with_suffix(".jpg")
    atomic_bytes(target, _encode(canvas, ".jpg", {"jpeg_quality": 93}, None))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def augment_dataset(config: AugmentConfig, *, dry_run: bool = False) -> JsonObject:
    """Validate all inputs, then publish a complete independent dataset.

    Library callers using multiprocessing must use the normal __main__ guard.
    Only child processes configure OpenCV's thread count.
    """
    plan = preflight(config)
    if dry_run:
        return {"status": "validated", "counts": plan.counts, "output": str(plan.output)}
    plan.output.parent.mkdir(parents=True, exist_ok=True)
    lock = plan.output.parent / f".{plan.output.name}.augmentation.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(f"{os.getpid()}\n")
    temporary: Path | None = None
    handler: RotatingFileHandler | None = None
    logger = logging.Logger("dataset_augment.run")
    state: JsonObject = {"status": "running", "started_utc": _now(), "completed_outputs": 0}
    completed = 0
    try:
        temporary = Path(tempfile.mkdtemp(prefix=f".{plan.output.name}.augmentation-", dir=plan.output.parent))
        audit = temporary / "augmentation"
        audit.mkdir()
        log_settings = section(config.settings, "logging")
        handler = RotatingFileHandler(audit / "run.log", maxBytes=integer(log_settings, "max_bytes"),
                                      backupCount=integer(log_settings, "backup_count"), encoding="utf-8")
        logger.addHandler(handler)
        logger.setLevel(string(log_settings, "level"))
        resolved = config_to_dict(config)
        resolved["resolved_splits"] = {key: value for key, value in plan.split_settings.items()}
        atomic_bytes(audit / "config.json", _json_bytes(resolved))
        atomic_bytes(audit / "status.json", _json_bytes(state))
        runtime = section(config.settings, "runtime")
        preview = section(config.settings, "preview")
        workers = integer(runtime, "workers")
        limit = integer(runtime, "max_in_flight")
        tasks = _jobs(plan, config.settings, temporary)
        pending: deque[tuple[Job, Future[JsonObject]]] = deque()
        # Select uniformly spaced variant indices within each split, never by visual quality.
        preview_indices: dict[str, set[int]] = {}
        preview_seen: dict[str, int] = {}
        counts: JsonObject = {}
        for split, count_value in plan.counts.items():
            split_count = obj(count_value)
            population = integer(split_count, "selected") * integer(section(config.settings, "generation"), "copies_per_image")
            preview_count = min(integer(preview, "count_per_split"), population)
            preview_indices[split] = set() if preview_count == 0 else {0} if preview_count == 1 else {i*(population-1)//(preview_count-1) for i in range(preview_count)}
            preview_seen[split] = 0
            counts[split] = {"outputs": 0, "variants": 0, "empty_outputs": 0, "geometry_fallbacks": 0, "operations": {}}
        with (audit / "images.jsonl").open("w", encoding="utf-8") as records:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_worker_init, initargs=(integer(runtime, "opencv_threads"),)) as pool:
                def submit_one() -> bool:
                    job = next(tasks, None)
                    if job is None:
                        return False
                    pending.append((job, pool.submit(_process, job)))
                    return True

                for _ in range(limit):
                    if not submit_one():
                        break
                try:
                    while pending:
                        job, future = pending.popleft()
                        record = future.result()
                        records.write(json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True)+"\n")
                        completed += 1
                        count = obj(counts[job.item.split])
                        count["outputs"] = integer(count, "outputs")+1
                        if integer(record, "output_instances") == 0:
                            count["empty_outputs"] = integer(count, "empty_outputs")+1
                        if job.variant is not None:
                            count["variants"] = integer(count, "variants")+1
                            _count_operations(count, record)
                            current = preview_seen[job.item.split]
                            if current in preview_indices[job.item.split]:
                                _preview(job, record, preview)
                            preview_seen[job.item.split] += 1
                        if completed % integer(runtime, "progress_every") == 0:
                            state["completed_outputs"] = completed
                            records.flush()
                            atomic_bytes(audit / "status.json", _json_bytes(state))
                            logger.info("completed_outputs=%s", completed)
                        submit_one()
                except BaseException:
                    for _, future in pending:
                        future.cancel()
                    raise
            records.flush()
            os.fsync(records.fileno())
        expected = sum(integer(obj(value), "output") for value in plan.counts.values())
        if completed != expected:
            raise RuntimeError(f"Output count differs: {completed} != {expected}")
        atomic_bytes(temporary / "dataset.yaml", yaml.safe_dump(output_metadata(plan), allow_unicode=True, sort_keys=False).encode())
        # Verify every output byte and source again before writing the completion marker.
        with (audit / "images.jsonl").open(encoding="utf-8") as records:
            for line in records:
                record = obj(json_value(json.loads(line)))
                hashes = section(record, "sha256_after")
                for kind in ("image", "label"):
                    if digest((temporary / string(record, kind)).read_bytes()) != string(hashes, kind):
                        raise RuntimeError(f"Output hash changed: {record[kind]}")
        for item in plan.items:
            if digest(item.image.read_bytes()) != item.image_sha256 or digest(item.label.read_bytes() if item.label else b"") != item.label_sha256:
                raise RuntimeError(f"Source changed during augmentation: {item.image}")
        if digest(plan.source_yaml.read_bytes()) != plan.source_yaml_sha256:
            raise RuntimeError(f"Source metadata changed during augmentation: {plan.source_yaml}")
        manifest: JsonObject = {"status": "complete", "source_commit": SOURCE_COMMIT,
                               "schema_version": 1, "cpu_only": True, "completed_utc": _now(),
                               "planned": plan.counts, "counts": counts,
                               "completed_outputs": completed, "source_yaml": str(plan.source_yaml),
                               "source_yaml_sha256": plan.source_yaml_sha256,
                               "output_yaml_sha256": _file_digest(temporary / "dataset.yaml"),
                               "records_sha256": _file_digest(audit / "images.jsonl"),
                               "runtime": {"python": sys.version, "numpy": np.__version__, "opencv": cv2.__version__,
                                           "pyyaml": version("PyYAML"), "project": version("tech-core-estimate-demo")},
                               "preview_selection": "uniform variant indices in stable source/variant order"}
        atomic_bytes(audit / "manifest.json", _json_bytes(manifest))
        state.update(status="complete", completed_outputs=completed, completed_utc=_now(),
                     manifest_sha256=digest((audit / "manifest.json").read_bytes()))
        atomic_bytes(audit / "status.json", _json_bytes(state))
        logger.info("complete outputs=%s", completed)
        handler.close()
        logger.removeHandler(handler)
        handler = None
        if plan.output.exists():
            raise FileExistsError(plan.output)
        temporary.rename(plan.output)
        return {"status": "complete", "output": str(plan.output), "counts": counts,
                "manifest": str(plan.output / "augmentation/manifest.json")}
    except BaseException as error:
        if temporary is not None and temporary.exists():
            state.update(status="failed", completed_outputs=completed, failed_utc=_now(),
                         error=f"{type(error).__name__}: {error}")
            logger.exception("augmentation failed")
            for folder in ("images", "labels", "augmentation/previews"):
                shutil.rmtree(temporary / folder, ignore_errors=True)
            (temporary / "dataset.yaml").unlink(missing_ok=True)
            (temporary / "augmentation/manifest.json").unlink(missing_ok=True)
            atomic_bytes(temporary / "augmentation/status.json", _json_bytes(state))
            print(f"Augmentation failed; diagnostics: {temporary}", file=sys.stderr)
        raise
    finally:
        if handler is not None:
            handler.close()
            logger.removeHandler(handler)
        lock.unlink(missing_ok=True)
