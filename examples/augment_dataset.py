"""Minimal callable example; guard multiprocessing with __main__."""
from __future__ import annotations

import argparse
import json

from dataset_augment import augment_dataset, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment a standalone YOLO dataset from YAML")
    parser.add_argument("config", help="Configuration YAML path")
    args = parser.parse_args()
    config = load_config(args.config)
    summary = augment_dataset(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
