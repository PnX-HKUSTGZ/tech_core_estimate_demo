"""Command-line entry point; execution imports no other repository modules."""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

import yaml

from .config import config_to_dict, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a standalone offline-augmented YOLO detection or pose dataset.")
    parser.add_argument("--config", help="YAML configuration; relative paths use its directory")
    parser.add_argument("--source", help="Input dataset directory or dataset YAML")
    parser.add_argument("--output", help="New output directory (must not exist)")
    parser.add_argument("--workers", type=int, help="CPU worker processes (default: 4)")
    parser.add_argument("--seed", type=int, help="Unsigned 64-bit reproducibility seed")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=YAML_VALUE", help="Override any configuration field; repeat as needed")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--print-config", action="store_true", help="Print resolved configuration and exit without writing")
    mode.add_argument("--dry-run", action="store_true", help="Validate all inputs and report output counts without writing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, overrides=args.overrides, source=args.source, output=args.output, workers=args.workers, seed=args.seed)
        if args.print_config:
            print(yaml.safe_dump(config_to_dict(config), allow_unicode=True, sort_keys=False), end="")
            return 0
        from .engine import augment_dataset
        summary = augment_dataset(config, dry_run=args.dry_run)
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (ValueError, OSError, yaml.YAMLError) as exc:
        print(f"dataset_augment: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
