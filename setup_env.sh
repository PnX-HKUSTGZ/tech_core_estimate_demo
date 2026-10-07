#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export NO_ALBUMENTATIONS_UPDATE=1

if [ ! -f "${PROJECT_ROOT}/uv.lock" ]; then
  echo "Missing root uv.lock; run uv lock before installation." >&2
  exit 1
fi

# Chumpy and MMCV use legacy build scripts that need these tools in the target
# environment. Keep build intermediates under uv's cache and clean temporary
# files created by this script when it exits.
TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/tech-core-env.XXXXXXXX")"
trap 'rm -rf -- "${TEMP_ROOT}"' EXIT
export TMPDIR="${TEMP_ROOT}"

if [ ! -x "${PROJECT_ROOT}/.venv/bin/pip" ]; then
  uv venv --python 3.10 --seed --allow-existing
fi

# MMCV imports PyTorch while compiling its native extension. Install the rest
# first so the non-isolated MMCV build uses the locked PyTorch version.
uv sync --locked \
  --no-install-package mmcv \
  --no-build-isolation-package chumpy

# LiteHRNet needs MMCV CPU ops, not its CUDA operators.
CUDA_VISIBLE_DEVICES="" MMCV_WITH_OPS=1 \
  uv sync --locked --no-build-isolation-package mmcv

# The published xtcocotools wheel targets the NumPy 1.x ABI. Rebuild its
# native extension against this environment's locked NumPy 2.x installation.
uv pip install --python "${PROJECT_ROOT}/.venv/bin/python" \
  --reinstall \
  --no-deps \
  --no-binary xtcocotools \
  --no-build-isolation \
  "xtcocotools==1.14.3"

CUDA_VISIBLE_DEVICES="" uv run --no-sync rm26-nn check
