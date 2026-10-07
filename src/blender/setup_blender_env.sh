#!/usr/bin/env bash
# Install Blender 4.5.0 and its pinned dependencies in Blender's own Python.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "${script_dir}/../.." && pwd -P)"
blender_version="4.5.0"
install_dir="${BLENDER_INSTALL_DIR:-${project_root}/.tools}"
blender_folder="blender-${blender_version}-linux-x64"
blender_dir="${install_dir}/${blender_folder}"
blender_bin="${blender_dir}/blender"
blender_python="${blender_dir}/4.5/python/bin/python3.11"
package_name="${blender_folder}.tar.xz"
download_url="${BLENDER_URL:-https://mirrors.aliyun.com/blender/release/Blender4.5/${package_name}}"
temporary_dir=""

cleanup() {
    if [[ -n "${temporary_dir}" && -d "${temporary_dir}" ]]; then
        rm -rf -- "${temporary_dir}"
    fi
}
trap cleanup EXIT

command -v uv >/dev/null 2>&1 || { echo "uv is required" >&2; exit 1; }
mkdir -p -- "${install_dir}"

if [[ ! -x "${blender_bin}" ]]; then
    [[ ! -e "${blender_dir}" ]] || {
        echo "Incomplete Blender installation at ${blender_dir}; inspect it before retrying" >&2
        exit 1
    }
    temporary_dir="$(mktemp -d "${install_dir}/.blender-install.XXXXXXXX")"
    echo "Downloading Blender ${blender_version} from ${download_url}"
    curl --fail --location --retry 3 --progress-bar \
        --output "${temporary_dir}/${package_name}" "${download_url}"
    tar -xJf "${temporary_dir}/${package_name}" -C "${temporary_dir}"
    [[ -x "${temporary_dir}/${blender_folder}/blender" ]] || {
        echo "Downloaded archive does not contain ${blender_folder}/blender" >&2
        exit 1
    }
    mv -- "${temporary_dir}/${blender_folder}" "${blender_dir}"
else
    echo "Using existing Blender at ${blender_bin}"
fi

[[ -x "${blender_python}" ]] || {
    echo "Blender Python not found: ${blender_python}" >&2
    exit 1
}

echo "Installing pinned render dependencies into ${blender_python}"
uv pip install --python "${blender_python}" --requirement "${script_dir}/requirements-blender.txt"

"${blender_python}" - <<'PY'
import cv2
import numpy
import yaml

assert numpy.__version__ == "1.26.4", numpy.__version__
assert cv2.__version__ == "4.11.0", cv2.__version__
assert yaml.__version__ == "6.0.2", yaml.__version__
print(f"Blender Python: {__import__('sys').executable}")
print(f"numpy={numpy.__version__}, opencv={cv2.__version__}, PyYAML={yaml.__version__}")
PY
echo "Blender executable: ${blender_bin}"
