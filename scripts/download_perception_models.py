"""Fetch the public perception models, verifying every file before publication.

Uses only the standard library and honors HTTPS_PROXY / HF_ENDPOINT. Interrupted
downloads keep a resumable .part file. Model loading never calls this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY = "hkustenterprize/RM26_engineer_exchange_model"
REVISION = "afab41a8b61c399bf71480f0fd52352ed660b994"


@dataclass(frozen=True)
class Artifact:
    remote_path: str
    size: int
    git_blob: str
    lfs_sha256: str | None = None


# Metadata from the immutable Hugging Face tree at REVISION. Small files use
# Git's blob SHA-1; LFS files use the remote LFS SHA-256 of their actual content.
ARTIFACTS = (
    Artifact("yolo/v10.12/openvino_best/best.bin", 18982704,
             "5e08ce2eb1ddf62a39445e353663130a73f267fe",
             "443af62aa9c7cb80605bd4ed6ba8f2d9323ad542b66ebd928be08705ed64a884"),
    Artifact("yolo/v10.12/openvino_best/best.xml", 498904,
             "a4291a258e437c257cd650375335f21e561667eb"),
    Artifact("yolo/v10.12/openvino_best/metadata.yaml", 497,
             "214206273451939e6a3e88dba29914c8c44af54e"),
    Artifact("litehr/litehrnet30_v9.1/openvino_best/hrnet_heatmap.bin", 3808392,
             "2adb10c0019ff337d16ccb4bf9a53fb5a4de064a",
             "3a1db63035c5d686acceb06d4c38562766913f526df7408ea2c2cccf62457def"),
    Artifact("litehr/litehrnet30_v9.1/openvino_best/hrnet_heatmap.xml", 3392114,
             "71acec1f8e4300fb7ae4adccb6101ad463f794ac"),
)


def verify(path: Path, artifact: Artifact) -> str:
    """Return SHA-256 after checking the pinned size and upstream content hash."""
    actual_size = path.stat().st_size
    if actual_size != artifact.size:
        raise ValueError(f"{path.name}: size {actual_size}, expected {artifact.size}")
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(f"blob {actual_size}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            sha256.update(chunk)
            git_sha1.update(chunk)
    digest = sha256.hexdigest()
    if artifact.lfs_sha256 is not None:
        valid = digest == artifact.lfs_sha256
    else:
        valid = git_sha1.hexdigest() == artifact.git_blob
    if not valid:
        raise ValueError(f"{path.name}: upstream content hash mismatch")
    return digest


def download(
    artifact: Artifact, output: Path, endpoint: str, context: ssl.SSLContext
) -> dict[str, str | int | None]:
    destination = output / artifact.remote_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{endpoint}/{REPOSITORY}/resolve/{REVISION}/{artifact.remote_path}"
    if destination.exists():
        digest = verify(destination, artifact)
    else:
        partial = destination.with_name(destination.name + ".part")
        digest = ""
        for attempt in range(1, 4):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset >= artifact.size:
                    try:
                        digest = verify(partial, artifact)
                    except ValueError:
                        partial.unlink()
                        offset = 0
                    else:
                        break
                headers = {"Range": f"bytes={offset}-"} if offset else {}
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, context=context, timeout=45) as response:
                    resumed = offset > 0 and response.status == 206
                    if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise ValueError("Server returned an incorrect resume offset")
                    with partial.open("ab" if resumed else "wb") as stream:
                        while chunk := response.read(1024 * 1024):
                            stream.write(chunk)
                digest = verify(partial, artifact)
                break
            except (OSError, ValueError, urllib.error.URLError) as error:
                # Avoid logging exception URLs: redirects can contain signed tokens.
                reason = type(error).__name__
                if isinstance(error, urllib.error.HTTPError):
                    reason += f" (HTTP {error.code})"
                elif isinstance(error, urllib.error.URLError):
                    reason += f" ({type(error.reason).__name__})"
                print(f"{artifact.remote_path}: attempt {attempt}/3 failed: {reason}", flush=True)
                if attempt == 3:
                    raise RuntimeError(f"Download failed for {artifact.remote_path}; partial file retained") from None
                time.sleep(attempt)
        partial.replace(destination)
    print(f"verified {artifact.remote_path} ({artifact.size} bytes)", flush=True)
    return {
        "path": artifact.remote_path,
        "size": artifact.size,
        "sha256": digest,
        "remote_git_blob": artifact.git_blob,
        "remote_lfs_sha256": artifact.lfs_sha256,
        "url": f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{artifact.remote_path}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/perception"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--tls12", action="store_true", help="Use TLS 1.2 for proxies incompatible with TLS 1.3")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    endpoint = str(args.endpoint).rstrip("/")
    if not endpoint.startswith("https://"):
        parser.error("--endpoint must use HTTPS")
    context = ssl.create_default_context()
    if args.tls12:
        context.maximum_version = ssl.TLSVersion.TLSv1_2
    records = [download(artifact, output, endpoint, context) for artifact in ARTIFACTS]
    manifest = Path(args.manifest).resolve() if args.manifest else output / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    content = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "source_code_commit": "cb5ef2f3",
        "models": {"yolo": "YOLO26-s-v10.12", "hrnet": "LiteHRNet-30-v9.1"},
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_root": os.path.relpath(output, Path.cwd()),
        "verification": "size plus upstream Git blob SHA-1 or LFS SHA-256; local SHA-256 for all files",
        "files": records,
    }
    temporary = manifest.with_name(manifest.name + ".part")
    temporary.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest)
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
