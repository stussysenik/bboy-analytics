"""Helpers to download and extract BRACE release artifacts."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


BRACE_RELEASES = {
    "interpolated_keypoints": {
        "url": "https://github.com/dmoltisanti/brace/releases/download/v1.0/dataset.zip",
        "archive_name": "dataset.zip",
    },
    "manual_keypoints": {
        "url": "https://github.com/dmoltisanti/brace/releases/download/mk_v1.0/manual_keypoints.zip",
        "archive_name": "manual_keypoints.zip",
    },
    "audio_features": {
        "url": "https://github.com/dmoltisanti/brace/releases/download/af_v1.0/audio_features.zip",
        "archive_name": "audio_features.zip",
    },
}


def _member_matches_video(member: str, *, year: int | None, video_id: str | None) -> bool:
    """Return whether a zip member belongs to one BRACE video/year subtree."""
    if member.endswith("/"):
        return False
    if year is None and video_id is None:
        return True
    parts = PurePosixPath(member).parts
    year_str = str(year) if year is not None else None
    for idx in range(len(parts) - 1):
        if year_str is not None and parts[idx] != year_str:
            continue
        if video_id is None:
            return True
        if idx + 1 < len(parts) and parts[idx + 1] == video_id:
            return True
    return False


def download_brace_artifact(
    artifact: str,
    *,
    brace_dir: str | Path = "data/brace",
    overwrite: bool = False,
) -> Path:
    """Download one BRACE release archive into the local cache."""
    if artifact not in BRACE_RELEASES:
        raise ValueError(f"Unknown BRACE artifact {artifact!r}")
    brace_dir = Path(brace_dir)
    downloads_dir = brace_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    release = BRACE_RELEASES[artifact]
    archive_path = downloads_dir / release["archive_name"]
    if archive_path.exists() and not overwrite:
        return archive_path

    tmp_path = archive_path.with_suffix(archive_path.suffix + ".part")
    with urllib.request.urlopen(release["url"]) as response, open(tmp_path, "wb") as out:
        shutil.copyfileobj(response, out)
    tmp_path.replace(archive_path)
    return archive_path


def extract_brace_artifact(
    artifact: str,
    *,
    brace_dir: str | Path = "data/brace",
    archive_path: str | Path | None = None,
    year: int | None = None,
    video_id: str | None = None,
    overwrite: bool = False,
) -> list[str]:
    """Extract one BRACE archive, optionally filtered to a single video."""
    brace_dir = Path(brace_dir)
    archive_path = Path(archive_path) if archive_path is not None else download_brace_artifact(
        artifact,
        brace_dir=brace_dir,
        overwrite=False,
    )

    extracted: list[str] = []
    with zipfile.ZipFile(archive_path) as zf:
        members = [
            member for member in zf.namelist()
            if _member_matches_video(member, year=year, video_id=video_id)
        ]
        for member in members:
            target = brace_dir / member
            if target.exists() and not overwrite:
                extracted.append(str(target))
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(str(target))
    return extracted

