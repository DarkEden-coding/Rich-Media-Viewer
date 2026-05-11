"""HEIC/HEIF conversion helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


HEIC_SUFFIXES = {".heic", ".heif"}


@dataclass(frozen=True)
class HeicConversion:
    source: str
    converted: str


def register_heif_opener() -> None:
    try:
        from pillow_heif import register_heif_opener
    except Exception as exc:
        raise RuntimeError(
            "HEIC/HEIF support requires pillow-heif. Install python-sidecar requirements again."
        ) from exc
    register_heif_opener()


def is_heic_path(path: Path) -> bool:
    return path.suffix.lower() in HEIC_SUFFIXES


def _conversion_path(source: Path, cache_dir: Path) -> Path:
    stat = source.stat()
    key = f"{source.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    digest = hashlib.sha256(key.encode("utf-8", "surrogatepass")).hexdigest()
    return cache_dir / f"{digest}.jpg"


def convert_heic_paths(paths: list[Path], cache_dir: Path) -> list[HeicConversion]:
    register_heif_opener()
    cache_dir.mkdir(parents=True, exist_ok=True)
    conversions: list[HeicConversion] = []
    for source in paths:
        if not is_heic_path(source):
            conversions.append(HeicConversion(str(source), str(source)))
            continue
        converted = _conversion_path(source, cache_dir)
        if not converted.exists():
            with Image.open(source) as image:
                image = ImageOps.exif_transpose(image)
                if image.mode not in ("RGB", "L"):
                    image = image.convert("RGB")
                image.save(converted, format="JPEG", quality=92, optimize=True)
        conversions.append(HeicConversion(str(source), str(converted)))
    return conversions
