"""Cleanup helpers for local duplicate detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class VisualFingerprint:
    path: str
    width: int
    height: int
    ahash: str
    dhash: str
    phash: str


def _hex_bits(bits: np.ndarray) -> str:
    flat = np.asarray(bits, dtype=np.uint8).reshape(-1)
    value = 0
    for bit in flat:
        value = (value << 1) | int(bit)
    return f"{value:0{max(1, (len(flat) + 3) // 4)}x}"


def _dct_2d(values: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    basis = np.empty((n, n), dtype=np.float32)
    factor = np.pi / (2.0 * n)
    scale0 = np.sqrt(1.0 / n)
    scale = np.sqrt(2.0 / n)
    for k in range(n):
        alpha = scale0 if k == 0 else scale
        for i in range(n):
            basis[k, i] = alpha * np.cos((2 * i + 1) * k * factor)
    return basis @ values @ basis.T


def fingerprint_image(path: Path) -> VisualFingerprint:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        width, height = image.size
        gray = image.convert("L")

        small = np.asarray(gray.resize((8, 8), Image.Resampling.LANCZOS), dtype=np.float32)
        ahash = _hex_bits(small >= small.mean())

        diff_img = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
        dhash = _hex_bits(diff_img[:, 1:] > diff_img[:, :-1])

        psrc = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
        low = _dct_2d(psrc)[:8, :8]
        vals = low.reshape(-1)[1:]
        phash = _hex_bits(low >= np.median(vals))

    return VisualFingerprint(str(path), int(width), int(height), ahash, dhash, phash)


def fingerprint_paths(paths: list[Path]) -> dict:
    items: list[VisualFingerprint] = []
    errors: list[dict[str, str]] = []
    for path in paths:
        try:
            items.append(fingerprint_image(path))
        except Exception as exc:
            errors.append({"path": str(path), "error": str(exc)})
    return {"fingerprints": items, "errors": errors}
