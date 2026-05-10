"""Embedding providers and local deterministic media embeddings."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Protocol
from urllib import request, error

import numpy as np
from PIL import Image, ImageStat

DEFAULT_DIMENSIONS = 64
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mpeg", ".mpg"}


@dataclass(frozen=True)
class EmbeddingResult:
    source: str
    provider: str
    embedding: list[float]
    kind: str = "path"
    model: str = "local-deterministic-v1"
    error: str | None = None


class EmbeddingProvider(Protocol):
    name: str
    def embed_path(self, path: Path) -> EmbeddingResult: ...
    def embed_text(self, text: str, source: str | None = None) -> EmbeddingResult: ...


def _normalize(values: np.ndarray, dimensions: int = DEFAULT_DIMENSIONS) -> list[float]:
    arr = np.asarray(values, dtype=np.float32).flatten()
    if arr.size < dimensions:
        arr = np.pad(arr, (0, dimensions - arr.size))
    elif arr.size > dimensions:
        # deterministic average pooling
        bins = np.array_split(arr, dimensions)
        arr = np.asarray([float(b.mean()) for b in bins], dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return [round(float(x), 6) for x in arr]


def deterministic_vector(seed: str, dimensions: int = DEFAULT_DIMENSIONS) -> list[float]:
    vals: list[float] = []
    counter = 0
    while len(vals) < dimensions:
        digest = sha256(f"{seed}:{counter}".encode()).digest()
        vals.extend([(b / 127.5) - 1.0 for b in digest])
        counter += 1
    return _normalize(np.asarray(vals[:dimensions], dtype=np.float32), dimensions)


def text_embedding(text: str, dimensions: int = DEFAULT_DIMENSIONS) -> list[float]:
    vec = np.zeros(dimensions, dtype=np.float32)
    tokens = [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if t]
    if not tokens:
        return deterministic_vector("empty-text", dimensions)
    for tok in tokens:
        digest = sha256(tok.encode()).digest()
        idx = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign * (1.0 + math.log1p(len(tok)))
    # character trigram signal
    compact = " ".join(tokens)
    for i in range(max(0, len(compact) - 2)):
        tri = compact[i:i+3]
        digest = sha256(tri.encode()).digest()
        vec[int.from_bytes(digest[:2], "big") % dimensions] += 0.25
    return _normalize(vec, dimensions)


def image_embedding(path: Path, dimensions: int = DEFAULT_DIMENSIONS) -> list[float]:
    with Image.open(path) as img:
        img = img.convert("RGB")
        small = img.resize((32, 32))
        arr = np.asarray(small, dtype=np.float32) / 255.0
        means = arr.mean(axis=(0, 1))
        stds = arr.std(axis=(0, 1))
        # RGB histograms (8 bins each), thumbnail luminance, simple gradients
        hists = []
        for c in range(3):
            hist, _ = np.histogram(arr[:, :, c], bins=8, range=(0.0, 1.0), density=False)
            hists.extend((hist / max(1, hist.sum())).tolist())
        lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])
        thumb = np.asarray(Image.fromarray((lum * 255).astype("uint8")).resize((6, 6)), dtype=np.float32).flatten() / 255.0
        gy, gx = np.gradient(lum)
        features = np.concatenate([means, stds, np.asarray(hists), thumb, [gx.mean(), gy.mean(), gx.std(), gy.std(), img.width / max(1, img.height)]])
        return _normalize(features, dimensions)


def path_metadata_embedding(path: Path, dimensions: int = DEFAULT_DIMENSIONS) -> list[float]:
    try:
        st = path.stat()
        seed = f"{path.suffix.lower()}:{path.name}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        seed = str(path)
    return deterministic_vector(seed, dimensions)


class LocalEmbeddingProvider:
    name = "local"

    def embed_path(self, path: Path) -> EmbeddingResult:
        suffix = path.suffix.lower()
        try:
            if suffix in IMAGE_EXTS:
                return EmbeddingResult(str(path), self.name, image_embedding(path), kind="image")
            if suffix in VIDEO_EXTS:
                return EmbeddingResult(str(path), self.name, path_metadata_embedding(path), kind="video")
            return EmbeddingResult(str(path), self.name, path_metadata_embedding(path), kind="path")
        except Exception as exc:
            return EmbeddingResult(str(path), self.name, path_metadata_embedding(path), kind="path", error=str(exc))

    def embed_text(self, text: str, source: str | None = None) -> EmbeddingResult:
        return EmbeddingResult(source or text[:80], self.name, text_embedding(text), kind="text")


class RemoteConsentRequiredError(RuntimeError): pass
class RemoteProviderUnavailableError(RuntimeError): pass


class _RemoteProvider:
    name = "remote"
    endpoint = ""
    env_key = ""

    def __init__(self, *, allow_remote: bool = False) -> None:
        self.allow_remote = allow_remote
        self.api_key = os.getenv(self.env_key, "")

    def _check(self) -> None:
        if not self.allow_remote:
            raise RemoteConsentRequiredError(f"Provider '{self.name}' requires --allow-remote.")
        if not self.api_key:
            raise RemoteProviderUnavailableError(f"Provider '{self.name}' requires environment variable {self.env_key}.")

    def _post(self, payload: dict) -> dict:
        self._check()
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(self.endpoint, data=data, headers={"Content-Type":"application/json", "Authorization": f"Bearer {self.api_key}"})
        try:
            with request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.URLError as exc:
            raise RemoteProviderUnavailableError(f"{self.name} request failed: {exc}") from exc

    def embed_path(self, path: Path) -> EmbeddingResult:
        self._check()
        # Safe structure: do not upload local media bytes; embed metadata only.
        return EmbeddingResult(str(path), self.name, path_metadata_embedding(path), kind="path", model=f"{self.name}-metadata-safe")

    def embed_text(self, text: str, source: str | None = None) -> EmbeddingResult:
        self._check()
        # Keep deterministic fallback until response mapping is configured.
        return EmbeddingResult(source or text[:80], self.name, text_embedding(text), kind="text", model=f"{self.name}-stub")


class GoogleEmbeddingProvider(_RemoteProvider):
    name = "google"; env_key = "GOOGLE_API_KEY"; endpoint = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"


class OpenRouterEmbeddingProvider(_RemoteProvider):
    name = "openrouter"; env_key = "OPENROUTER_API_KEY"; endpoint = "https://openrouter.ai/api/v1/embeddings"


def create_provider(name: str, *, allow_remote: bool = False) -> EmbeddingProvider:
    n = name.strip().lower()
    if n == "local": return LocalEmbeddingProvider()
    if n == "google": return GoogleEmbeddingProvider(allow_remote=allow_remote)
    if n == "openrouter": return OpenRouterEmbeddingProvider(allow_remote=allow_remote)
    raise ValueError(f"Unknown embedding provider: {name}")


def embed_paths(paths: Iterable[Path], provider: EmbeddingProvider) -> list[EmbeddingResult]:
    return [provider.embed_path(p) for p in paths]


def embed_texts(texts: Iterable[str], provider: EmbeddingProvider) -> list[EmbeddingResult]:
    return [provider.embed_text(t) for t in texts]
