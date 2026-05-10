"""Embedding provider abstraction for the Rich Media Viewer sidecar.

The providers in this module intentionally return deterministic placeholder
embeddings. They define the boundary for future local/cloud model integrations
without transmitting user media today.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Protocol


DEFAULT_DIMENSIONS = 16


@dataclass(frozen=True)
class EmbeddingResult:
    """Embedding output for one media item."""

    source: str
    provider: str
    embedding: list[float]
    placeholder: bool = True


class EmbeddingProvider(Protocol):
    """Common interface for embedding providers."""

    name: str

    def embed_path(self, path: Path) -> EmbeddingResult:
        """Return an embedding for a media path."""


def _deterministic_vector(seed: str, dimensions: int = DEFAULT_DIMENSIONS) -> list[float]:
    """Create a stable pseudo-embedding from a string seed.

    This avoids reading file contents and keeps the current prototype private by
    default. It is not semantically meaningful.
    """

    digest = sha256(seed.encode("utf-8")).digest()
    values: list[float] = []
    for index in range(dimensions):
        byte = digest[index % len(digest)]
        values.append(round((byte / 255.0) * 2.0 - 1.0, 6))
    return values


class LocalEmbeddingProvider:
    """Local placeholder provider.

    Future versions can wrap local ML models such as CLIP or face-recognition
    embeddings. Today this provider does not inspect image bytes.
    """

    name = "local"

    def embed_path(self, path: Path) -> EmbeddingResult:
        return EmbeddingResult(
            source=str(path),
            provider=self.name,
            embedding=_deterministic_vector(f"{self.name}:{path}"),
        )


class RemoteConsentRequiredError(RuntimeError):
    """Raised when a remote provider is used without explicit consent."""


class _RemotePlaceholderProvider:
    """Base class for cloud provider stubs with consent enforcement."""

    name = "remote"

    def __init__(self, *, allow_remote: bool = False) -> None:
        self.allow_remote = allow_remote

    def embed_path(self, path: Path) -> EmbeddingResult:
        if not self.allow_remote:
            raise RemoteConsentRequiredError(
                f"Provider '{self.name}' requires explicit --allow-remote consent."
            )
        return EmbeddingResult(
            source=str(path),
            provider=self.name,
            embedding=_deterministic_vector(f"{self.name}:{path}"),
        )


class GoogleEmbeddingProvider(_RemotePlaceholderProvider):
    """Google embedding provider stub.

    Placeholder only: does not call Google APIs or upload media.
    """

    name = "google"


class OpenRouterEmbeddingProvider(_RemotePlaceholderProvider):
    """OpenRouter embedding provider stub.

    Placeholder only: does not call OpenRouter APIs or upload media.
    """

    name = "openrouter"


def create_provider(name: str, *, allow_remote: bool = False) -> EmbeddingProvider:
    """Factory for known embedding providers."""

    normalized = name.strip().lower()
    if normalized == "local":
        return LocalEmbeddingProvider()
    if normalized == "google":
        return GoogleEmbeddingProvider(allow_remote=allow_remote)
    if normalized == "openrouter":
        return OpenRouterEmbeddingProvider(allow_remote=allow_remote)
    raise ValueError(f"Unknown embedding provider: {name}")


def embed_paths(
    paths: Iterable[Path], provider: EmbeddingProvider
) -> list[EmbeddingResult]:
    """Embed multiple paths using a provider."""

    return [provider.embed_path(path) for path in paths]
