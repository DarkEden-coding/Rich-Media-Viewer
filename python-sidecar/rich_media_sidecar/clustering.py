"""Local face clustering skeleton.

This module currently implements deterministic placeholder clustering over the
provider abstraction. It does not perform face detection or biometric analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .providers import EmbeddingProvider, EmbeddingResult, embed_paths


@dataclass(frozen=True)
class FaceCluster:
    """Placeholder representation of a cluster of visually similar faces."""

    cluster_id: str
    sources: list[str]
    placeholder: bool = True


@dataclass(frozen=True)
class ClusterResponse:
    """Clustering response returned by the sidecar."""

    provider: str
    clusters: list[FaceCluster]
    embeddings: list[EmbeddingResult]
    placeholder: bool = True


class LocalFaceClusterer:
    """Skeleton face clusterer for future local ML integration.

    Current behavior groups paths by a deterministic bucket derived from their
    placeholder embeddings. This is useful for plumbing tests only.
    """

    def __init__(self, provider: EmbeddingProvider, *, buckets: int = 3) -> None:
        self.provider = provider
        self.buckets = max(1, buckets)

    def cluster_paths(self, paths: list[Path]) -> ClusterResponse:
        embeddings = embed_paths(paths, self.provider)
        grouped: dict[int, list[str]] = {}
        for item in embeddings:
            first_value = item.embedding[0] if item.embedding else 0.0
            bucket = int(abs(first_value) * 1000) % self.buckets
            grouped.setdefault(bucket, []).append(item.source)

        clusters = [
            FaceCluster(cluster_id=f"placeholder-{bucket}", sources=sources)
            for bucket, sources in sorted(grouped.items())
        ]
        return ClusterResponse(
            provider=self.provider.name,
            clusters=clusters,
            embeddings=embeddings,
        )
