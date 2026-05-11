"""Shared face result types and cosine similarity (used by semantic search and InsightFace)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaceResult:
    path: str
    bbox: list[int]  # [x, y, width, height]
    cluster_id: int
    embedding: list[float]
    confidence: float | None = None


@dataclass(frozen=True)
class ClusterResponse:
    provider: str
    faces: list[FaceResult]
    clusters: dict[str, list[int]]
    detector: str
    benchmark: dict[str, object] | None = None


def cosine(a: list[float], b: list[float]) -> float:
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return float(np.dot(av, bv) / denom) if denom else 0.0
