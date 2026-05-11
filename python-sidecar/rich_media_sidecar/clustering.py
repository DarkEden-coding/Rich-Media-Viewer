"""Local face detection and deterministic face clustering."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None


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


def cosine(a: list[float], b: list[float]) -> float:
    av = np.asarray(a, dtype=np.float32); bv = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return float(np.dot(av, bv) / denom) if denom else 0.0


def deterministic_face_vector(seed: str, dimensions: int = 64) -> list[float]:
    vals: list[float] = []
    counter = 0
    while len(vals) < dimensions:
        digest = sha256(f"{seed}:{counter}".encode()).digest()
        vals.extend([(b / 127.5) - 1.0 for b in digest])
        counter += 1
    norm = sum(v * v for v in vals[:dimensions]) ** 0.5
    return [round(float(v / norm), 6) if norm else 0.0 for v in vals[:dimensions]]


def face_embedding(image_path: Path, bbox: list[int], dimensions: int = 64) -> list[float]:
    try:
        with Image.open(image_path) as img:
            img = img.convert("L")
            x, y, w, h = bbox
            crop = img.crop((x, y, x + w, y + h)).resize((16, 16))
            arr = np.asarray(crop, dtype=np.float32) / 255.0
            # Equalized coarse pixels + gradient statistics form deterministic content-aware vector.
            arr = (arr - arr.mean()) / (arr.std() + 1e-6)
            gy, gx = np.gradient(arr)
            pooled = np.asarray(Image.fromarray(((arr - arr.min()) / (arr.max() - arr.min() + 1e-6) * 255).astype("uint8")).resize((8, 8)), dtype=np.float32).flatten() / 255.0
            feats = np.concatenate([pooled, [gx.mean(), gy.mean(), gx.std(), gy.std(), arr.mean(), arr.std()]])
            if feats.size > dimensions:
                feats = feats[:dimensions]
            elif feats.size < dimensions:
                feats = np.pad(feats, (0, dimensions - feats.size))
            norm = float(np.linalg.norm(feats))
            if norm: feats = feats / norm
            return [round(float(v), 6) for v in feats]
    except Exception:
        return deterministic_face_vector(f"face:{image_path}:{bbox}", dimensions)


class LocalFaceClusterer:
    def __init__(self, *, threshold: float = 0.88, min_size: int = 1) -> None:
        self.threshold = threshold
        self.min_size = min_size

    def _detect(self, path: Path) -> tuple[list[list[int]], str]:
        if cv2 is None:
            return [], "none-opencv-unavailable"
        img = cv2.imread(str(path))
        if img is None:
            return [], "opencv-haar-read-failed"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return [], "opencv-haar-unavailable"
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
        boxes = [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in faces]
        boxes.sort(key=lambda b: (b[1], b[0], -b[2] * b[3]))
        return boxes, "opencv-haar"

    def _cluster_ids(self, embeddings: list[list[float]]) -> list[int]:
        centroids: list[list[float]] = []
        ids: list[int] = []
        for emb in embeddings:
            best_i, best_s = -1, -2.0
            for i, c in enumerate(centroids):
                s = cosine(emb, c)
                if s > best_s: best_i, best_s = i, s
            if best_i >= 0 and best_s >= self.threshold:
                ids.append(best_i)
                members = [embeddings[j] for j, cid in enumerate(ids[:-1]) if cid == best_i] + [emb]
                centroids[best_i] = np.mean(np.asarray(members), axis=0).tolist()
            else:
                ids.append(len(centroids)); centroids.append(emb)
        return ids

    def cluster_paths(self, paths: list[Path]) -> ClusterResponse:
        pending: list[tuple[str, list[int], list[float]]] = []
        detector = "opencv-haar" if cv2 is not None else "none-opencv-unavailable"
        for path in paths:
            boxes, det = self._detect(path)
            detector = det if det.startswith("opencv") else detector
            for box in boxes:
                pending.append((str(path), box, face_embedding(path, box)))
        ids = self._cluster_ids([p[2] for p in pending])
        faces = [FaceResult(path=p, bbox=b, cluster_id=cid, embedding=e) for (p, b, e), cid in zip(pending, ids)]
        clusters: dict[str, list[int]] = {}
        for idx, f in enumerate(faces):
            clusters.setdefault(str(f.cluster_id), []).append(idx)
        return ClusterResponse(provider="local", faces=faces, clusters=clusters, detector=detector)
