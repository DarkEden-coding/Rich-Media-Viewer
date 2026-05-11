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
    def __init__(self, *, threshold: float = 0.88, min_size: int = 40, max_detector_side: int = 960) -> None:
        self.threshold = threshold
        self.min_size = min_size
        self.max_detector_side = max_detector_side

    def _detect(self, path: Path) -> tuple[list[tuple[list[int], float]], str]:
        if cv2 is None:
            return [], "none-opencv-unavailable"
        img = cv2.imread(str(path))
        if img is None:
            return [], "opencv-haar-read-failed"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        longest = max(width, height)
        scale = min(1.0, float(self.max_detector_side) / float(longest)) if longest else 1.0
        detector_gray = gray
        if scale < 1.0:
            detector_gray = cv2.resize(gray, (max(1, int(round(width * scale))), max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
        detector_gray = cv2.equalizeHist(detector_gray)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return [], "opencv-haar-unavailable"
        min_detector_size = max(24, int(round(self.min_size * scale)))
        candidates: list[tuple[list[int], float]] = []
        try:
            faces, _rejects, weights = cascade.detectMultiScale3(
                detector_gray,
                scaleFactor=1.08,
                minNeighbors=7,
                minSize=(min_detector_size, min_detector_size),
                outputRejectLevels=True,
            )
        except Exception:
            faces = cascade.detectMultiScale(
                detector_gray,
                scaleFactor=1.08,
                minNeighbors=7,
                minSize=(min_detector_size, min_detector_size),
            )
            weights = [1.0] * len(faces)

        inv_scale = 1.0 / scale if scale else 1.0
        for (x, y, w, h), weight in zip(faces, weights):
            ox = max(0, int(round(float(x) * inv_scale)))
            oy = max(0, int(round(float(y) * inv_scale)))
            ow = min(width - ox, int(round(float(w) * inv_scale)))
            oh = min(height - oy, int(round(float(h) * inv_scale)))
            if ow < self.min_size or oh < self.min_size:
                continue
            aspect = ow / float(oh or 1)
            area_ratio = (ow * oh) / float(max(1, width * height))
            if not 0.72 <= aspect <= 1.35:
                continue
            if area_ratio < 0.00035:
                continue
            candidates.append(([ox, oy, ow, oh], float(weight)))

        boxes = self._dedupe_boxes(candidates)
        boxes.sort(key=lambda item: (item[0][1], item[0][0], -item[0][2] * item[0][3]))
        return boxes, f"opencv-haar-downscaled-{detector_gray.shape[1]}x{detector_gray.shape[0]}"

    @staticmethod
    def _dedupe_boxes(candidates: list[tuple[list[int], float]]) -> list[tuple[list[int], float]]:
        kept: list[tuple[list[int], float]] = []
        for box, confidence in sorted(candidates, key=lambda item: item[1], reverse=True):
            if all(LocalFaceClusterer._iou(box, existing) < 0.35 for existing, _ in kept):
                kept.append((box, confidence))
        return kept

    @staticmethod
    def _iou(a: list[int], b: list[int]) -> float:
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return float(inter) / float(union) if union else 0.0

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
        pending: list[tuple[str, list[int], list[float], float]] = []
        detector = "opencv-haar" if cv2 is not None else "none-opencv-unavailable"
        for path in paths:
            boxes, det = self._detect(path)
            detector = det if det.startswith("opencv") else detector
            for box, confidence in boxes:
                pending.append((str(path), box, face_embedding(path, box), confidence))
        ids = self._cluster_ids([p[2] for p in pending])
        faces = [FaceResult(path=p, bbox=b, cluster_id=cid, embedding=e, confidence=confidence) for (p, b, e, confidence), cid in zip(pending, ids)]
        clusters: dict[str, list[int]] = {}
        for idx, f in enumerate(faces):
            clusters.setdefault(str(f.cluster_id), []).append(idx)
        return ClusterResponse(provider="local", faces=faces, clusters=clusters, detector=detector)
