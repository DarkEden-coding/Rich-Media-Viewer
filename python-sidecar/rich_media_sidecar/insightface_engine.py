"""InsightFace detection/recognition with FAISS-backed batch clustering."""

from __future__ import annotations

import contextlib
import io
import os
import site
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

from .clustering import ClusterResponse, FaceResult, cosine

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


def _l2_normalize_rows(mat: np.ndarray) -> None:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    mat /= norms


def _model_root() -> str:
    return os.environ.get(
        "INSIGHTFACE_ROOT",
        str(Path(__file__).resolve().parent / "models"),
    )


_face_app = None
_face_app_info: dict[str, object] = {}
_dll_directory_handles: list[Any] = []


@contextlib.contextmanager
def _silence_stdout():
    buf = io.StringIO()
    prev = sys.stdout
    sys.stdout = buf
    try:
        yield
    finally:
        sys.stdout = prev


def _onnxruntime_info() -> dict[str, object]:
    """Return ONNX Runtime provider details and preload CUDA DLLs when available."""
    dll_dirs = _register_nvidia_dll_directories()
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:  # pragma: no cover
        return {"available": False, "error": str(exc), "providers": []}
    preload_error = None
    preload = getattr(ort, "preload_dlls", None)
    if callable(preload):
        try:
            preload(cuda=True, cudnn=True, msvc=True)
        except Exception as exc:  # pragma: no cover
            preload_error = str(exc)
    info: dict[str, object] = {
        "available": True,
        "version": getattr(ort, "__version__", None),
        "providers": list(ort.get_available_providers()),
        "dll_dirs": dll_dirs,
    }
    if preload_error:
        info["preload_error"] = preload_error
    return info


def _register_nvidia_dll_directories() -> list[str]:
    """Add NVIDIA wheel DLL folders to the Windows loader search path."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return []
    added: list[str] = []
    roots = [Path(path) for path in site.getsitepackages()]
    user_site = site.getusersitepackages()
    if user_site:
        roots.append(Path(user_site))
    for root in roots:
        nvidia_root = root / "nvidia"
        if not nvidia_root.exists():
            continue
        for dll_dir in nvidia_root.glob("*/*"):
            if dll_dir.name.lower() != "bin" or not dll_dir.is_dir():
                continue
            dll_path = str(dll_dir)
            if dll_path in added:
                continue
            _dll_directory_handles.append(os.add_dll_directory(dll_path))
            os.environ["PATH"] = f"{dll_path}{os.pathsep}{os.environ.get('PATH', '')}"
            added.append(dll_path)
    return added


def _preferred_ctx_ids(providers: list[str]) -> list[int]:
    """Choose InsightFace ctx_id order, preferring CUDA unless explicitly overridden."""
    raw_ctx = os.environ.get("INSIGHTFACE_CTX_ID")
    if raw_ctx is not None:
        return [int(raw_ctx)]
    if "CUDAExecutionProvider" in providers:
        return [0, -1]
    return [-1]


def get_face_app(force_cpu: bool = False):
    """Lazily construct a shared FaceAnalysis instance (SCRFD + ArcFace, buffalo_l)."""
    global _face_app, _face_app_info
    if _face_app is not None and not force_cpu:
        return _face_app
    if force_cpu:
        _face_app = None
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for InsightFace image loading.")
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise RuntimeError(
            "The insightface package is not installed. "
            "From python-sidecar run: pip install -r requirements.txt"
        ) from exc
    root = _model_root()
    Path(root).mkdir(parents=True, exist_ok=True)
    ort_info = _onnxruntime_info()
    providers = [str(p) for p in ort_info.get("providers", [])]
    last_error = None
    init_start = time.perf_counter()
    ctx_ids = [-1] if force_cpu else _preferred_ctx_ids(providers)
    for ctx in ctx_ids:
        try:
            with _silence_stdout():
                app = FaceAnalysis(
                    name="buffalo_l",
                    root=root,
                    allowed_modules=["detection", "recognition"],
                )
                app.prepare(ctx_id=ctx, det_size=(640, 640))
            _face_app = app
            _face_app_info = {
                "ctx_id": ctx,
                "device": "cuda" if ctx >= 0 else "cpu",
                "onnxruntime": ort_info,
                "init_ms": round((time.perf_counter() - init_start) * 1000, 2),
            }
            if force_cpu:
                _face_app_info["fallback_reason"] = "cuda inference failed"
            return _face_app
        except Exception as exc:
            last_error = exc
            if ctx < 0:
                break
    raise RuntimeError(f"failed to initialize InsightFace: {last_error}") from last_error


def _norm_embedding(face: Any) -> np.ndarray:
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = getattr(face, "embedding", None)
    if emb is None:
        return np.zeros((0,), dtype=np.float32)
    v = np.asarray(emb, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n > 0:
        v /= n
    return v


def _bbox_xywh(face: Any) -> tuple[list[int], float]:
    x1, y1, x2, y2 = (float(x) for x in face.bbox[:4])
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    box = [int(round(x1)), int(round(y1)), int(round(w)), int(round(h))]
    score = float(getattr(face, "det_score", 1.0) or 1.0)
    return box, score


def _detect_faces(app: Any, img: Any) -> list[Any]:
    """Run InsightFace detection, retrying once on CPU if CUDA fails at inference time."""
    try:
        return app.get(img)
    except Exception:
        if _face_app_info.get("device") != "cuda":
            raise
        cpu_app = get_face_app(force_cpu=True)
        return cpu_app.get(img)


def _read_oriented_bgr(path: Path) -> Any | None:
    """Load an image in the same EXIF-oriented coordinate space browsers display."""
    if cv2 is None:
        return None
    try:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            rgb = np.asarray(image)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return cv2.imread(str(path))


def cluster_union_find(embeddings: list[list[float]], threshold: float) -> list[int]:
    """Assign cluster ids using union-find on pairs with cosine >= threshold (via FAISS IP)."""
    n = len(embeddings)
    if n == 0:
        return []
    if faiss is None:
        raise RuntimeError(
            "faiss is not installed. From python-sidecar run: pip install -r requirements.txt"
        )
    if n == 1:
        return [0]
    x = np.asarray(embeddings, dtype=np.float32)
    _l2_normalize_rows(x)
    d = x.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(x)
    k = n
    sims, idx = index.search(x, k)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    for i in range(n):
        for j in range(k):
            col = int(idx[i, j])
            if col < 0:
                continue
            if float(sims[i, j]) >= threshold:
                union(i, col)
    roots: dict[int, int] = {}
    out: list[int] = []
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        out.append(roots[r])
    return out


def cluster_paths(paths: list[Path], threshold: float) -> ClusterResponse:
    """Detect faces with InsightFace and cluster embeddings using FAISS-assisted union-find."""
    started = time.perf_counter()
    if cv2 is None:
        return ClusterResponse(
            provider="insightface",
            faces=[],
            clusters={},
            detector="none-opencv-unavailable",
            benchmark={"images": len(paths), "faces": 0, "elapsed_ms": 0.0},
        )
    app = get_face_app()
    pending: list[tuple[str, list[int], list[float], float]] = []
    detector = "insightface-buffalo_l-scrfd"
    images_read = 0
    for path in paths:
        img = _read_oriented_bgr(path)
        if img is None:
            continue
        images_read += 1
        faces = _detect_faces(app, img)
        app = _face_app
        for face in faces:
            emb = _norm_embedding(face)
            if emb.size == 0:
                continue
            box, conf = _bbox_xywh(face)
            pending.append((str(path), box, emb.astype(float).tolist(), conf))
    if not pending:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return ClusterResponse(
            provider="insightface",
            faces=[],
            clusters={},
            detector=detector,
            benchmark={
                "images": images_read,
                "faces": 0,
                "elapsed_ms": elapsed_ms,
                "ms_per_image": round(elapsed_ms / images_read, 2) if images_read else None,
                "face_app": _face_app_info,
            },
        )
    ids = cluster_union_find([p[2] for p in pending], threshold)
    faces = [
        FaceResult(path=p, bbox=b, cluster_id=cid, embedding=e, confidence=confidence)
        for (p, b, e, confidence), cid in zip(pending, ids)
    ]
    clusters: dict[str, list[int]] = {}
    for idx, f in enumerate(faces):
        clusters.setdefault(str(f.cluster_id), []).append(idx)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    return ClusterResponse(
        provider="insightface",
        faces=faces,
        clusters=clusters,
        detector=detector,
        benchmark={
            "images": images_read,
            "faces": len(faces),
            "elapsed_ms": elapsed_ms,
            "ms_per_image": round(elapsed_ms / images_read, 2) if images_read else None,
            "face_app": _face_app_info,
        },
    )


def match_propagate(
    seed: list[float],
    threshold: float,
    exclude_media_item_id: int,
    candidates: list[dict[str, Any]],
) -> list[int]:
    """Return face_id values whose embedding cosine to seed >= threshold (excluding media id)."""
    if not candidates:
        return []
    seed_v = np.asarray(seed, dtype=np.float32).reshape(1, -1)
    _l2_normalize_rows(seed_v)
    rows: list[list[float]] = []
    meta: list[tuple[int, int]] = []
    for c in candidates:
        fid = int(c["face_id"])
        mid = int(c["media_item_id"])
        emb = c.get("embedding") or c.get("vector")
        if not isinstance(emb, list) or len(emb) != seed_v.shape[1]:
            continue
        rows.append([float(x) for x in emb])
        meta.append((fid, mid))
    if not rows:
        return []
    x = np.asarray(rows, dtype=np.float32)
    _l2_normalize_rows(x)
    sims = (x @ seed_v.T).flatten()
    return [
        fid
        for i, (fid, mid) in enumerate(meta)
        if mid != exclude_media_item_id and float(sims[i]) >= threshold
    ]


def match_best_person(
    query: list[float],
    threshold: float,
    named_rows: list[dict[str, Any]],
) -> int | None:
    """Pick the person_id with highest cosine to query among named reference embeddings."""
    pids: list[int] = []
    vecs: list[list[float]] = []
    for row in named_rows:
        emb = row.get("embedding") or row.get("vector")
        pid = row.get("person_id")
        if not isinstance(emb, list) or pid is None:
            continue
        pids.append(int(pid))
        vecs.append([float(x) for x in emb])
    if not vecs:
        return None
    q = np.asarray(query, dtype=np.float32).reshape(1, -1)
    _l2_normalize_rows(q)
    x = np.asarray(vecs, dtype=np.float32)
    _l2_normalize_rows(x)
    if faiss is not None and x.shape[0] >= 1:
        d = x.shape[1]
        if q.shape[1] != d:
            return None
        index = faiss.IndexFlatIP(d)
        index.add(x)
        sims, idx = index.search(q, 1)
        s = float(sims[0, 0])
        j = int(idx[0, 0])
        if j < 0 or s < float(threshold):
            return None
        return pids[j]
    best_s = float(threshold)
    best_pid: int | None = None
    for row in named_rows:
        emb = row.get("embedding") or row.get("vector")
        pid = row.get("person_id")
        if not isinstance(emb, list) or pid is None:
            continue
        s = cosine(query, emb)
        if s >= best_s:
            best_s = s
            best_pid = int(pid)
    return best_pid


def handle_face_match_request(payload: dict[str, Any]) -> dict[str, Any]:
    mode = str(payload.get("mode", ""))
    if mode == "propagate":
        seed = payload.get("seed") or payload.get("query")
        if not isinstance(seed, list):
            raise ValueError("propagate requires seed: list[float]")
        th = float(payload.get("threshold", 0.42))
        ex = int(payload.get("exclude_media_item_id", -1))
        cands = payload.get("candidates")
        if not isinstance(cands, list):
            raise ValueError("propagate requires candidates: list")
        ids = match_propagate(seed, th, ex, cands)
        return {"matching_face_ids": ids}
    if mode == "best_person":
        query = payload.get("query") or payload.get("seed")
        if not isinstance(query, list):
            raise ValueError("best_person requires query: list[float]")
        th = float(payload.get("threshold", 0.42))
        rows = payload.get("named_faces")
        if not isinstance(rows, list):
            raise ValueError("best_person requires named_faces: list")
        pid = match_best_person(query, th, rows)
        return {"person_id": pid}
    raise ValueError(f"unknown face-match mode: {mode!r}")
