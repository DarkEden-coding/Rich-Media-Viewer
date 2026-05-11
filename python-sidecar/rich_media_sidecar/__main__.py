"""JSON CLI entrypoint for the Rich Media Viewer Python sidecar."""

from __future__ import annotations

import argparse, json, os, sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .clustering import cosine
from .heic import convert_heic_paths
from .insightface_engine import cluster_paths as insightface_cluster_paths
from .insightface_engine import handle_face_match_request
from .providers import FASTEMBED_EMBEDDING_MODELS, GOOGLE_EMBEDDING_MODELS, OPENROUTER_EMBEDDING_MODELS, RemoteProviderUnavailableError, create_provider, embed_paths, embed_texts


def _jsonable(v: Any) -> Any:
    if is_dataclass(v): return {k: _jsonable(val) for k, val in asdict(v).items()}
    if isinstance(v, list): return [_jsonable(x) for x in v]
    if isinstance(v, dict): return {str(k): _jsonable(val) for k, val in v.items()}
    return v


def write_ok(data: Any) -> None:
    print(json.dumps({"ok": True, "data": _jsonable(data)}, sort_keys=True, separators=(",", ":")))


def write_err(exc: Exception, code: str = "error") -> None:
    print(json.dumps({"ok": False, "error": {"code": code, "message": str(exc)}}, sort_keys=True, separators=(",", ":")), file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rich-media-sidecar", description="Media intelligence JSON CLI")
    sub = p.add_subparsers(dest="command", required=True)

    def provider_args(c: argparse.ArgumentParser) -> None:
        c.add_argument("--provider", choices=("fastembed", "google", "openrouter"), default="fastembed", help="Embedding provider.")
        c.add_argument("--model", default="Qdrant/clip-ViT-B-32", help="Embedding model to use")

    e = sub.add_parser("embed", help="Embed paths and/or text; emits JSON")
    provider_args(e)
    e.add_argument("paths", nargs="*", type=Path)
    e.add_argument("--text", action="append", default=[], help="Text to embed; may be repeated")
    e.add_argument("--json", dest="json_payload", help="JSON request: {paths:[], texts:[]}")
    e.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)), help="Embedding worker threads; kept low because embedding is computationally expensive")
    e.add_argument("--batch-size", type=int, default=None, help="Provider batch size for local embedding models")
    e.add_argument("--image-max-width", type=int, default=None, help="Optional max image pixel width before remote embedding")

    cf = sub.add_parser("cluster-faces", help="Detect faces in images and cluster them")
    cf.add_argument("paths", nargs="*", type=Path)
    cf.add_argument("--threshold", type=float, default=0.42)
    cf.add_argument("--json", dest="json_payload", help="JSON request: {paths:[], threshold?:number}")

    sf = sub.add_parser("serve-faces", help="Read cluster-faces JSON requests from stdin")
    sf.add_argument("--threshold", type=float, default=0.42)

    old = sub.add_parser("cluster", help="Alias for cluster-faces")
    old.add_argument("paths", nargs="*", type=Path)
    old.add_argument("--threshold", type=float, default=0.42)

    fm = sub.add_parser(
        "face-match",
        help="FAISS-backed similarity: read JSON request from stdin (propagate / best_person)",
    )
    fm.add_argument(
        "--stdin",
        action="store_true",
        help="Read one JSON object from stdin (avoids huge argv on Windows)",
    )

    ch = sub.add_parser("convert-heic", help="Convert HEIC/HEIF files to cached JPEG files")
    ch.add_argument("paths", nargs="*", type=Path)
    ch.add_argument("--cache-dir", type=Path, default=None)
    ch.add_argument("--json", dest="json_payload", help="JSON request: {paths:[], cache_dir:string}")

    m = sub.add_parser("embedding-models", help="List common embedding models by provider")

    s = sub.add_parser("semantic-search", help="Rank vectors by cosine similarity to query text")
    provider_args(s)
    s.add_argument("--query", required=True)
    s.add_argument("--vectors", required=True, help="JSON array of {source,embedding} or path to JSON file")
    return p


def _payload(arg: str | None) -> dict:
    if not arg: return {}
    try:
        if arg.startswith("@"):
            return json.loads(Path(arg[1:]).read_text())
        return json.loads(arg)
    except Exception as exc:
        raise ValueError(f"Invalid --json payload: {exc}") from exc


def _load_vectors(value: str) -> list[dict]:
    try:
        text = Path(value).read_text() if Path(value).exists() else value
        obj = json.loads(text)
        if isinstance(obj, dict) and "data" in obj: obj = obj["data"]
        if isinstance(obj, dict) and "embeddings" in obj: obj = obj["embeddings"]
        if not isinstance(obj, list): raise ValueError("vectors must be an array")
        return obj
    except OSError:
        return json.loads(value)


def serve_faces(default_threshold: float) -> int:
    """Process newline-delimited face clustering requests with a warm model."""
    for line in sys.stdin:
        try:
            payload = json.loads(line) if line.strip() else {}
            paths = [Path(p) for p in payload.get("paths", [])]
            threshold = float(payload.get("threshold", default_threshold))
            response = {"ok": True, "data": _jsonable(insightface_cluster_paths(paths, threshold))}
            print(json.dumps(response, sort_keys=True, separators=(",", ":")), flush=True)
        except Exception as exc:
            response = {"ok": False, "error": {"code": "error", "message": str(exc)}}
            print(json.dumps(response, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "embed":
            payload = _payload(args.json_payload)
            paths = [Path(p) for p in payload.get("paths", [])] + list(args.paths)
            texts = list(payload.get("texts", [])) + list(args.text)
            image_max_width = payload.get("image_max_width", args.image_max_width)
            image_max_width = int(image_max_width) if image_max_width else None
            batch_size = payload.get("batch_size", args.batch_size)
            batch_size = int(batch_size) if batch_size else None
            provider = create_provider(args.provider, args.model, image_max_width, batch_size)
            workers = max(1, args.workers)
            write_ok({"provider": provider.name, "workers": workers, "embeddings": embed_paths(paths, provider, workers) + embed_texts(texts, provider, workers)})
            return 0
        if args.command in ("cluster-faces", "cluster"):
            payload = _payload(getattr(args, "json_payload", None))
            paths = [Path(p) for p in payload.get("paths", [])] + list(args.paths)
            threshold = float(payload.get("threshold", args.threshold))
            write_ok(insightface_cluster_paths(paths, threshold))
            return 0
        if args.command == "convert-heic":
            payload = _payload(args.json_payload)
            paths = [Path(p) for p in payload.get("paths", [])] + list(args.paths)
            cache_dir_value = payload.get("cache_dir", args.cache_dir)
            if not cache_dir_value:
                raise ValueError("convert-heic requires --cache-dir or cache_dir in --json")
            conversions = convert_heic_paths(paths, Path(cache_dir_value))
            write_ok({"conversions": conversions})
            return 0
        if args.command == "serve-faces":
            return serve_faces(float(args.threshold))
        if args.command == "face-match":
            if not getattr(args, "stdin", False):
                raise ValueError("face-match requires --stdin")
            body = sys.stdin.read()
            req = json.loads(body) if body.strip() else {}
            write_ok(handle_face_match_request(req))
            return 0
        if args.command == "embedding-models":
            write_ok({"models": {"fastembed": FASTEMBED_EMBEDDING_MODELS, "google": GOOGLE_EMBEDDING_MODELS, "openrouter": OPENROUTER_EMBEDDING_MODELS}})
            return 0
        if args.command == "semantic-search":
            q = create_provider(args.provider, args.model).embed_text(args.query).embedding
            results = []
            for item in _load_vectors(args.vectors):
                emb = item.get("embedding") or item.get("vector")
                if emb:
                    results.append({"source": item.get("source", ""), "score": round(cosine(q, emb), 6)})
            results.sort(key=lambda x: x["score"], reverse=True)
            write_ok({"query": args.query, "results": results})
            return 0
    except RemoteProviderUnavailableError as exc:
        write_err(exc, "remote_unavailable"); return 3
    except Exception as exc:
        write_err(exc, "error"); return 1
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
