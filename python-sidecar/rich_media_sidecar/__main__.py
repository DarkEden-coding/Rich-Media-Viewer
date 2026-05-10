"""JSON CLI entrypoint for the Rich Media Viewer Python sidecar."""

from __future__ import annotations

import argparse, json, sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .clustering import LocalFaceClusterer, cosine
from .providers import RemoteConsentRequiredError, RemoteProviderUnavailableError, create_provider, embed_paths, embed_texts, text_embedding


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
    p = argparse.ArgumentParser(prog="rich-media-sidecar", description="Local/remote media intelligence JSON CLI")
    sub = p.add_subparsers(dest="command", required=True)

    def provider_args(c: argparse.ArgumentParser) -> None:
        c.add_argument("--provider", choices=("local", "google", "openrouter"), default="local")
        c.add_argument("--allow-remote", action="store_true")

    e = sub.add_parser("embed", help="Embed paths and/or text; emits JSON")
    provider_args(e)
    e.add_argument("paths", nargs="*", type=Path)
    e.add_argument("--text", action="append", default=[], help="Text to embed; may be repeated")
    e.add_argument("--json", dest="json_payload", help="JSON request: {paths:[], texts:[]}")

    cf = sub.add_parser("cluster-faces", help="Detect faces in images and cluster them")
    cf.add_argument("paths", nargs="*", type=Path)
    cf.add_argument("--threshold", type=float, default=0.88)
    cf.add_argument("--json", dest="json_payload", help="JSON request: {paths:[], threshold?:number}")

    old = sub.add_parser("cluster", help="Alias for cluster-faces")
    old.add_argument("paths", nargs="*", type=Path)
    old.add_argument("--threshold", type=float, default=0.88)

    s = sub.add_parser("semantic-search", help="Rank vectors by cosine similarity to query text")
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "embed":
            payload = _payload(args.json_payload)
            paths = [Path(p) for p in payload.get("paths", [])] + list(args.paths)
            texts = list(payload.get("texts", [])) + list(args.text)
            provider = create_provider(args.provider, allow_remote=args.allow_remote)
            write_ok({"provider": provider.name, "embeddings": embed_paths(paths, provider) + embed_texts(texts, provider)})
            return 0
        if args.command in ("cluster-faces", "cluster"):
            payload = _payload(getattr(args, "json_payload", None))
            paths = [Path(p) for p in payload.get("paths", [])] + list(args.paths)
            threshold = float(payload.get("threshold", args.threshold))
            write_ok(LocalFaceClusterer(threshold=threshold).cluster_paths(paths))
            return 0
        if args.command == "semantic-search":
            q = text_embedding(args.query)
            results = []
            for item in _load_vectors(args.vectors):
                emb = item.get("embedding") or item.get("vector")
                if emb:
                    results.append({"source": item.get("source", ""), "score": round(cosine(q, emb), 6)})
            results.sort(key=lambda x: x["score"], reverse=True)
            write_ok({"query": args.query, "results": results})
            return 0
    except RemoteConsentRequiredError as exc:
        write_err(exc, "remote_consent_required"); return 2
    except RemoteProviderUnavailableError as exc:
        write_err(exc, "remote_unavailable"); return 3
    except Exception as exc:
        write_err(exc, "error"); return 1
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
