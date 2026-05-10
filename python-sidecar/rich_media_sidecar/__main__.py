"""CLI entrypoint for the Rich Media Viewer Python sidecar."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .clustering import LocalFaceClusterer
from .providers import RemoteConsentRequiredError, create_provider, embed_paths


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


def _write_json(value: Any) -> None:
    print(json.dumps(_to_jsonable(value), indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rich-media-sidecar",
        description="Prototype Python sidecar for Rich Media Viewer media intelligence.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_provider_args(command: argparse.ArgumentParser) -> None:
        command.add_argument("paths", nargs="+", type=Path, help="Media file paths")
        command.add_argument(
            "--provider",
            choices=("local", "google", "openrouter"),
            default="local",
            help="Embedding provider to use (default: local)",
        )
        command.add_argument(
            "--allow-remote",
            action="store_true",
            help="Explicitly consent to remote provider use. Current remote providers are stubs and do not upload media.",
        )

    embed_parser = subparsers.add_parser("embed", help="Generate placeholder embeddings")
    add_provider_args(embed_parser)

    cluster_parser = subparsers.add_parser("cluster", help="Run placeholder face clustering")
    add_provider_args(cluster_parser)
    cluster_parser.add_argument(
        "--buckets",
        type=int,
        default=3,
        help="Number of deterministic placeholder buckets (default: 3)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        provider = create_provider(args.provider, allow_remote=args.allow_remote)
        if args.command == "embed":
            _write_json(embed_paths(args.paths, provider))
            return 0
        if args.command == "cluster":
            result = LocalFaceClusterer(provider, buckets=args.buckets).cluster_paths(args.paths)
            _write_json(result)
            return 0
    except (RemoteConsentRequiredError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
