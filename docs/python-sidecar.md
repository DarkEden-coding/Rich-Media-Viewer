# Python Sidecar Architecture Notes

The Python sidecar lives under `python-sidecar/` and is invoked by the Rust/Tauri backend in development.

## Responsibilities

- Local-first media intelligence.
- OpenCV Haar face detection and deterministic face-embedding clustering.
- Local deterministic image/text/video embeddings.
- Remote provider interfaces for Google/OpenRouter-style calls, gated by explicit consent.
- JSON CLI protocol for Rust subprocess integration.

## Commands

- `python3 -m rich_media_sidecar embed --provider local --text "query"`
- `python3 -m rich_media_sidecar embed --provider local /path/to/image.jpg`
- `python3 -m rich_media_sidecar cluster-faces /path/to/image.jpg`
- `python3 -m rich_media_sidecar semantic-search --query "beach" --vectors '[...]'`

Rust commands currently call the sidecar for:

- `cluster_faces`
- `generate_embeddings`
- `search_semantic_text`

## Privacy

Face recognition is local only. Remote embedding providers refuse to run unless the app passes explicit `allow_remote` consent.

## Packaging

Development resolves the sidecar from the repository root. Production bundling still needs Python/runtime resource packaging for macOS and Windows installers.
