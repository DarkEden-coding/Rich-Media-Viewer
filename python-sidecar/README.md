# Rich Media Viewer Python Sidecar

This directory contains a **prototype Python sidecar** for future media intelligence features in Rich Media Viewer. It is intentionally isolated from the frontend (`src/`) and Rust/Tauri backend (`src-tauri/`) so it can evolve independently.

## Current status

The implementation is a skeleton with safe placeholder behavior:

- Local face clustering is represented by deterministic stub logic.
- Embedding providers expose a common abstraction for:
  - `local` placeholder embeddings
  - `google` placeholder provider
  - `openrouter` placeholder provider
- No image bytes are uploaded anywhere by default.
- Cloud providers currently require explicit consent flags and still return placeholder embeddings unless future integration code is added.

## Privacy and consent model

Media libraries can contain sensitive biometric data. Future production use should follow these rules:

1. Prefer local processing for face detection, embeddings, and clustering.
2. Require explicit opt-in before sending any image, face crop, metadata, or embedding to a third-party provider.
3. Clearly display which provider is used and what data leaves the device.
4. Avoid storing raw face crops unless the user enables that behavior.
5. Treat face embeddings as sensitive biometric-derived data.

The current CLI enforces a conservative model: cloud provider classes reject calls unless `allow_remote=True` is passed.

## Install

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No heavy ML dependencies are required yet.

## CLI usage

Show help:

```bash
python -m rich_media_sidecar --help
```

Generate placeholder embeddings for files:

```bash
python -m rich_media_sidecar embed --provider local /path/to/image1.jpg /path/to/image2.jpg
```

Run placeholder clustering:

```bash
python -m rich_media_sidecar cluster /path/to/image1.jpg /path/to/image2.jpg
```

Try a remote provider stub with explicit consent:

```bash
python -m rich_media_sidecar embed --provider openrouter --allow-remote /path/to/image.jpg
```

## Future Tauri integration

A future Rust backend can launch this sidecar as a subprocess and communicate using JSON over stdin/stdout or command arguments. Recommended next steps:

- Define a stable JSON-RPC-like protocol for requests and responses.
- Package the sidecar with the Tauri app bundle.
- Add local model support for face detection and embedding extraction.
- Add secure user settings for provider choice and consent.
- Add integration tests that exercise CLI behavior without requiring cloud credentials.
