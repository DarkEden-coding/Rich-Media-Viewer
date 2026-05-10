# Rich Media Viewer Python Sidecar

Local-first JSON CLI for media intelligence. It does not touch the Tauri/Rust or frontend code.

## Features

- Robust JSON responses: success is `{"ok":true,"data":...}`; errors are `{"ok":false,"error":{"code", "message"}}` on stderr.
- Local embeddings:
  - images: deterministic Pillow/numpy color, histogram, thumbnail, and gradient features
  - text: deterministic hashing-vector embedding
  - videos/other paths: deterministic metadata/path fallback (no ffmpeg required)
- Local face detection and clustering:
  - OpenCV Haar cascade face boxes when `opencv-python-headless` is available
  - deterministic face embeddings from cropped pixels
  - simple DBSCAN-like online cosine clustering
- Remote provider stubs for Google and OpenRouter:
  - require `--allow-remote`
  - require `GOOGLE_API_KEY` or `OPENROUTER_API_KEY`
  - contain real HTTP endpoint structure, but current path embedding does **not** upload media bytes

## Install

```bash
cd python-sidecar
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## CLI

Embed paths and/or text:

```bash
python -m rich_media_sidecar embed /path/a.jpg /path/video.mp4 --text "beach sunset"
```

JSON request form:

```bash
python -m rich_media_sidecar embed --json '{"paths":["/path/a.jpg"],"texts":["cat"]}'
```

Cluster faces:

```bash
python -m rich_media_sidecar cluster-faces /path/a.jpg /path/b.jpg --threshold 0.88
```

Output face records include `path`, `bbox` (`[x,y,width,height]`), `cluster_id`, and `embedding`.

Semantic search over prior embeddings:

```bash
python -m rich_media_sidecar semantic-search --query "red car" --vectors embeddings.json
```

Remote stubs:

```bash
OPENROUTER_API_KEY=... python -m rich_media_sidecar embed --provider openrouter --allow-remote --text "hello"
```

Without `--allow-remote`, remote providers return a JSON error and make no request.

## Privacy

Default processing is local. Face embeddings are biometric-derived data; store and transmit them only with explicit user consent.
