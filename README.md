# Rich Media Viewer

Local-first desktop app for cataloging, searching, and viewing image/video libraries.

## Implemented

- Tauri v2 + React + TypeScript + Tailwind dark UI.
- Local SQLite catalog in `dev-data/` for debug builds and the OS app data directory for production.
- User-selected media folders are persisted and media stays in place.
- Native folder picker plus manual path entry.
- Synchronous indexing with a blocking progress/status UI.
- Supported first-pass formats: jpg/jpeg, png, gif, webp, heic/heif, mp4, mov, webm.
- EXIF extraction for image capture date, camera make/model, lens, and GPS where available.
- Unified filters for filename/path, date range, GPS radius, person, media type, missing status, camera, has-GPS, and has-camera.
- Grid/list results, image/video viewer, and OpenStreetMap links/embeds for GPS-tagged media.
- Python sidecar integration for OpenCV face detection/clustering and provider-backed embeddings.
- People naming/search through detected clusters.
- Embedding generation and semantic text search over stored vectors with Ollama, Google, or OpenRouter providers.

## Run in development

```bash
npm install
npm run tauri dev
```

For frontend-only iteration:

```bash
npm run dev
```

## Validate

```bash
npm run build
(cd src-tauri && cargo check --lib)
python3 -m compileall python-sidecar/rich_media_sidecar
```

## Python sidecar dependencies

The sidecar uses `numpy`, `Pillow`, and `opencv-python-headless` for local face detection and clustering. Install them for full face-recognition functionality:

```bash
python3 -m pip install -r python-sidecar/requirements.txt
```

## Production packaging note

Development resolves the sidecar from the repo's `python-sidecar/` directory. Packaged app bundling still needs a production resource/binary packaging step for Python and native dependencies.
