# Rich Media Viewer Python Sidecar

JSON CLI for media intelligence. In development it is invoked by the Tauri/Rust backend for face clustering, embedding generation, and semantic text search.

## Features

- Robust JSON responses: success is `{"ok":true,"data":...}`; errors are `{"ok":false,"error":{"code", "message"}}` on stderr.
- Embedding providers:
  - FastEmbed local CLIP image embeddings with text query embeddings
  - Google Gemini embeddings through `models.embedContent`
  - OpenRouter embeddings through its OpenAI-compatible `/embeddings` endpoint
- Local face detection and clustering:
  - OpenCV Haar cascade face boxes when `opencv-python-headless` is available
  - deterministic face embeddings from cropped pixels
  - simple DBSCAN-like online cosine clustering
- Capability-aware media embedding:
  - FastEmbed `Qdrant/clip-ViT-B-32` embeds local images and text queries in a shared CLIP vector space
  - Google `gemini-embedding-2` sends supported image, video, audio, and PDF bytes to Gemini
  - OpenRouter `google/gemini-embedding-2-preview` sends supported image bytes through OpenRouter's multimodal embedding input format
  - Text-only models and unsupported media files are skipped instead of inventing proxy vectors

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
python -m rich_media_sidecar embed --provider google --model gemini-embedding-2 /path/a.jpg /path/video.mp4 --text "beach sunset"
```

Resize large images before remote embedding:

```bash
python -m rich_media_sidecar embed --provider openrouter --model google/gemini-embedding-2-preview --image-max-width 1024 /path/a.jpg
```

JSON request form:

```bash
python -m rich_media_sidecar embed --provider fastembed --model Qdrant/clip-ViT-B-32 --json '{"paths":["/path/a.jpg"],"texts":["cat"]}'
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

OpenRouter text embeddings:

```bash
OPENROUTER_API_KEY=... python -m rich_media_sidecar embed --provider openrouter --model google/gemini-embedding-2-preview /path/a.jpg --text "hello"
```

## Privacy

Face embeddings are biometric-derived data and remain local. Google and OpenRouter embedding providers send selected query text and supported media inputs to their APIs when configured.
