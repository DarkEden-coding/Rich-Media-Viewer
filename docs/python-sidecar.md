# Python Sidecar Architecture Notes

The Python sidecar lives under `python-sidecar/` and is invoked by the Rust/Tauri backend in development.

## Responsibilities

- Media intelligence for local indexing and optional remote embeddings.
- OpenCV Haar face detection and deterministic face-embedding clustering.
- Embedding provider interfaces for local FastEmbed, Google, and OpenRouter.
- Capability-aware media embedding. OpenRouter uses its documented multimodal embedding format for image-capable models such as `google/gemini-embedding-2-preview`.
- JSON CLI protocol for Rust subprocess integration.

## Commands

- `python3 -m rich_media_sidecar embed --provider fastembed --model Qdrant/clip-ViT-B-32 /path/to/image.jpg --text "query"`
- `python3 -m rich_media_sidecar embed --provider google --model gemini-embedding-2 /path/to/image.jpg`
- `python3 -m rich_media_sidecar embed --provider openrouter --model google/gemini-embedding-2-preview /path/to/image.jpg`
- `python3 -m rich_media_sidecar cluster-faces /path/to/image.jpg`
- `python3 -m rich_media_sidecar semantic-search --query "beach" --vectors '[...]'`

Rust commands currently call the sidecar for:

- `cluster_faces`
- `generate_embeddings`
- `search_semantic_text`

## Privacy

Face recognition is local only. Google and OpenRouter embedding providers send selected query text and supported media inputs to remote APIs when the user selects and configures those providers.
FastEmbed embeddings remain local and use CUDA when ONNX Runtime can load the CUDA execution provider; otherwise they fall back to CPU.

## Packaging

Development resolves the sidecar from the repository root. Production bundling still needs Python/runtime resource packaging for macOS and Windows installers.
