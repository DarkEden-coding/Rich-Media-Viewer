# Python Sidecar Architecture Notes

The Python sidecar lives under `python-sidecar/` and is currently a standalone prototype. It is not wired into the Tauri/Rust backend or frontend yet.

## Responsibilities

- Define a provider abstraction for embeddings.
- Keep local processing as the default path.
- Provide explicit consent gates for future remote providers.
- Offer a CLI that can later be launched by Tauri as a subprocess.

## Placeholder behavior

All providers currently generate deterministic pseudo-embeddings from path strings. The sidecar does not read image bytes, detect faces, cluster real identities, or call network APIs.

## Integration direction

Future integration should use a stable JSON protocol between Rust and Python, with user-visible settings for provider choice and cloud consent. Face embeddings should be handled as sensitive biometric-derived data.
