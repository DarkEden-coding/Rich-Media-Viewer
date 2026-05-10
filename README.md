# Rich Media Viewer

Local-first desktop app for cataloging and viewing image/video libraries.

## Current implementation

- Tauri v2 + React + TypeScript + Tailwind dark UI.
- Local SQLite catalog in `dev-data/` for debug builds and the OS app data directory for production.
- Folder scanning keeps media in place and stores paths/metadata only.
- Supported first-pass formats: jpg/jpeg, png, gif, webp, heic/heif, mp4, mov, webm.
- Grid/list results, manual setup modal, search panel, and image/video viewer modal.
- Python sidecar skeleton for future local face clustering and embedding providers.

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
cd src-tauri && cargo check --lib
python3 -m compileall python-sidecar/rich_media_sidecar
```

## Notes and limitations

- Setup currently accepts manual absolute folder paths; native folder picker is not wired yet.
- Indexing is synchronous and shown through a blocking progress state in the UI.
- EXIF/GPS/camera fields are represented in the schema/UI but real parsing is still TODO.
- Date, GPS radius, people, and camera UI filters are placeholders until backend search expands beyond filename/type/missing filters.
- Remote embedding providers must require explicit user consent before uploading media.
- The Python sidecar is standalone scaffolding only; it is not packaged or invoked by Tauri yet.
