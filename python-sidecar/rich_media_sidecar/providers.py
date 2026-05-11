"""Embedding providers for Rich Media Viewer."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import base64
import json
import mimetypes
import os
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol
from urllib import request, error

from PIL import Image

OLLAMA_EMBEDDING_MODELS = [
    "nomic-embed-text",
    "mxbai-embed-large",
    "snowflake-arctic-embed",
    "all-minilm",
]

GOOGLE_EMBEDDING_MODELS = [
    "gemini-embedding-2",
    "gemini-embedding-001",
    "text-embedding-004",
]

OPENROUTER_EMBEDDING_MODELS = [
    "google/gemini-embedding-2-preview",
    "openai/text-embedding-3-small",
    "openai/text-embedding-3-large",
]

GOOGLE_MULTIMODAL_MODEL = "gemini-embedding-2"
GOOGLE_TEXT_ONLY_MODELS = {"gemini-embedding-001", "text-embedding-004", "embedding-001"}
GOOGLE_SUPPORTED_MIME_PREFIXES = ("image/", "video/", "audio/")
GOOGLE_SUPPORTED_MIME_TYPES = {"application/pdf"}
OPENROUTER_DEFAULT_MODEL = "google/gemini-embedding-2-preview"
OPENROUTER_MODEL_MODALITIES = {
    "google/gemini-embedding-2-preview": {"text", "image"},
    "openai/text-embedding-3-small": {"text"},
    "openai/text-embedding-3-large": {"text"},
}


@dataclass(frozen=True)
class EmbeddingResult:
    source: str
    provider: str
    embedding: list[float]
    kind: str = "text"
    model: str = "unknown"
    error: str | None = None


class EmbeddingProvider(Protocol):
    name: str
    def embed_path(self, path: Path) -> EmbeddingResult: ...
    def embed_text(self, text: str, source: str | None = None) -> EmbeddingResult: ...


class RemoteProviderUnavailableError(RuntimeError): pass


def _model_name(model: str) -> str:
    return model if model.startswith("models/") else f"models/{model}"


def _guess_mime(path: Path) -> str | None:
    mime, _ = mimetypes.guess_type(path.name)
    return mime


def _unsupported(path: Path, provider: str, model: str, message: str) -> EmbeddingResult:
    return EmbeddingResult(str(path), provider, [], kind="unsupported", model=model, error=message)


def _image_bytes_for_embedding(path: Path, mime: str, max_width: int | None) -> tuple[str, str]:
    if not max_width or max_width <= 0:
        return mime, base64.b64encode(path.read_bytes()).decode("ascii")
    with Image.open(path) as img:
        if img.width <= max_width:
            return mime, base64.b64encode(path.read_bytes()).decode("ascii")
        height = max(1, round(img.height * (max_width / img.width)))
        resized = img.copy()
        resized.thumbnail((max_width, height), Image.Resampling.LANCZOS)
        output = BytesIO()
        output_mime = mime if mime in {"image/jpeg", "image/png", "image/webp"} else "image/jpeg"
        format_name = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}[output_mime]
        if output_mime == "image/jpeg" and resized.mode not in ("RGB", "L"):
            resized = resized.convert("RGB")
        elif output_mime in {"image/png", "image/webp"} and resized.mode == "P":
            resized = resized.convert("RGBA")
        save_kwargs = {"quality": 85, "optimize": True} if output_mime in {"image/jpeg", "image/webp"} else {"optimize": True}
        resized.save(output, format=format_name, **save_kwargs)
        return output_mime, base64.b64encode(output.getvalue()).decode("ascii")


class OllamaEmbeddingProvider:
    """Local Ollama embeddings provider.

    Ollama's embedding endpoint accepts text input. Media bytes are intentionally
    not proxied through filename/metadata embeddings; unsupported media paths are
    skipped so the app only embeds formats supported by the selected model/API.
    """

    name = "ollama"

    def __init__(self, model: str = "nomic-embed-text", base_url: str | None = None) -> None:
        self.model = model.strip() or "nomic-embed-text"
        self.base_url = (base_url or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")

    def _post(self, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(f"{self.base_url}/api/embed", data=data, headers={"Content-Type": "application/json"})
        try:
            with request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.URLError as exc:
            raise RemoteProviderUnavailableError(f"Ollama request failed. Is Ollama running and is model '{self.model}' pulled? {exc}") from exc

    def embed_path(self, path: Path) -> EmbeddingResult:
        return EmbeddingResult(str(path), self.name, [], kind="unsupported", model=self.model, error="Ollama embedding models accept text input only; media files are not proxied into embeddings.")

    def embed_text(self, text: str, source: str | None = None) -> EmbeddingResult:
        root = self._post({"model": self.model, "input": text})
        embeddings = root.get("embeddings") or []
        if not embeddings:
            raise RemoteProviderUnavailableError(f"Ollama model '{self.model}' did not return an embedding.")
        return EmbeddingResult(source or text[:80], self.name, [float(x) for x in embeddings[0]], kind="text", model=self.model)


class GoogleEmbeddingProvider:
    """Google Gemini embeddings provider."""

    name = "google"

    def __init__(self, model: str = GOOGLE_MULTIMODAL_MODEL, image_max_width: int | None = None) -> None:
        self.model = (model or GOOGLE_MULTIMODAL_MODEL).strip()
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
        self.image_max_width = image_max_width

    def _post(self, payload: dict) -> dict:
        if not self.api_key:
            raise RemoteProviderUnavailableError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set.")
        
        url = f"https://generativelanguage.googleapis.com/v1beta/{_model_name(self.model)}:embedContent"
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=data, headers={"Content-Type": "application/json", "x-goog-api-key": self.api_key})
        try:
            with request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RemoteProviderUnavailableError(f"Google API request failed ({exc.code}): {details}") from exc
        except error.URLError as exc:
            raise RemoteProviderUnavailableError(f"Google API request failed: {exc}") from exc

    def _embedding_from_content(self, content: dict, source: str, kind: str) -> EmbeddingResult:
        res = self._post({"content": content})
        embedding = res.get("embedding", {}).get("values")
        if not embedding:
            raise RemoteProviderUnavailableError(f"Google API returned no embedding for model {self.model}")
        return EmbeddingResult(source, self.name, [float(x) for x in embedding], kind=kind, model=self.model)

    def embed_path(self, path: Path) -> EmbeddingResult:
        if self.model in GOOGLE_TEXT_ONLY_MODELS:
            return _unsupported(path, self.name, self.model, f"Google model '{self.model}' accepts text input only.")
        mime = _guess_mime(path)
        if not mime or (not mime.startswith(GOOGLE_SUPPORTED_MIME_PREFIXES) and mime not in GOOGLE_SUPPORTED_MIME_TYPES):
            return _unsupported(path, self.name, self.model, f"Unsupported media MIME type for Google embeddings: {mime or 'unknown'}.")
        try:
            if mime.startswith("image/"):
                mime, data = _image_bytes_for_embedding(path, mime, self.image_max_width)
            else:
                data = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError as exc:
            return _unsupported(path, self.name, self.model, f"Failed to read media file: {exc}")
        except Exception as exc:
            return _unsupported(path, self.name, self.model, f"Failed to prepare media file: {exc}")
        content = {"parts": [{"inlineData": {"mimeType": mime, "data": data}}]}
        return self._embedding_from_content(content, str(path), mime.split("/", 1)[0])

    def embed_text(self, text: str, source: str | None = None) -> EmbeddingResult:
        payload = {
            "content": {"parts": [{"text": text}]},
            "taskType": "RETRIEVAL_QUERY"
        }
        res = self._post(payload)
        embedding = res.get("embedding", {}).get("values")
        if not embedding:
            raise RemoteProviderUnavailableError(f"Google API returned no embedding for model {self.model}")
        return EmbeddingResult(source or text[:80], self.name, [float(x) for x in embedding], kind="text", model=self.model)


class OpenRouterEmbeddingProvider:
    """OpenRouter embeddings provider."""

    name = "openrouter"

    def __init__(self, model: str = OPENROUTER_DEFAULT_MODEL, image_max_width: int | None = None) -> None:
        self.model = (model or OPENROUTER_DEFAULT_MODEL).strip()
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
        self.modalities = OPENROUTER_MODEL_MODALITIES.get(self.model, {"text", "image"})
        self.image_max_width = image_max_width

    def _post(self, payload: dict) -> dict:
        if not self.api_key:
            raise RemoteProviderUnavailableError("OPENROUTER_API_KEY environment variable not set.")
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/embeddings",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/rich-media-viewer",
                "X-Title": "Rich Media Viewer",
            },
        )
        try:
            with request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RemoteProviderUnavailableError(f"OpenRouter API request failed ({exc.code}): {details}") from exc
        except error.URLError as exc:
            raise RemoteProviderUnavailableError(f"OpenRouter API request failed: {exc}") from exc

    def _embedding_from_input(self, input_value: str | list[dict], source: str, kind: str) -> EmbeddingResult:
        res = self._post({"model": self.model, "input": input_value, "encoding_format": "float"})
        data = res.get("data") or []
        embedding = data[0].get("embedding") if data else None
        if not embedding:
            raise RemoteProviderUnavailableError(f"OpenRouter model '{self.model}' did not return an embedding.")
        return EmbeddingResult(source, self.name, [float(x) for x in embedding], kind=kind, model=self.model)

    def embed_path(self, path: Path) -> EmbeddingResult:
        mime = _guess_mime(path)
        if mime and mime.startswith("image/") and "image" in self.modalities:
            try:
                mime, data = _image_bytes_for_embedding(path, mime, self.image_max_width)
            except OSError as exc:
                return _unsupported(path, self.name, self.model, f"Failed to read media file: {exc}")
            except Exception as exc:
                return _unsupported(path, self.name, self.model, f"Failed to prepare media file: {exc}")
            input_value = [{"content": [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}]}]
            return self._embedding_from_input(input_value, str(path), "image")
        if mime and mime.startswith("image/"):
            return _unsupported(path, self.name, self.model, f"OpenRouter model '{self.model}' does not advertise image embedding support.")
        return _unsupported(path, self.name, self.model, f"OpenRouter embeddings currently document text and image inputs; unsupported MIME type: {mime or 'unknown'}.")

    def embed_text(self, text: str, source: str | None = None) -> EmbeddingResult:
        return self._embedding_from_input(text, source or text[:80], "text")


def create_provider(name: str, model: str | None = None, image_max_width: int | None = None) -> EmbeddingProvider:
    n = name.strip().lower()
    if n == "google":
        return GoogleEmbeddingProvider(model, image_max_width)
    if n == "openrouter":
        return OpenRouterEmbeddingProvider(model or OPENROUTER_DEFAULT_MODEL, image_max_width)
    if n == "ollama":
        return OllamaEmbeddingProvider(model or "nomic-embed-text")
    raise ValueError(f"Unknown embedding provider: {name}")


def embed_paths(paths: Iterable[Path], provider: EmbeddingProvider, max_workers: int = 1) -> list[EmbeddingResult]:
    items = list(paths)
    if max_workers <= 1 or len(items) <= 1:
        return [provider.embed_path(p) for p in items]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(provider.embed_path, items))


def embed_texts(texts: Iterable[str], provider: EmbeddingProvider, max_workers: int = 1) -> list[EmbeddingResult]:
    items = list(texts)
    if max_workers <= 1 or len(items) <= 1:
        return [provider.embed_text(t) for t in items]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(provider.embed_text, items))
