"""Embedding providers for Rich Media Viewer."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import base64
import json
import mimetypes
import os
import sysconfig
import threading
import tempfile
import ctypes
import ctypes.util
import contextlib
import warnings
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol
from urllib import request, error

from PIL import Image

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass

os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")
warnings.filterwarnings("ignore", message="Cannot enable progress bars.*", category=UserWarning)

_DLL_DIRECTORY_HANDLES: list[object] = []

FASTEMBED_EMBEDDING_MODELS = [
    "Qdrant/clip-ViT-B-32",
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


def _add_nvidia_dll_directories() -> None:
    if os.name != "nt":
        return
    site_paths = {
        Path(p)
        for p in sysconfig.get_paths().values()
        if p
    }
    for site_path in list(site_paths):
        site_paths.add(site_path.parent)
    candidate_dirs: list[Path] = []
    for site_path in site_paths:
        nvidia_root = site_path / "nvidia"
        if not nvidia_root.exists():
            continue
        for child in nvidia_root.iterdir():
            for name in ("bin", "lib"):
                path = child / name
                if path.is_dir():
                    candidate_dirs.append(path)
    for path in candidate_dirs:
        path_str = str(path)
        if path_str not in os.environ.get("PATH", ""):
            os.environ["PATH"] = path_str + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            try:
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(path_str))
            except OSError:
                pass


@contextlib.contextmanager
def _suppress_native_stderr():
    if os.getenv("RMV_SUPPRESS_NATIVE_STDERR", "1").lower() in {"0", "false", "no"}:
        yield
        return
    try:
        stderr_fd = 2
        saved_fd = os.dup(stderr_fd)
        with open(os.devnull, "wb") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            try:
                yield
            finally:
                os.dup2(saved_fd, stderr_fd)
                os.close(saved_fd)
    except Exception:
        yield


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


def _resize_image_to_temp_file(path: Path, mime: str, max_width: int) -> Path:
    with Image.open(path) as img:
        if img.width <= max_width:
            return path
        height = max(1, round(img.height * (max_width / img.width)))
        resized = img.copy()
        resized.thumbnail((max_width, height), Image.Resampling.LANCZOS)
        suffix = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
        }.get(mime, ".jpg")
        output_mime = mime if suffix != ".jpg" else "image/jpeg"
        format_name = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}[output_mime]
        if output_mime == "image/jpeg" and resized.mode not in ("RGB", "L"):
            resized = resized.convert("RGB")
        elif output_mime in {"image/png", "image/webp"} and resized.mode == "P":
            resized = resized.convert("RGBA")
        tmp = tempfile.NamedTemporaryFile(prefix="rmv-fastembed-", suffix=suffix, delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        save_kwargs = {"quality": 85, "optimize": True} if output_mime in {"image/jpeg", "image/webp"} else {"optimize": True}
        resized.save(tmp_path, format=format_name, **save_kwargs)
        return tmp_path


class FastEmbedEmbeddingProvider:
    """Local FastEmbed CLIP provider for image embeddings and text queries."""

    name = "fastembed"

    def __init__(
        self,
        model: str = "Qdrant/clip-ViT-B-32",
        image_max_width: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.model = self._canonical_model(model)
        self.image_model_name = f"{self.model}-vision"
        self.text_model_name = f"{self.model}-text"
        self.image_max_width = image_max_width
        self.batch_size = max(1, batch_size or int(os.getenv("RMV_FASTEMBED_BATCH_SIZE", "16")))
        self.parallel = max(1, int(os.getenv("RMV_FASTEMBED_PARALLEL", "1")))
        self.threads = max(1, int(os.getenv("RMV_FASTEMBED_THREADS", str(os.cpu_count() or 4))))
        self.device = (os.getenv("RMV_FASTEMBED_DEVICE") or "auto").strip().lower()
        self.providers = self._execution_providers()
        self._image_model = None
        self._text_model = None
        self._lock = threading.Lock()

    @staticmethod
    def _canonical_model(model: str | None) -> str:
        name = (model or "Qdrant/clip-ViT-B-32").strip() or "Qdrant/clip-ViT-B-32"
        if name.endswith("-vision"):
            return name.removesuffix("-vision")
        if name.endswith("-text"):
            return name.removesuffix("-text")
        return name

    def _execution_providers(self) -> list[str]:
        if self.device == "cpu":
            return ["CPUExecutionProvider"]
        cuda_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        cuda_ready, reason = self._cuda_runtime_ready()
        if self.device == "cuda":
            if not cuda_ready:
                raise RemoteProviderUnavailableError(f"FastEmbed CUDA requested but unavailable: {reason}")
            return cuda_providers
        if not cuda_ready:
            return ["CPUExecutionProvider"]
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" in ort.get_available_providers():
                return cuda_providers
        except Exception:
            pass
        return ["CPUExecutionProvider"]

    @staticmethod
    def _cuda_runtime_ready() -> tuple[bool, str]:
        _add_nvidia_dll_directories()
        try:
            import onnxruntime as ort
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                return False, "ONNX Runtime CUDAExecutionProvider is not installed."
        except Exception as exc:
            return False, f"failed to inspect ONNX Runtime providers: {exc}"
        if os.name == "nt":
            missing = []
            for dll in ("cudnn64_9.dll",):
                found = ctypes.util.find_library(dll) or any(
                    Path(part).joinpath(dll).exists()
                    for part in os.environ.get("PATH", "").split(os.pathsep)
                    if part
                )
                if not found:
                    missing.append(dll)
            if missing:
                return False, f"missing {', '.join(missing)} on PATH."
        try:
            ctypes.CDLL("cudnn64_9.dll" if os.name == "nt" else "libcudnn.so.9")
        except OSError as exc:
            return False, f"cuDNN runtime could not be loaded: {exc}"
        return True, "CUDA runtime is available."

    def _get_image_model(self):
        if self._image_model is None:
            try:
                from fastembed import ImageEmbedding
            except ImportError as exc:
                raise RemoteProviderUnavailableError("FastEmbed is not installed. Install python-sidecar dependencies before using local FastEmbed embeddings.") from exc
            with _suppress_native_stderr():
                self._image_model = ImageEmbedding(
                    self.image_model_name,
                    threads=self.threads,
                    providers=self.providers,
                )
        return self._image_model

    def _get_text_model(self):
        if self._text_model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise RemoteProviderUnavailableError("FastEmbed is not installed. Install python-sidecar dependencies before using local FastEmbed embeddings.") from exc
            with _suppress_native_stderr():
                self._text_model = TextEmbedding(
                    self.text_model_name,
                    threads=self.threads,
                    providers=self.providers,
                )
        return self._text_model

    def embed_path(self, path: Path) -> EmbeddingResult:
        mime = _guess_mime(path)
        if not mime or not mime.startswith("image/"):
            return _unsupported(path, self.name, self.model, f"FastEmbed local CLIP embeddings support images only; unsupported MIME type: {mime or 'unknown'}.")
        try:
            embed_path = path
            cleanup_path: Path | None = None
            if self.image_max_width and self.image_max_width > 0:
                embed_path = _resize_image_to_temp_file(path, mime, self.image_max_width)
                if embed_path != path:
                    cleanup_path = embed_path
            with self._lock:
                vector = list(self._get_image_model().embed([str(embed_path)]))[0]
            return EmbeddingResult(str(path), self.name, [float(x) for x in vector], kind="image", model=self.model)
        except OSError as exc:
            return _unsupported(path, self.name, self.model, f"Failed to read image file: {exc}")
        except Exception as exc:
            return _unsupported(path, self.name, self.model, f"FastEmbed image embedding failed: {exc}")
        finally:
            if "cleanup_path" in locals() and cleanup_path:
                try:
                    cleanup_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def embed_text(self, text: str, source: str | None = None) -> EmbeddingResult:
        try:
            with self._lock:
                vector = list(self._get_text_model().embed([text]))[0]
            return EmbeddingResult(source or text[:80], self.name, [float(x) for x in vector], kind="text", model=self.model)
        except Exception as exc:
            raise RemoteProviderUnavailableError(f"FastEmbed text embedding failed: {exc}") from exc

    def embed_paths(self, paths: Iterable[Path], max_workers: int = 1) -> list[EmbeddingResult]:
        items = list(paths)
        supported: list[tuple[Path, Path]] = []
        cleanup: list[Path] = []
        results: list[EmbeddingResult] = []
        for path in items:
            mime = _guess_mime(path)
            if not mime or not mime.startswith("image/"):
                results.append(_unsupported(path, self.name, self.model, f"FastEmbed local CLIP embeddings support images only; unsupported MIME type: {mime or 'unknown'}."))
                continue
            try:
                embed_path = path
                if self.image_max_width and self.image_max_width > 0:
                    embed_path = _resize_image_to_temp_file(path, mime, self.image_max_width)
                    if embed_path != path:
                        cleanup.append(embed_path)
                supported.append((path, embed_path))
            except OSError as exc:
                results.append(_unsupported(path, self.name, self.model, f"Failed to read image file: {exc}"))
            except Exception as exc:
                results.append(_unsupported(path, self.name, self.model, f"Failed to prepare image file: {exc}"))
        if supported:
            results.extend(self._embed_supported_batch(supported, max_workers))
        for path in cleanup:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        return results

    def _embed_supported_batch(self, supported: list[tuple[Path, Path]], max_workers: int) -> list[EmbeddingResult]:
        try:
            embed_inputs = [str(embed_path) for _, embed_path in supported]
            parallel = self.parallel if "CUDAExecutionProvider" in self.providers else (max_workers if max_workers > 1 else self.parallel)
            with self._lock:
                vectors = list(
                    self._get_image_model().embed(
                        embed_inputs,
                        batch_size=self.batch_size,
                        parallel=parallel,
                    )
                )
            return [
                EmbeddingResult(str(source_path), self.name, [float(x) for x in vector], kind="image", model=self.model)
                for (source_path, _), vector in zip(supported, vectors)
            ]
        except Exception:
            if len(supported) <= 1:
                source_path, embed_path = supported[0]
                try:
                    with self._lock:
                        vector = list(self._get_image_model().embed([str(embed_path)], batch_size=1, parallel=None))[0]
                    return [EmbeddingResult(str(source_path), self.name, [float(x) for x in vector], kind="image", model=self.model)]
                except Exception as exc:
                    return [_unsupported(source_path, self.name, self.model, f"FastEmbed image embedding failed: {exc}")]
            midpoint = len(supported) // 2
            return self._embed_supported_batch(supported[:midpoint], max_workers) + self._embed_supported_batch(supported[midpoint:], max_workers)


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


def create_provider(
    name: str,
    model: str | None = None,
    image_max_width: int | None = None,
    batch_size: int | None = None,
) -> EmbeddingProvider:
    n = name.strip().lower()
    if n == "fastembed":
        return FastEmbedEmbeddingProvider(model or "Qdrant/clip-ViT-B-32", image_max_width, batch_size)
    if n == "google":
        return GoogleEmbeddingProvider(model, image_max_width)
    if n == "openrouter":
        return OpenRouterEmbeddingProvider(model or OPENROUTER_DEFAULT_MODEL, image_max_width)
    raise ValueError(f"Unknown embedding provider: {name}")


def embed_paths(paths: Iterable[Path], provider: EmbeddingProvider, max_workers: int = 1) -> list[EmbeddingResult]:
    items = list(paths)
    provider_batch_embed = getattr(provider, "embed_paths", None)
    if callable(provider_batch_embed):
        return provider_batch_embed(items, max_workers)
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
