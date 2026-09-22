"""Shared image encoding helpers for multimodal LLM requests."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path


MAX_BASE64_IMAGE_BYTES = 7 * 1024 * 1024
SUPPORTED_IMAGE_MIME_TYPES = {
    "image/bmp",
    "image/heic",
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/webp",
}


def image_to_data_url(
    path: Path,
    *,
    max_bytes: int = MAX_BASE64_IMAGE_BYTES,
) -> str:
    """Encode a local image as a data URL accepted by compatible APIs."""

    image_path = Path(path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    size = image_path.stat().st_size
    if size > max_bytes:
        raise ValueError(
            f"Image file is too large for base64 API upload: {size} bytes; "
            f"maximum is {max_bytes} bytes."
        )

    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type == "image/jpg":
        mime_type = "image/jpeg"
    if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
        raise ValueError(
            f"Unsupported image format for {image_path}; detected MIME type {mime_type!r}."
        )

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def is_supported_image_url(value: str) -> bool:
    return value.startswith("https://") or value.startswith("data:image/")


def parse_data_url(value: str) -> tuple[str, bytes]:
    """Return MIME type and raw bytes from a data URL."""

    if not value.startswith("data:"):
        raise ValueError("Expected a data URL.")
    header, _, payload = value.partition(",")
    if not payload:
        raise ValueError("Data URL is missing payload.")
    mime_type = header[5:].split(";", 1)[0].strip()
    if ";base64" in header:
        return mime_type, base64.b64decode(payload)
    return mime_type, payload.encode("utf-8")
