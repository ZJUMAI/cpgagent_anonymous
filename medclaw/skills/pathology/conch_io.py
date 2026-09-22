"""Shared, metadata-safe helpers for cached CONCH ROI packets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_METADATA_CHUNKS = {b"tEXt", b"zTXt", b"iTXt", b"iCCP"}


def select_prompt_ids(
    prompt_entries: Mapping[str, Mapping[str, Any]],
    *,
    prompt_ids: Iterable[str] | None,
    default_prompt_ids: Iterable[str],
    fallback_limit: int = 2,
) -> tuple[list[str], list[str]]:
    """Resolve requested prompts, falling back when a packet uses different IDs."""

    warnings: list[str] = []
    defaults = [str(item) for item in default_prompt_ids]
    requested = (
        [str(item) for item in prompt_ids]
        if prompt_ids is not None
        else [item for item in defaults if item in prompt_entries]
    )
    if prompt_ids is None and not requested:
        requested = list(prompt_entries)[:fallback_limit]

    selected: list[str] = []
    for prompt_id in requested:
        if prompt_id not in prompt_entries:
            warnings.append(f"Requested CONCH prompt id not found: {prompt_id}")
            continue
        if prompt_id not in selected:
            selected.append(prompt_id)

    if requested and not selected and prompt_entries:
        fallback = _ranked_fallback_ids(prompt_entries, fallback_limit)
        selected.extend(fallback)
        warnings.append(
            "None of the requested CONCH prompt IDs matched this cached packet; "
            f"fell back to available prompt IDs: {', '.join(fallback)}"
        )
    return selected, warnings


def copy_png_pixels(source: Path, destination: Path) -> tuple[int, int]:
    """Copy PNG pixels while discarding ancillary text chunks.

    Some WSI patch exporters embed very large compressed text fields. Pillow applies
    a deliberately small decompression limit to those fields even though CONCH only
    needs the pixels. Rebuilding the PNG stream without text or ICC metadata avoids
    globally weakening Pillow's safety limits and preserves all pixel-bearing chunks.
    """

    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _png_without_text_chunks(source) as image_stream, Image.open(
        image_stream
    ) as image:
        image.load()
        rgb = image.convert("RGB")
        size = rgb.size
        rgb.save(destination)
    return size


def _ranked_fallback_ids(
    prompt_entries: Mapping[str, Mapping[str, Any]],
    limit: int,
) -> list[str]:
    with_rankings = [
        prompt_id
        for prompt_id, prompt in prompt_entries.items()
        if isinstance(prompt.get("top_indices"), list) and prompt["top_indices"]
    ]
    candidates = with_rankings or list(prompt_entries)
    return candidates[:limit]


def _png_without_text_chunks(source: Path) -> BytesIO:
    stream = BytesIO()
    with source.open("rb") as handle:
        signature = handle.read(len(PNG_SIGNATURE))
        if signature != PNG_SIGNATURE:
            raise ValueError(f"ROI image is not a valid PNG file: {source}")
        stream.write(signature)

        saw_iend = False
        while True:
            header = handle.read(8)
            if not header:
                break
            if len(header) != 8:
                raise ValueError(f"Truncated PNG chunk header: {source}")
            length = int.from_bytes(header[:4], "big")
            chunk_type = header[4:]
            payload_and_crc = handle.read(length + 4)
            if len(payload_and_crc) != length + 4:
                raise ValueError(f"Truncated PNG chunk payload: {source}")
            if chunk_type not in PNG_METADATA_CHUNKS:
                stream.write(header)
                stream.write(payload_and_crc)
            if chunk_type == b"IEND":
                saw_iend = True
                break

    if not saw_iend:
        raise ValueError(f"PNG file has no IEND chunk: {source}")
    stream.seek(0)
    return stream
