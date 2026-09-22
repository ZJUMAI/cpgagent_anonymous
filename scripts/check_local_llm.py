"""Smoke-test the local OpenAI-compatible model used by CancerClaw."""

from __future__ import annotations

import base64
import json
import struct
import sys
import zlib
from pathlib import Path

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medclaw.llm.providers.local_openai import LocalOpenAIConfig


def _solid_png_data_url(width: int = 64, height: int = 64) -> str:
    row = b"\x00" + (b"\xff\x00\x00" * width)
    raw = row * height

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def main() -> int:
    config = LocalOpenAIConfig.from_env()
    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout_sec,
    )

    model_ids = [item.id for item in client.models.list().data]
    if config.model not in model_ids:
        raise RuntimeError(
            f"Configured model {config.model!r} was not returned by /v1/models: "
            f"{model_ids}"
        )

    tool_response = client.chat.completions.create(
        model=config.model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Call get_case_summary for case TCGA-38-4626. "
                    "Do not answer without calling the tool."
                ),
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_case_summary",
                    "description": "Read a structured cancer case summary.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "case_id": {"type": "string"},
                        },
                        "required": ["case_id"],
                    },
                },
            }
        ],
        tool_choice="auto",
        stream=False,
    )
    tool_calls = tool_response.choices[0].message.tool_calls or []
    if not tool_calls:
        raise RuntimeError(
            "The local model did not return an OpenAI tool_call. Restart vLLM "
            "with --enable-auto-tool-choice --tool-call-parser qwen3_coder."
        )

    vision_response = client.chat.completions.create(
        model=config.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": _solid_png_data_url()},
                    },
                    {
                        "type": "text",
                        "text": "Briefly state the dominant color in this image.",
                    },
                ],
            }
        ],
        stream=False,
    )
    vision_message = vision_response.choices[0].message
    vision_text = vision_message.content or getattr(
        vision_message, "reasoning_content", None
    )
    if not vision_text:
        raise RuntimeError("The local model returned no text for the image request.")

    print(
        json.dumps(
            {
                "status": "success",
                "provider": config.provider,
                "base_url": config.base_url,
                "model": config.model,
                "models": model_ids,
                "tool_call": tool_calls[0].function.model_dump(),
                "vision_response": vision_text,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
