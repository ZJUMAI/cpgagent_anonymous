"""MedClaw execution entrypoint for radiology.ucec_mri_roi."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SKILL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SKILL_DIR.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medclaw.skills.radiology.ucec_mri_roi.workflow import (  # noqa: E402
    DEFAULT_MODALITIES,
    DEFAULT_TIMEPOINT,
    run_ucec_mri_roi,
)


SKILL_NAME = "radiology.ucec_mri_roi"
SKILL_VERSION = "0.1.0"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Input JSON must contain an object.")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def execute(input_path: Path, output_path: Path) -> dict[str, Any]:
    started_at = now()
    payload = read_json(input_path)
    arguments = payload["arguments"]
    case_id = str(arguments["case_id"])
    modalities = arguments.get("modalities", list(DEFAULT_MODALITIES))
    if isinstance(modalities, str):
        modalities = [modalities]
    workflow = run_ucec_mri_roi(
        case_id,
        output_path.parent,
        timepoint=str(arguments.get("timepoint", DEFAULT_TIMEPOINT)),
        modalities=tuple(str(item).lower() for item in modalities),
        t2_roi_uri=arguments.get("t2_roi_uri"),
        dwi_roi_uri=arguments.get("dwi_roi_uri"),
    )

    artifacts = [
        {
            "type": artifact_type,
            "role": role,
            "path": path.name,
        }
        for artifact_type, role, path in workflow.artifact_files
    ]
    return {
        "status": "success",
        "findings": workflow.findings,
        "artifacts": artifacts,
        "provenance": {
            "skill_name": SKILL_NAME,
            "skill_version": SKILL_VERSION,
            "input_hash": file_hash(input_path),
            "started_at": started_at,
            "finished_at": now(),
        },
        "warnings": list(workflow.warnings),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    started_at = now()
    try:
        result = execute(args.input.resolve(), args.output.resolve())
    except Exception as exc:
        result = {
            "status": "failed",
            "findings": {
                "summary": f"UCEC MRI ROI skill failed: {exc}",
                "roi_available": {},
                "error": str(exc),
            },
            "artifacts": [],
            "provenance": {
                "skill_name": SKILL_NAME,
                "skill_version": SKILL_VERSION,
                "input_hash": file_hash(args.input) if args.input.is_file() else "",
                "started_at": started_at,
                "finished_at": now(),
            },
            "warnings": [str(exc)],
        }
    write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
