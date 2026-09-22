"""UCEC pathology patch ROI selection from cached CONCH prompt scores."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import unquote, urlparse

from PIL import Image, ImageDraw

from medclaw.skills.pathology.conch_io import copy_png_pixels, select_prompt_ids
from medclaw.utils import read_json, read_yaml, utc_now, write_json


SKILL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SKILL_DIR.parents[3]
DEFAULT_PROMPT_IDS = ("endometrioid_tumor", "tumor_glands")
DEFAULT_TOP_K_PER_PROMPT = 3
MAX_TOP_K_PER_PROMPT = 10
METADATA_FILENAME = "ucec_conch_patch_roi_metadata.json"


@dataclass(frozen=True)
class SelectedPatch:
    prompt_id: str
    prompt_text: str
    rank: int
    patch_index: int | None
    score: float | None
    coord: list[int] | None
    source_path: Path
    output_path: Path
    width: int
    height: int


@dataclass(frozen=True)
class UcecConchPatchRoiOutput:
    findings: dict[str, Any]
    warnings: tuple[str, ...]
    artifact_files: tuple[tuple[str, str, Path], ...]
    metadata: dict[str, Any]


def run_ucec_conch_patch_roi(
    case_id: str,
    output_dir: Path,
    *,
    slide_stem: str | None = None,
    prompt_ids: Iterable[str] | None = None,
    top_k_per_prompt: int = DEFAULT_TOP_K_PER_PROMPT,
    conch_roi_uri: str | None = None,
) -> UcecConchPatchRoiOutput:
    """Select and export UCEC pathology patch ROIs for one case."""

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    top_k = _validate_top_k(top_k_per_prompt)
    case_dir = _case_dir(case_id)
    case_yaml = _read_case_yaml(case_dir)
    slide_dir = resolve_conch_roi_slide_dir(
        case_id,
        case_yaml=case_yaml,
        slide_stem=slide_stem,
        conch_roi_uri=conch_roi_uri,
    )

    scores_path = slide_dir / "prompt_scores.json"
    packet_path = slide_dir / "roi_packet.json"
    scores = _load_prompt_scores(scores_path)
    prompt_entries = _prompt_entries(scores)
    selected_prompt_ids, prompt_warnings = _select_prompt_ids(
        prompt_entries,
        prompt_ids=prompt_ids,
    )

    selected: list[SelectedPatch] = []
    warnings = list(prompt_warnings)
    for prompt_id in selected_prompt_ids:
        prompt = prompt_entries[prompt_id]
        selected.extend(
            _export_prompt_rois(
                slide_dir=slide_dir,
                prompt=prompt,
                prompt_id=prompt_id,
                top_k=top_k,
                output_dir=output_dir,
                warnings=warnings,
            )
        )

    artifacts: list[tuple[str, str, Path]] = []
    contact_sheet = output_dir / "patch_contact_sheet.png"
    if selected:
        _write_contact_sheet(selected, contact_sheet)
        artifacts.append(("image", "patch_contact_sheet", contact_sheet))
        artifacts.extend(("image", "patch_roi", item.output_path) for item in selected)
    else:
        warnings.append("No patch ROI images were selected from the CONCH packet.")

    metadata = _metadata(
        case_id=case_id,
        case_dir=case_dir,
        case_yaml=case_yaml,
        slide_dir=slide_dir,
        scores=scores,
        scores_path=scores_path,
        packet_path=packet_path if packet_path.is_file() else None,
        selected=selected,
        selected_prompt_ids=selected_prompt_ids,
        top_k=top_k,
        warnings=warnings,
    )
    metadata_path = output_dir / METADATA_FILENAME
    write_json(metadata_path, metadata)
    artifacts.append(("json", "roi_metadata", metadata_path))

    findings = {
        "summary": (
            f"Selected {len(selected)} UCEC pathology patch ROI image(s) from "
            f"{metadata['slide_stem']} using cached CONCH v1.5 prompt scores."
        ),
        "roi_selected": bool(selected),
        "patches_selected": len(selected),
        "slide_stem": metadata["slide_stem"],
        "prompt_ids": selected_prompt_ids,
    }
    return UcecConchPatchRoiOutput(
        findings=findings,
        warnings=tuple(warnings),
        artifact_files=tuple(artifacts),
        metadata=metadata,
    )


def resolve_conch_roi_slide_dir(
    case_id: str,
    *,
    case_yaml: Mapping[str, Any] | None = None,
    slide_stem: str | None = None,
    conch_roi_uri: str | None = None,
) -> Path:
    """Resolve a slide-specific directory containing CONCH ROI outputs."""

    case_dir = _case_dir(case_id)
    if conch_roi_uri:
        roots = [_path_from_uri(conch_roi_uri)]
    else:
        case_yaml = case_yaml or _read_case_yaml(case_dir)
        roots = _candidate_conch_roi_roots(case_dir, case_yaml)
        if slide_stem is None:
            slide_stem = _pathology_string_from_case_yaml(case_yaml, "slide_stem")

    errors: list[str] = []
    for root in roots:
        try:
            return _resolve_slide_dir_from_root(root, slide_stem=slide_stem)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))
    searched = ", ".join(str(path.resolve()) for path in roots)
    raise FileNotFoundError(
        f"No slide CONCH ROI packet was found for case {case_id} under {searched}. "
        + "; ".join(errors)
    )


def _candidate_conch_roi_roots(
    case_dir: Path,
    case_yaml: Mapping[str, Any],
) -> list[Path]:
    roots: list[Path] = []
    configured = _pathology_path_from_case_yaml(case_yaml, "conch_roi_dir")
    if configured is not None:
        roots.append(configured)
    pathology_dir = case_dir / "pathology"
    roots.extend(
        [pathology_dir / "roi_256", pathology_dir / "roi_512", pathology_dir / "conch_roi"]
    )
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(root)
    return unique


def _resolve_slide_dir_from_root(root: Path, *, slide_stem: str | None) -> Path:
    root = root.resolve()
    if _is_slide_roi_dir(root):
        if slide_stem and root.name != slide_stem:
            raise FileNotFoundError(
                f"CONCH ROI override points to slide {root.name!r}, "
                f"not requested slide {slide_stem!r}."
            )
        return root
    if not root.is_dir():
        raise FileNotFoundError(f"CONCH ROI directory does not exist: {root}")
    if slide_stem:
        exact = root / slide_stem
        if _is_slide_roi_dir(exact):
            return exact.resolve()
        matches = [
            path.parent
            for path in root.rglob("prompt_scores.json")
            if slide_stem in path.parent.name
        ]
    else:
        matches = [path.parent for path in root.rglob("prompt_scores.json")]
    matches = sorted({path.resolve() for path in matches})
    if not matches:
        raise FileNotFoundError(
            f"No slide CONCH ROI packet with prompt_scores.json was found under {root}"
        )
    if len(matches) > 1 and not slide_stem:
        names = ", ".join(path.name for path in matches[:5])
        raise ValueError(
            "Multiple pathology slides have CONCH ROI packets; pass slide_stem. "
            f"Candidates: {names}"
        )
    return matches[0]


def _case_root() -> Path:
    override = os.environ.get("MEDCLAW_CASES_ROOT")
    if override:
        return _path_from_uri(override)
    return PROJECT_ROOT / "examples" / "cases"


def _case_dir(case_id: str) -> Path:
    path = (_case_root() / case_id).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Case directory does not exist: {path}")
    return path


def _read_case_yaml(case_dir: Path) -> Mapping[str, Any]:
    path = case_dir / "case.yaml"
    if not path.is_file():
        return {}
    data = read_yaml(path)
    if not isinstance(data, Mapping):
        raise ValueError(f"case.yaml must contain a mapping: {path}")
    return data


def _pathology_path_from_case_yaml(
    case_yaml: Mapping[str, Any],
    key: str,
) -> Path | None:
    data = case_yaml.get("data")
    if not isinstance(data, Mapping):
        return None
    pathology = data.get("pathology")
    if not isinstance(pathology, Mapping):
        return None
    value = pathology.get(key)
    if not isinstance(value, str) or not value:
        return None
    return _path_from_uri(value)


def _pathology_string_from_case_yaml(
    case_yaml: Mapping[str, Any],
    key: str,
) -> str | None:
    data = case_yaml.get("data")
    if not isinstance(data, Mapping):
        return None
    pathology = data.get("pathology")
    if not isinstance(pathology, Mapping):
        return None
    value = pathology.get(key)
    return value if isinstance(value, str) and value else None


def _path_from_uri(value: str) -> Path:
    if value.startswith("mock://"):
        raise ValueError(
            f"Mock URI cannot be used by UCEC pathology CONCH ROI skill: {value}"
        )
    parsed = urlparse(value)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _is_slide_roi_dir(path: Path) -> bool:
    return path.is_dir() and (path / "prompt_scores.json").is_file()


def _load_prompt_scores(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"CONCH prompt scores JSON does not exist: {path}")
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"CONCH prompt scores must be a JSON object: {path}")
    prompts = data.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"CONCH prompt scores must contain a non-empty prompts list: {path}")
    return data


def _prompt_entries(scores: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    entries: dict[str, Mapping[str, Any]] = {}
    for index, prompt in enumerate(scores.get("prompts", [])):
        if not isinstance(prompt, Mapping):
            raise ValueError(f"Prompt entry at index {index} must be an object")
        prompt_id = prompt.get("id")
        if not isinstance(prompt_id, str) or not prompt_id:
            prompt_id = f"prompt_{index:02d}"
        entries[prompt_id] = prompt
    return entries


def _select_prompt_ids(
    prompt_entries: Mapping[str, Mapping[str, Any]],
    *,
    prompt_ids: Iterable[str] | None,
) -> tuple[list[str], list[str]]:
    return select_prompt_ids(
        prompt_entries,
        prompt_ids=prompt_ids,
        default_prompt_ids=DEFAULT_PROMPT_IDS,
    )


def _validate_top_k(value: int) -> int:
    try:
        top_k = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("top_k_per_prompt must be an integer") from exc
    if top_k < 1 or top_k > MAX_TOP_K_PER_PROMPT:
        raise ValueError(
            f"top_k_per_prompt must be between 1 and {MAX_TOP_K_PER_PROMPT}"
        )
    return top_k


def _export_prompt_rois(
    *,
    slide_dir: Path,
    prompt: Mapping[str, Any],
    prompt_id: str,
    top_k: int,
    output_dir: Path,
    warnings: list[str],
) -> list[SelectedPatch]:
    top_indices = _as_sequence(prompt.get("top_indices"))
    top_scores = _as_sequence(prompt.get("top_scores"))
    top_coords = _as_sequence(prompt.get("top_coords"))
    prompt_text = str(prompt.get("text") or prompt_id)

    selected: list[SelectedPatch] = []
    for rank in range(min(top_k, len(top_indices))):
        patch_index = _as_int(top_indices[rank])
        source = _find_roi_image(slide_dir, prompt, prompt_id, rank, patch_index)
        if source is None:
            warnings.append(
                f"Patch ROI image missing for prompt={prompt_id} rank={rank} "
                f"index={patch_index}"
            )
            continue

        out_name = f"{prompt_id}_rank{rank:02d}_idx{patch_index or 0:05d}.png"
        output_path = output_dir / out_name
        width, height = _copy_png(source, output_path)
        selected.append(
            SelectedPatch(
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                rank=rank,
                patch_index=patch_index,
                score=_as_float(top_scores[rank]) if rank < len(top_scores) else None,
                coord=_as_coord(top_coords[rank]) if rank < len(top_coords) else None,
                source_path=source.resolve(),
                output_path=output_path.resolve(),
                width=width,
                height=height,
            )
        )
    return selected


def _as_sequence(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_coord(value: Any) -> list[int] | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    x = _as_int(value[0])
    y = _as_int(value[1])
    if x is None or y is None:
        return None
    return [x, y]


def _find_roi_image(
    slide_dir: Path,
    prompt: Mapping[str, Any],
    prompt_id: str,
    rank: int,
    patch_index: int | None,
) -> Path | None:
    roi_paths = prompt.get("roi_paths")
    if isinstance(roi_paths, list) and rank < len(roi_paths):
        path_value = roi_paths[rank]
        if isinstance(path_value, str) and path_value:
            candidate = _path_from_uri(path_value)
            if candidate.is_file():
                return candidate

    roi_dir = slide_dir / "roi" / prompt_id
    if not roi_dir.is_dir():
        return None
    if patch_index is not None:
        exact = roi_dir / f"{prompt_id}_roi_{rank:02d}_idx{patch_index:05d}.png"
        if exact.is_file():
            return exact
        matches = sorted(roi_dir.glob(f"*roi_{rank:02d}_idx{patch_index:05d}.png"))
        if matches:
            return matches[0]
    matches = sorted(roi_dir.glob(f"*roi_{rank:02d}_idx*.png"))
    return matches[0] if matches else None


def _copy_png(source: Path, destination: Path) -> tuple[int, int]:
    return copy_png_pixels(source, destination)


def _write_contact_sheet(selected: list[SelectedPatch], destination: Path) -> None:
    thumb_size = 256
    label_height = 42
    margin = 8
    cols = min(3, max(1, len(selected)))
    rows = (len(selected) + cols - 1) // cols
    cell_w = thumb_size + margin * 2
    cell_h = thumb_size + label_height + margin * 2
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(sheet)

    for index, patch in enumerate(selected):
        col = index % cols
        row = index // cols
        x0 = col * cell_w + margin
        y0 = row * cell_h + margin
        with Image.open(patch.output_path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((thumb_size, thumb_size))
            x = x0 + (thumb_size - thumb.width) // 2
            y = y0 + (thumb_size - thumb.height) // 2
            sheet.paste(thumb, (x, y))
        score = "" if patch.score is None else f" score={patch.score:.4f}"
        label = f"{patch.prompt_id} r{patch.rank} idx={patch.patch_index}{score}"
        draw.text((x0, y0 + thumb_size + 4), label[:52], fill="black")

    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def _metadata(
    *,
    case_id: str,
    case_dir: Path,
    case_yaml: Mapping[str, Any],
    slide_dir: Path,
    scores: Mapping[str, Any],
    scores_path: Path,
    packet_path: Path | None,
    selected: list[SelectedPatch],
    selected_prompt_ids: list[str],
    top_k: int,
    warnings: list[str],
) -> dict[str, Any]:
    wsi_path = _resolve_wsi_path(case_yaml, str(scores.get("stem") or slide_dir.name))
    return {
        "schema_version": "1.0",
        "created_at": utc_now(),
        "case_id": case_id,
        "case_dir": str(case_dir),
        "cancer_type": "UCEC",
        "slide_stem": str(scores.get("stem") or slide_dir.name),
        "method": "cached_conch_v1.5_prompt_roi",
        "conch_model": "CONCH v1.5",
        "n_patches": scores.get("n_patches"),
        "feature_dim": scores.get("feature_dim"),
        "patch_size": scores.get("patch_size"),
        "patch_level": scores.get("patch_level"),
        "selected_prompt_ids": selected_prompt_ids,
        "top_k_per_prompt": top_k,
        "source_prompt_scores": str(scores_path.resolve()),
        "source_roi_packet": str(packet_path.resolve()) if packet_path else None,
        "source_wsi": str(wsi_path.resolve()) if wsi_path else None,
        "selected_patches": [
            {
                "prompt_id": item.prompt_id,
                "prompt_text": item.prompt_text,
                "rank": item.rank,
                "patch_index": item.patch_index,
                "score": item.score,
                "coord": item.coord,
                "source_path": str(item.source_path),
                "output_file": item.output_path.name,
                "width": item.width,
                "height": item.height,
            }
            for item in selected
        ],
        "warnings": list(warnings),
    }


def _resolve_wsi_path(case_yaml: Mapping[str, Any], slide_stem: str) -> Path | None:
    wsi_dir = _pathology_path_from_case_yaml(case_yaml, "wsi_dir")
    if wsi_dir is None or not wsi_dir.is_dir():
        return None
    matches = sorted(wsi_dir.glob(f"{slide_stem}*"))
    return matches[0] if matches else None


def copy_default_prompts(destination: Path) -> None:
    """Copy the bundled UCEC default prompt file for setup helpers."""

    source = SKILL_DIR / "prompts" / "ucec_default.json"
    if not source.is_file():
        raise FileNotFoundError(f"Default prompt file is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def read_metadata(path: Path) -> dict[str, Any]:
    """Read a metadata artifact emitted by this skill."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Metadata must be a JSON object: {path}")
    return data
