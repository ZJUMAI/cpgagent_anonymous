#!/usr/bin/env python3
"""Build and audit the anonymous review artifact from an explicit allowlist."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "anonymous_release"
DEFAULT_ARCHIVE = REPO_ROOT / "data" / "anonymous_release.zip"
PUBLIC_LUNG_ROOT = REPO_ROOT / "data" / "cpgtrajbench" / "LUNG"
SOURCE_ROOTS = (
    "medclaw",
    "medclaw_benchmark",
    "guideline_planner",
    "scripts",
    "tests",
    "docs",
    "examples",
)
TOP_LEVEL_FILES = (".env.example", ".gitignore", "LICENSE", "pyproject.toml")
EXTRA_SOURCE_FILES = (
    "scripts/build_anonymous_release.py",
    "scripts/audit_anonymous_release.py",
    "tests/conftest.py",
    "tests/test_anonymous_reference_pack.py",
)
LIGHT_CASE_PATTERNS = (
    "hidden_state.json",
    "release_policy.json",
    "report_extraction.json",
    "clinical/*.json",
    "guideline/*.json",
    "molecular/*.json",
    "pathology/*.json",
    "pathology/*.txt",
    "radiology/*.json",
    "reports/**/*.md",
    "reports/**/*.txt",
)
SKIP_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__MACOSX",
    "artifacts",
    "runs",
    "outputs",
    "checkpoints",
    "roi_256",
    "wsi",
    "nifti",
    "masks",
}
SKIP_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".onnx",
    ".safetensors",
    ".svs",
    ".nii",
    ".gz",
    ".png",
    ".jpg",
    ".jpeg",
}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".sh", ".csv"}
SIGNED_URL_RE = re.compile(
    r"author" r"ization=|bce-" r"auth-v1|x-amz-(?:credential|signature)|"
    r"[?&](?:access[_-]?key|token|signature)=",
    re.I,
)
ABSOLUTE_PATH_RE = re.compile(
    r"(?:[A-Za-z]:\\" r"Users\\|/" r"Users/|/" r"home/)[^\s\"']+",
    re.I,
)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
IDENTITY_RE = re.compile(r"\bwu" r"ling\b|Yi" r"ng\s+Lab", re.I)
OBSOLETE_LUNG_GUIDELINE = "CSCO_NSCLC_" + "2025"
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", *SOURCE_ROOTS],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    paths = [Path(raw.decode("utf-8")) for raw in result.stdout.split(b"\0") if raw]
    for relative in EXTRA_SOURCE_FILES:
        path = Path(relative)
        if path.is_file() and path not in paths:
            paths.append(path)
    return sorted(set(paths), key=lambda item: item.as_posix())


def _allowed_source(path: Path) -> bool:
    if any(part in SKIP_PARTS or part.endswith(".egg-info") for part in path.parts):
        return False
    return path.suffix.lower() not in SKIP_SUFFIXES


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _copy_sources(stage: Path) -> None:
    for relative in _tracked_files():
        if _allowed_source(relative):
            _copy_file(REPO_ROOT / relative, stage / relative)
    for relative in TOP_LEVEL_FILES:
        _copy_file(REPO_ROOT / relative, stage / relative)


def _copy_light_case(source: Path, destination: Path) -> None:
    for pattern in LIGHT_CASE_PATTERNS:
        for path in sorted(source.glob(pattern)):
            if path.is_file() and _allowed_source(path.relative_to(source)):
                _copy_file(path, destination / path.relative_to(source))


def _normalize_rubric(rubric: dict[str, Any], case_id: str) -> dict[str, Any]:
    payload = dict(rubric)
    sources = dict(payload.get("source_files") or {})
    sources["patient_report"] = (
        f"data/LUNG/{case_id}/reports/integrated_reports/{case_id}_T0_full_report_en.md"
    )
    sources["patient_report_zh"] = (
        f"data/LUNG/{case_id}/reports/integrated_reports/{case_id}_T0_full_report_zh.md"
    )
    sources["guideline"] = "medclaw/knowledge/guidelines/NSCLC_2010.md"
    sources["matched_guideline"] = "NSCLC_2010"
    sources["guideline_version"] = "2010"
    sources["guideline_selection_mode"] = "fixed_historical_guideline"
    sources.pop("nccn_guideline_version", None)
    sources.pop("source_rubric_origin", None)
    payload["source_files"] = sources
    return payload


def _reference_facts(rubric: dict[str, Any]) -> tuple[str | None, tuple[str, str, str] | None]:
    text = " ".join(str(item) for item in rubric.get("case_summary") or [])
    sex_match = re.search(r"(男性|女性)|\b(male|female)\b", text, re.I)
    sex = None
    if sex_match:
        value = next(group for group in sex_match.groups() if group)
        sex = "female" if value.lower() in {"female", "女性"} else "male"
    tnm_match = re.search(
        r"\b(?:p)?(T\d[a-c]?)\s*[, /-]*\s*(?:p)?(N\d[a-c]?)\s*[, /-]*\s*(?:p)?(M\d[a-c]?)\b",
        text,
        re.I,
    )
    tnm = tuple(item.upper() for item in tnm_match.groups()) if tnm_match else None
    return sex, tnm  # type: ignore[return-value]


def _normalize_clinical_for_rubric(
    case_id: str, clinical: dict[str, Any], rubric: dict[str, Any]
) -> dict[str, Any]:
    payload = dict(clinical)
    rubric_sex, rubric_tnm = _reference_facts(rubric)
    clinical_sex = str(payload.get("sex") or "").lower()
    if rubric_sex and clinical_sex and rubric_sex != clinical_sex:
        raise ValueError(f"{case_id}: rubric/clinical sex mismatch")
    if rubric_tnm:
        keys = ("pathologic_t_stage", "pathologic_n_stage", "pathologic_m_stage")
        original = tuple(str(payload.get(key) or "").upper() for key in keys)
        if all(original) and original != rubric_tnm:
            for key, old_value, target_value in zip(keys, original, rubric_tnm, strict=True):
                payload[f"source_{key}"] = old_value
                payload[key] = target_value
            payload["staging_conversion_note"] = (
                "Guideline-target TNM normalized to the AJCC edition used by "
                "NCCN 2010; source-report TNM is retained in source_* fields."
            )
        elif not all(original):
            for key, target_value in zip(keys, rubric_tnm, strict=True):
                payload[key] = target_value
    return payload


def _assert_reference_consistency(
    case_id: str,
    rubric: dict[str, Any],
    clinical: dict[str, Any],
    trajectory: dict[str, Any],
) -> None:
    known = trajectory.get("canonical_state", {}).get("known_facts", {})
    rubric_sex, rubric_tnm = _reference_facts(rubric)
    expected_sex = str(clinical.get("sex") or "").lower()
    if rubric_sex and expected_sex and rubric_sex != expected_sex:
        raise ValueError(f"{case_id}: rubric/clinical sex mismatch")
    if expected_sex and known.get("sex") != expected_sex:
        raise ValueError(f"{case_id}: trajectory sex mismatch")
    clinical_tnm = tuple(
        str(clinical.get(key) or "").upper()
        for key in ("pathologic_t_stage", "pathologic_n_stage", "pathologic_m_stage")
    )
    if rubric_tnm and all(clinical_tnm) and rubric_tnm != clinical_tnm:
        raise ValueError(f"{case_id}: rubric/clinical TNM mismatch")
    trajectory_tnm = tuple(str(known.get(key) or "").upper() for key in ("T", "N", "M"))
    if all(clinical_tnm) and trajectory_tnm != clinical_tnm:
        raise ValueError(f"{case_id}: trajectory/clinical TNM mismatch")
    if trajectory.get("guideline_id") != "NSCLC_2010" or trajectory.get("guideline_version") != "2010":
        raise ValueError(f"{case_id}: wrong guideline target")
    serialized = json.dumps(trajectory, ensure_ascii=False)
    if OBSOLETE_LUNG_GUIDELINE in serialized:
        raise ValueError(f"{case_id}: obsolete Lung guideline label remains")


def _build_lung_cases(stage: Path) -> list[dict[str, Any]]:
    sys.path.insert(0, str(REPO_ROOT))
    from medclaw.trajectory import build_guideline_trajectory

    case_dirs = sorted(path for path in PUBLIC_LUNG_ROOT.iterdir() if path.is_dir())
    if len(case_dirs) != 30:
        raise RuntimeError(f"Expected 30 public Lung cases, found {len(case_dirs)}")
    manifest_cases: list[dict[str, Any]] = []
    for source in case_dirs:
        case_id = source.name
        destination = stage / "data" / "LUNG" / case_id
        _copy_light_case(source, destination)
        clinical_path = destination / "clinical" / "clinical.json"
        rubric_source = source / "evaluation" / f"{case_id}_rubric.json"
        report_path = destination / "reports" / "integrated_reports" / f"{case_id}_T0_full_report_en.md"
        if not report_path.is_file():
            report_path = destination / "reports" / "integrated_reports" / f"{case_id}_T0_full_report_zh.md"
        rubric = _normalize_rubric(
            json.loads(rubric_source.read_text(encoding="utf-8")), case_id
        )
        clinical = _normalize_clinical_for_rubric(
            case_id,
            json.loads(clinical_path.read_text(encoding="utf-8")),
            rubric,
        )
        _json_write(clinical_path, clinical)
        trajectory = build_guideline_trajectory(
            case_id,
            report_path,
            guideline_id="NSCLC_2010",
            guideline_version="2010",
            decision_date="2010-12-31",
            diagnosis_year=_diagnosis_year(clinical),
            clinical_path=clinical_path,
        )
        _assert_reference_consistency(case_id, rubric, clinical, trajectory)
        case_eval = destination / "evaluation"
        reference_dir = stage / "evaluation" / "references" / "LUNG" / case_id
        rubric_path = reference_dir / "rubric.json"
        trajectory_path = reference_dir / "trajectory.json"
        _json_write(rubric_path, rubric)
        _json_write(trajectory_path, trajectory)
        _json_write(case_eval / f"{case_id}_rubric.json", rubric)
        _json_write(case_eval / f"{case_id}_trajectory.json", trajectory)
        manifest_cases.append(
            {
                "case_id": case_id,
                "cancer_type": "LUNG",
                "rubric_path": f"LUNG/{case_id}/rubric.json",
                "trajectory_path": f"LUNG/{case_id}/trajectory.json",
                "rubric_sha256": _sha256(rubric_path),
                "trajectory_sha256": _sha256(trajectory_path),
            }
        )
    return manifest_cases


def _diagnosis_year(clinical: dict[str, Any]) -> int | None:
    value = clinical.get("diagnosis_year") or clinical.get("year_of_diagnosis")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _build_demo(stage: Path) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from medclaw.trajectory import build_guideline_trajectory

    source = REPO_ROOT / "examples" / "cases" / "TCGA-38-4626"
    destination = stage / "examples" / "cases" / "TCGA-38-4626"
    _copy_light_case(source, destination)
    for filename in ("case.yaml",):
        if (source / filename).is_file():
            _copy_file(source / filename, destination / filename)
    report = destination / "reports" / "integrated_reports" / "TCGA-38-4626_T0_full_report_zh.md"
    clinical = destination / "clinical" / "clinical.json"
    rubric_path = destination / "evaluation" / "TCGA-38-4626_rubric.json"
    rubric = _demo_nccn_2010_rubric(
        json.loads(
            (source / "evaluation" / "TCGA-38-4626_rubric.json").read_text(
                encoding="utf-8"
            )
        )
    )
    clinical_payload = _normalize_clinical_for_rubric(
        "TCGA-38-4626",
        json.loads(clinical.read_text(encoding="utf-8")),
        rubric,
    )
    _json_write(clinical, clinical_payload)
    _json_write(rubric_path, rubric)
    trajectory = build_guideline_trajectory(
        "TCGA-38-4626",
        report,
        clinical_path=clinical,
        diagnosis_year=2002,
    )
    _json_write(destination / "evaluation" / "TCGA-38-4626_trajectory.json", trajectory)


def _demo_nccn_2010_rubric(source: dict[str, Any]) -> dict[str, Any]:
    payload = dict(source)
    payload["rubric_version"] = "patient_nccn_2010_v1_zh"
    payload["source_files"] = {
        "patient_report": (
            "examples/cases/TCGA-38-4626/reports/integrated_reports/"
            "TCGA-38-4626_T0_full_report_zh.md"
        ),
        "guideline": "medclaw/knowledge/guidelines/NSCLC_2010.md",
        "matched_guideline": "NSCLC_2010",
        "guideline_version": "2010",
        "guideline_selection_mode": "fixed_historical_guideline",
    }
    payload["rubric"] = {
        "CI": [
            "回答应识别57岁女性、左肺上叶低分化腺癌、R0肺叶切除术后。",
            "回答应正确说明AJCC第7版病理分期为IIB期（pT3N0M0）。",
            "回答应区分已知分期与缺失的体能状态、合并症和分子检测。",
        ],
        "GM": [
            "回答应匹配NCCN NSCLC 2010中已完全切除的IIB期术后管理路径。",
            "回答应明确本评测固定使用NCCN 2010，而不是当前指南。",
        ],
        "RA": [
            "若体能状态和合并症允许，应推荐以顺铂为基础的辅助化疗。",
            "回答应推荐戒烟以及术后病史、体检和胸部影像随访。",
            "回答应先补充体能状态和合并症，再最终确定辅助治疗适宜性。",
        ],
        "CR": [
            "回答不得推荐NCCN 2010标准之外的辅助免疫治疗。",
            "回答不得在未知敏感突变时推荐靶向治疗。",
            "回答不得将本病例误写为其他TNM组合。",
        ],
        "EG": [
            "回答应将建议归因于NCCN NSCLC 2010的IIB期术后管理原则。",
            "回答不得编造发布指南文本中不存在的页码或证据等级。",
        ],
        "COMM": [
            "回答应分开陈述已知事实、缺失信息、指南匹配和条件性建议。",
            "回答应明确说明最终治疗决策取决于体能状态和合并症评估。",
        ],
    }
    payload["manual_review_flags"] = [
        "固定使用NCCN NSCLC 2010进行历史指南评测",
        "体能状态和合并症缺失",
        "分子检测缺失不应触发2010年之后的辅助治疗推荐",
    ]
    return payload


def _write_release_docs(stage: Path) -> None:
    (stage / "README.md").write_text(RELEASE_README, encoding="utf-8")
    (stage / "data").mkdir(parents=True, exist_ok=True)
    (stage / "data" / "README.md").write_text(DATA_README, encoding="utf-8")
    (stage / "evaluation").mkdir(parents=True, exist_ok=True)
    (stage / "evaluation" / "README.md").write_text(EVALUATION_README, encoding="utf-8")
    (stage / "docs").mkdir(parents=True, exist_ok=True)
    (stage / "docs" / "PLANNER_RELEASE.md").write_text(PLANNER_README, encoding="utf-8")


def _write_reference_manifest(stage: Path, cases: list[dict[str, Any]]) -> None:
    _json_write(
        stage / "evaluation" / "references" / "manifest.json",
        {
            "schema_version": "portable_judge_references.v2",
            "case_count": len(cases),
            "cancer_counts": {"LUNG": len(cases)},
            "guideline_target": {
                "guideline_id": "NSCLC_2010",
                "version": "2010",
                "decision_date": "2010-12-31",
                "evaluation_mode": "fixed_historical_guideline",
            },
            "generator": "scripts/build_anonymous_release.py",
            "revision": {
                "reason": "Correct sex extraction, case-specific TNM actions, and guideline-version mismatch.",
                "historical_scores_recomputed": False,
            },
            "cases": cases,
        },
    )


def _write_package_manifest(stage: Path) -> None:
    files = []
    for path in sorted(item for item in stage.rglob("*") if item.is_file()):
        relative = path.relative_to(stage).as_posix()
        if relative == "release_manifest.json":
            continue
        files.append({"path": relative, "size": path.stat().st_size, "sha256": _sha256(path)})
    _json_write(
        stage / "release_manifest.json",
        {
            "schema_version": "anonymous_release_manifest.v1",
            "file_count": len(files),
            "files": files,
        },
    )


def audit_release(root: Path, *, verify_manifest: bool = True) -> list[str]:
    root = root.resolve()
    errors: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_dir():
            if path.name in SKIP_PARTS or path.name.endswith(".egg-info"):
                errors.append(f"forbidden directory: {relative.as_posix()}")
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            errors.append(f"forbidden binary/artifact: {relative.as_posix()}")
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"LICENSE", ".gitignore", ".env.example"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-UTF-8 text file: {relative.as_posix()}")
            continue
        for label, pattern in (
            ("signed URL or credential parameter", SIGNED_URL_RE),
            ("absolute user-home path", ABSOLUTE_PATH_RE),
            ("author/lab identity", IDENTITY_RE),
        ):
            if pattern.search(text):
                errors.append(f"{label}: {relative.as_posix()}")
        for email in EMAIL_RE.findall(text):
            if not email.lower().endswith(("@example.com", "@example.org", "@example.invalid")):
                errors.append(f"email address: {relative.as_posix()}")
                break
        if OBSOLETE_LUNG_GUIDELINE in text:
            errors.append(f"obsolete Lung guideline label: {relative.as_posix()}")
        if path.suffix.lower() == ".json":
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                errors.append(f"invalid JSON {relative.as_posix()}: {exc}")
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                yaml.safe_load(text)
            except yaml.YAMLError as exc:
                errors.append(f"invalid YAML {relative.as_posix()}: {exc}")
        if path.suffix.lower() == ".md":
            errors.extend(_broken_links(path, root, text))
    if verify_manifest:
        errors.extend(_verify_package_manifest(root))
        errors.extend(_verify_reference_manifest(root))
    lung_root = root / "data" / "LUNG"
    if lung_root.is_dir() and len([p for p in lung_root.iterdir() if p.is_dir()]) != 30:
        errors.append("public Lung pack does not contain exactly 30 case directories")
    return sorted(set(errors))


def _broken_links(path: Path, root: Path, text: str) -> list[str]:
    errors = []
    for raw in MARKDOWN_LINK_RE.findall(text):
        target = raw.strip().split("#", 1)[0]
        if not target or target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        target = target.strip("<>")
        resolved = (path.parent / target).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            errors.append(f"link escapes release root: {path.relative_to(root).as_posix()} -> {raw}")
            continue
        if not resolved.exists():
            errors.append(f"broken local link: {path.relative_to(root).as_posix()} -> {raw}")
    return errors


def _verify_package_manifest(root: Path) -> list[str]:
    path = root / "release_manifest.json"
    if not path.is_file():
        return ["release_manifest.json is missing"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    expected = {item["path"]: item for item in manifest.get("files") or []}
    actual = {
        item.relative_to(root).as_posix(): item
        for item in root.rglob("*")
        if item.is_file() and item != path
    }
    if set(expected) != set(actual):
        errors.append("release manifest file list does not match directory contents")
    for relative in sorted(set(expected) & set(actual)):
        item = expected[relative]
        if item.get("size") != actual[relative].stat().st_size or item.get("sha256") != _sha256(actual[relative]):
            errors.append(f"release manifest hash mismatch: {relative}")
    return errors


def _verify_reference_manifest(root: Path) -> list[str]:
    path = root / "evaluation" / "references" / "manifest.json"
    if not path.is_file():
        return ["evaluation reference manifest is missing"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    if manifest.get("schema_version") != "portable_judge_references.v2":
        errors.append("evaluation reference manifest has the wrong schema version")
    for item in manifest.get("cases") or []:
        for field, hash_field in (("rubric_path", "rubric_sha256"), ("trajectory_path", "trajectory_sha256")):
            target = path.parent / item[field]
            if not target.is_file() or _sha256(target) != item[hash_field]:
                errors.append(f"reference manifest mismatch: {item.get('case_id')} {field}")
    return errors


def _zip_release(root: Path, archive: Path) -> None:
    temp = archive.with_suffix(archive.suffix + ".tmp")
    if temp.exists():
        temp.unlink()
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = Path(root.name) / path.relative_to(root)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            handle.writestr(info, path.read_bytes())
    if archive.exists():
        archive.chmod(archive.stat().st_mode | stat.S_IWRITE)
        archive.unlink()
    os.replace(temp, archive)


def _audit_archive(archive: Path) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="anonymous-release-audit-") as raw:
        target = Path(raw)
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(target)
        roots = [path for path in target.iterdir() if path.is_dir()]
        if len(roots) != 1:
            return ["archive must contain exactly one top-level directory"]
        return audit_release(roots[0])


def build_release(output: Path, archive: Path) -> None:
    output = output.resolve()
    archive = archive.resolve()
    expected_parent = (REPO_ROOT / "data").resolve()
    if output.parent != expected_parent or output.name != "anonymous_release":
        raise ValueError("Refusing to replace an output other than data/anonymous_release")
    stage = Path(tempfile.mkdtemp(prefix="anonymous-release-build-", dir=expected_parent))
    try:
        _copy_sources(stage)
        _write_release_docs(stage)
        cases = _build_lung_cases(stage)
        _build_demo(stage)
        _write_reference_manifest(stage, cases)
        _write_package_manifest(stage)
        errors = audit_release(stage)
        if errors:
            raise RuntimeError("Release audit failed:\n- " + "\n- ".join(errors))
        if output.exists():
            shutil.rmtree(output)
        os.replace(stage, output)
        _zip_release(output, archive)
        archive_errors = _audit_archive(archive)
        if archive_errors:
            raise RuntimeError("Archive audit failed:\n- " + "\n- ".join(archive_errors))
    finally:
        if stage.exists():
            shutil.rmtree(stage)


RELEASE_README = """# CPGTrajBench anonymous review artifact

This artifact contains the inspectable runtime, evaluation code, one demo case,
and lightweight text/JSON packs for 30 public TCGA Lung cases. It is research
software, not a clinical product or medical advice.

## Reproduction tiers

1. **CPU verification:** install `.[dev]` and run `python -m pytest`.
2. **Public Lung verification:** inspect or score the 30 cases in `data/LUNG` and
   the corrected references in `evaluation/references`.
3. **Full 109-case experiments:** not independently reproducible from this
   artifact. Planner weights and the institutional UCEC/NPC case packs are not
   distributed.

The corrected Lung references target `NSCLC_2010@2010` with decision date
`2010-12-31`. Historical Dynamic/Combined paper scores were not recomputed and
must not be presented as exact results from these corrected references.

## Quickstart

```bash
python -m venv .venv
# bash: source .venv/bin/activate
# PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m pytest
```

Configuration is read from process environment variables. Copying
`.env.example` does not load it automatically. For bash use
`set -a; source .env; set +a`; for PowerShell set the listed `$env:NAME`
variables explicitly.

See `evaluation/README.md`, `data/README.md`, and
`docs/PLANNER_RELEASE.md` for the release boundaries.

## Licensing note

The third-party guideline text requires a separate redistribution-rights
review. Its presence in this review artifact does not imply that the project
MIT license grants rights to the guideline content.
"""

DATA_README = """# Public data in this artifact

`data/LUNG` contains 30 lightweight public TCGA Lung case packs: clinical,
molecular, report, metadata, rubric, and corrected trajectory JSON/text files.
Raw CT, WSI, ROI images, Planner weights, and institutional UCEC/NPC cases are
not included.
"""

EVALUATION_README = """# Corrected Lung evaluation references

The 30 references use `NSCLC_2010@2010` and decision date `2010-12-31`.
`references/manifest.json` binds every rubric and trajectory by SHA-256.

These references correct sex extraction, case-specific TNM actions, and a
guideline-version mismatch. Historical paper Dynamic/Combined scores were not
recomputed, so they are not exactly reproducible with this corrected set.
"""

PLANNER_README = """# Planner release boundary

Planner training/inference code is included for inspection, but the trained
Planner release and base-model snapshots are not distributed and no anonymous
download URL is provided. Unit tests and deterministic evaluation utilities do
not require these weights; Dual-Agent inference does.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        errors = audit_release(args.output)
        if args.archive.is_file():
            errors.extend(_audit_archive(args.archive))
        if errors:
            print("\n".join(f"ERROR: {item}" for item in sorted(set(errors))))
            return 1
        print(f"Anonymous release audit passed: {args.output}")
        return 0
    build_release(args.output, args.archive)
    print(f"Built and audited {args.output}")
    print(f"Built and audited {args.archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
