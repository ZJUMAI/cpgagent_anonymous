"""Guideline Markdown chunking by first-level clinical topic headings."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from guideline_planner.constants import DEFAULT_GUIDELINE_DIR
from guideline_planner.io_utils import safe_filename, stable_short_id, write_jsonl


@dataclass(frozen=True)
class GuidelineChunk:
    guideline_id: str
    version: str
    cancer_type: str
    chapter: str
    section: str
    h1_title: str
    source_span_id: str
    source_rule_ids: list[str]
    source_span_ids: list[str]
    page_start: int | None
    page_end: int | None
    text: str
    source_path: str
    language: str = "zh"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_PAGE_RE = re.compile(r"^##\s*第\s*(\d+)\s*页\s*$")
_H1_RE = re.compile(r"^#(?!#)\s+(.+?)\s*$")
_VERSION_RE = re.compile(r"(20\d{2})")
_RULE_RE = re.compile(
    r"(?:[ⅠⅡⅢIVX]+|[一二三四五六七八九十]+|I{1,3})\s*级推荐"
    r"|推荐|证据|适应证|治疗|诊断|检测|随访"
    r"|recommend(?:ation|ed)?|evidence|indicat(?:ion|ed)"
    r"|treatment|therapy|diagnos(?:is|tic)|testing|surveillance"
    r"|evaluation|staging",
    flags=re.IGNORECASE,
)


def chunk_guidelines(
    guideline_dir: str | Path = DEFAULT_GUIDELINE_DIR,
    *,
    output_path: str | Path | None = None,
) -> list[GuidelineChunk]:
    """Read guideline Markdown files and split them only on ``#`` headings."""

    directory = Path(guideline_dir)
    chunks: list[GuidelineChunk] = []
    for path in sorted(directory.rglob("*.md")):
        if ".embedding_cache" in path.parts:
            continue
        chunks.extend(chunk_guideline_file(path))
    if output_path is not None:
        write_jsonl(Path(output_path), (chunk.to_dict() for chunk in chunks))
    return chunks


def chunk_guideline_file(path: Path) -> list[GuidelineChunk]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    guideline_id = infer_guideline_id(path)
    version = infer_version(path, text)
    cancer_type = infer_cancer_type(path, text)

    starts: list[tuple[int, str, int | None]] = []
    current_page: int | None = None
    page_by_line: dict[int, int] = {}
    for index, line in enumerate(lines):
        page_match = _PAGE_RE.match(line.strip())
        if page_match:
            current_page = int(page_match.group(1))
        if current_page is not None:
            page_by_line[index] = current_page
        h1_match = _H1_RE.match(line)
        if h1_match:
            starts.append((index, h1_match.group(1).strip(), current_page))

    chunks: list[GuidelineChunk] = []
    for ordinal, (start, title, start_page) in enumerate(starts):
        end = starts[ordinal + 1][0] if ordinal + 1 < len(starts) else len(lines)
        body_lines = lines[start:end]
        body_text = "\n".join(body_lines).strip()
        pages = [page for line_no, page in page_by_line.items() if start <= line_no < end]
        if start_page is not None:
            pages.append(start_page)
        source_span_id = f"{guideline_id}::h1::{ordinal + 1:04d}"
        source_rule_ids = _extract_rule_ids(body_lines, source_span_id)
        chunks.append(
            GuidelineChunk(
                guideline_id=guideline_id,
                version=version,
                cancer_type=cancer_type,
                chapter=title,
                section=title,
                h1_title=title,
                source_span_id=source_span_id,
                source_rule_ids=source_rule_ids,
                source_span_ids=[source_span_id],
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                text=body_text,
                source_path=str(path),
                language=_infer_language(body_text),
            )
        )
    return chunks


def infer_guideline_id(path: Path) -> str:
    return safe_filename(path.stem, max_length=120)


def infer_version(path: Path, text: str) -> str:
    match = _VERSION_RE.search(path.name) or _VERSION_RE.search(text[:2000])
    return match.group(1) if match else "unknown"


def infer_cancer_type(path: Path, text: str = "") -> str:
    haystack = f"{path.name} {text[:1000]}".lower()
    rules = (
        ("nsclc", ("nsclc", "non-small", "非小细胞")),
        ("sclc", ("sclc", "small cell", "小细胞肺癌")),
        ("ucec", ("ucec", "endometrial", "子宫内膜")),
        ("nasopharyngeal", ("nasopharyngeal", "鼻咽")),
        ("breast", ("breast", "乳腺")),
        ("colorectal", ("colorectal", "结直肠", "colon", "rectal")),
        ("head_neck", ("head_and_neck", "head neck", "头颈")),
        ("lymphoma", ("lymphoma", "淋巴瘤")),
    )
    for cancer_type, needles in rules:
        if any(needle in haystack for needle in needles):
            return cancer_type
    return "unknown"


def _extract_rule_ids(lines: Iterable[str], source_span_id: str) -> list[str]:
    rule_ids: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and _RULE_RE.search(stripped):
            rule_ids.append(f"{source_span_id}:rule:{len(rule_ids) + 1:03d}")
    return rule_ids


def _infer_language(text: str) -> str:
    return "zh" if any("\u4e00" <= char <= "\u9fff" for char in text) else "en"
