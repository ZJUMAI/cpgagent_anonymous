from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from guideline_planner.chunking import chunk_guideline_file as chunk_guideline_file_for_planner
from medclaw.llm.providers.local_openai import (
    DEFAULT_LOCAL_BASE_URL,
    DEFAULT_LOCAL_MODEL,
)


SKILL_NAME = "guideline.retrieve"
SKILL_VERSION = "0.1.0"
DEFAULT_QUERY = "肿瘤 初始诊断 分期 病理 分子检测 生物标志物 治疗推荐 随访"
DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_EMBEDDING_DIMENSIONS = 1024
DEFAULT_EMBEDDING_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_RERANK_MODEL = "qwen3.7-plus"
DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_SEGMENT_MAX_CHARS = 6000
EMBEDDING_AGGREGATION = "segment_mean_v1"
MAX_EXCERPT_CHARS = 900
CHUNK_MAX_CHARS = 3200
CHUNK_OVERLAP_CHARS = 300
MAX_RERANK_CANDIDATES = 24


class EmbeddingCacheError(RuntimeError):
    """Raised when vector retrieval cannot be prepared."""


class RerankError(RuntimeError):
    """Raised when the core LLM cannot rerank retrieved snippets."""


@dataclass(frozen=True)
class GuidelineDocument:
    source_id: str
    cancer_type: str
    title: str
    path: Path
    text: str


@dataclass(frozen=True)
class GuidelineChunk:
    chunk_id: str
    source_id: str
    cancer_type: str
    title: str
    page: int | None
    heading: str
    text: str
    page_end: int | None = None
    chunk_mode: str = "chapter"
    guideline_memory_id: str | None = None
    planner_guideline_id: str | None = None
    source_rule_ids: tuple[str, ...] = ()
    source_span_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EmbeddingConfig:
    api_key: str | None
    base_url: str
    model: str
    dimensions: int
    batch_size: int


@dataclass(frozen=True)
class RerankConfig:
    provider: str
    api_key: str | None
    base_url: str
    model: str
    timeout_sec: float
    enable_thinking: bool


@dataclass(frozen=True)
class VectorIndex:
    chunks: list[GuidelineChunk]
    embeddings: Any
    faiss_index: Any | None
    backend: str
    cache_status: str
    cache_dir: Path
    corpus_hash: str
    manifest: dict[str, Any]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    arguments = payload.get("arguments", {})
    result = run(arguments, output_dir=output_path.parent)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


def run(arguments: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    query = str(arguments.get("query") or DEFAULT_QUERY).strip()
    cancer_type = str(arguments.get("cancer_type") or "auto").strip().lower()
    guideline_family = str(arguments.get("guideline_family") or "auto").strip().lower()
    guideline_version = str(arguments.get("guideline_version") or "auto").strip().lower()
    retrieval_mode = str(arguments.get("retrieval_mode") or "auto").strip().lower()
    if retrieval_mode not in {"auto", "vector", "lexical"}:
        retrieval_mode = "auto"
    rerank_mode = str(arguments.get("rerank_mode") or "auto").strip().lower()
    if rerank_mode not in {"auto", "llm", "none"}:
        rerank_mode = "auto"
    chunk_mode = str(arguments.get("chunk_mode") or "chapter").strip().lower()
    if chunk_mode not in {"page", "chapter"}:
        chunk_mode = "chapter"
    force_rebuild = bool(arguments.get("force_rebuild_embeddings", False))
    max_snippets = _bounded_int(arguments.get("max_snippets"), default=5, lower=1, upper=8)
    rerank_candidate_count = _bounded_int(
        arguments.get("rerank_candidate_count"),
        default=max(12, max_snippets * 4),
        lower=max_snippets,
        upper=MAX_RERANK_CANDIDATES,
    )

    try:
        guideline_dir = _guideline_dir()
        documents = _load_documents(guideline_dir)
        selected_type = _select_cancer_type(cancer_type, query)
        selected_family = _select_guideline_family(guideline_family, query)
        selected_version = _select_guideline_version(guideline_version, query)
        selected_documents = _filter_documents(
            documents,
            selected_type,
            guideline_family=selected_family,
            guideline_version=selected_version,
        )
        selected_source_ids = {document.source_id for document in selected_documents}
        chunks = [
            chunk
            for document in documents
            for chunk in _split_document_into_chunks(document, chunk_mode=chunk_mode)
        ]
        selected_chunks = [
            chunk
            for chunk in chunks
            if chunk.source_id in selected_source_ids
        ]

        embedding_config = _embedding_config_from_env()
        vector_index: VectorIndex | None = None
        query_cache_status = "not_used"
        retrieval_method = "lexical"
        rerank_config = _rerank_config_from_env()
        rerank_status = "not_requested"
        rerank_model: str | None = None
        pre_rerank_snippets: list[dict[str, Any]]
        snippets: list[dict[str, Any]]

        if retrieval_mode in {"auto", "vector"}:
            try:
                vector_index = _load_or_build_vector_index(
                    chunks,
                    documents,
                    guideline_dir=guideline_dir,
                    config=embedding_config,
                    force_rebuild=force_rebuild,
                )
                query_embedding, query_cache_status = _embed_query_with_cache(
                    query,
                    vector_index.cache_dir,
                    embedding_config,
                )
                snippets = _vector_search(
                    vector_index,
                    query,
                    query_embedding=query_embedding,
                    selected_type=selected_type,
                    selected_source_ids=selected_source_ids,
                    max_snippets=rerank_candidate_count,
                )
                retrieval_method = vector_index.backend
            except EmbeddingCacheError as exc:
                if retrieval_mode == "vector":
                    raise
                warnings.append(f"Vector retrieval unavailable; used lexical fallback. Reason: {exc}")
                snippets = _lexical_search(
                    selected_chunks,
                    query,
                    max_snippets=rerank_candidate_count,
                )
                retrieval_method = "lexical_fallback"
        else:
            snippets = _lexical_search(
                selected_chunks,
                query,
                max_snippets=rerank_candidate_count,
            )

        pre_rerank_snippets = snippets
        snippets, rerank_status, rerank_model, rerank_warning = _maybe_llm_rerank(
            query,
            pre_rerank_snippets,
            max_snippets=max_snippets,
            rerank_mode=rerank_mode,
            config=rerank_config,
        )
        if rerank_warning:
            warnings.append(rerank_warning)

        if not snippets:
            warnings.append("No guideline snippet matched the query; returned no snippets.")

        artifact_payload = {
            "query": query,
            "requested_cancer_type": cancer_type,
            "selected_cancer_type": selected_type,
            "requested_guideline_family": guideline_family,
            "selected_guideline_family": selected_family,
            "requested_guideline_version": guideline_version,
            "selected_guideline_version": selected_version,
            "max_snippets": max_snippets,
            "rerank_candidate_count": rerank_candidate_count,
            "chunk_mode": chunk_mode,
            "retrieval_method": retrieval_method,
            "rerank_mode": rerank_mode,
            "rerank_status": rerank_status,
            "rerank_model": rerank_model,
            "embedding_model": embedding_config.model,
            "embedding_dimensions": embedding_config.dimensions,
            "embedding_cache_status": vector_index.cache_status if vector_index else "not_used",
            "query_embedding_cache_status": query_cache_status,
            "embedding_cache_dir": str(vector_index.cache_dir) if vector_index else None,
            "snippets": snippets,
            "pre_rerank_snippets": pre_rerank_snippets,
            "source_documents": [
                {
                    "source_id": document.source_id,
                    "cancer_type": document.cancer_type,
                    "title": document.title,
                    "path": str(document.path),
                    "sha256": _sha256_file(document.path),
                }
                for document in selected_documents
            ],
        }
        artifact_path = output_dir / "guideline_retrieval.json"
        artifact_path.write_text(
            json.dumps(artifact_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "status": "success",
            "findings": {
                "summary": (
                    f"Retrieved {len(snippets)} guideline snippet(s) "
                    f"with {retrieval_method}; chunk_mode={chunk_mode}; "
                    f"rerank_status={rerank_status}; "
                    f"query: {query}"
                ),
                "retrieved": bool(snippets),
                "query": query,
                "requested_cancer_type": cancer_type,
                "selected_cancer_type": selected_type,
                "requested_guideline_family": guideline_family,
                "selected_guideline_family": selected_family,
                "requested_guideline_version": guideline_version,
                "selected_guideline_version": selected_version,
                "retrieval_method": retrieval_method,
                "rerank_mode": rerank_mode,
                "rerank_status": rerank_status,
                "rerank_model": rerank_model,
                "chunk_mode": chunk_mode,
                "embedding_model": embedding_config.model,
                "embedding_dimensions": embedding_config.dimensions,
                "embedding_cache_status": vector_index.cache_status if vector_index else "not_used",
                "query_embedding_cache_status": query_cache_status,
                "embedding_cache_dir": str(vector_index.cache_dir) if vector_index else None,
                "snippets": snippets,
                "source_documents": [
                    {
                        "source_id": document.source_id,
                        "cancer_type": document.cancer_type,
                        "title": document.title,
                    }
                    for document in selected_documents
                ],
            },
            "artifacts": [
                {
                    "type": "json",
                    "role": "guideline_retrieval_metadata",
                    "path": artifact_path.name,
                }
            ],
            "provenance": {},
            "warnings": warnings,
        }
    except Exception as exc:
        return {
            "status": "failed",
            "findings": {
                "summary": f"Guideline retrieval failed: {exc}",
                "retrieved": False,
                "error": str(exc),
            },
            "artifacts": [],
            "provenance": {},
            "warnings": [str(exc)],
        }


def _guideline_dir() -> Path:
    medclaw_root = Path(__file__).resolve().parents[3]
    guideline_dir = medclaw_root / "knowledge" / "guidelines"
    if not guideline_dir.is_dir():
        raise FileNotFoundError(f"Guideline directory does not exist: {guideline_dir}")
    return guideline_dir


def _load_documents(guideline_dir: Path) -> list[GuidelineDocument]:
    documents: list[GuidelineDocument] = []
    for path in sorted(guideline_dir.rglob("*.md")):
        if ".embedding_cache" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        source_id, cancer_type, title = _document_metadata(path, guideline_dir, text)
        documents.append(
            GuidelineDocument(
                source_id=source_id,
                cancer_type=cancer_type,
                title=title,
                path=path,
                text=text,
            )
        )
    if not documents:
        raise FileNotFoundError(f"No guideline Markdown files found under: {guideline_dir}")
    return documents


def _select_cancer_type(requested: str, query: str) -> str:
    if requested and requested not in {"auto", "all"}:
        return requested
    if requested == "all":
        return "all"
    query_lower = query.lower()
    if (
        "nsclc" in query_lower
        or "非小细胞" in query
        or "tcga-luad" in query_lower
        or "tcga-lusc" in query_lower
        or "luad" in query_lower
        or "lusc" in query_lower
        or "lung adenocarcinoma" in query_lower
        or "lung squamous" in query_lower
        or "肺腺癌" in query
    ):
        return "nsclc"
    if "sclc" in query_lower or ("小细胞" in query and "非小细胞" not in query):
        return "sclc"
    if (
        "ucec" in query_lower
        or "endometr" in query_lower
        or "子宫内膜" in query
    ):
        return "ucec"
    if (
        "nasopharyn" in query_lower
        or "npc" in query_lower
        or "鼻咽" in query
    ):
        return "npc"
    for project, cancer_type in {
        "tcga-ucec": "ucec",
        "tcga-brca": "brca",
        "tcga-coad": "coad",
        "tcga-read": "read",
        "tcga-stad": "stad",
        "tcga-lihc": "lihc",
        "tcga-kirc": "kirc",
        "tcga-prad": "prad",
    }.items():
        if project in query_lower:
            return cancer_type
    return "all"


def _filter_documents(
    documents: Iterable[GuidelineDocument],
    selected_type: str,
    *,
    guideline_family: str = "all",
    guideline_version: str = "all",
) -> list[GuidelineDocument]:
    selected: list[GuidelineDocument] = []
    for document in documents:
        if selected_type != "all" and document.cancer_type != selected_type:
            continue
        source_id = document.source_id.lower()
        if guideline_family != "all" and not source_id.startswith(f"{guideline_family}_"):
            continue
        if guideline_version != "all" and not source_id.endswith(f"_{guideline_version}"):
            continue
        selected.append(document)
    return selected


def _select_guideline_family(requested: str, query: str) -> str:
    normalized = requested.strip().lower()
    if normalized not in {"", "auto", "all"}:
        return normalized
    query_lower = query.lower()
    if "nccn" in query_lower:
        return "nccn"
    if "csco" in query_lower:
        return "csco"
    return "all"


def _select_guideline_version(requested: str, query: str) -> str:
    normalized = requested.strip().lower()
    if normalized not in {"", "auto", "all"}:
        return normalized
    match = re.search(
        r"(?:nccn|csco)\D{0,12}(20\d{2})|(20\d{2})\D{0,12}(?:nccn|csco)",
        query,
        flags=re.IGNORECASE,
    )
    if match:
        return next(group for group in match.groups() if group)
    return "all"


def _document_metadata(
    path: Path,
    guideline_dir: Path,
    text: str,
) -> tuple[str, str, str]:
    filename = path.name
    known = {
        "NSCLC_2010.md": (
            "nccn_nsclc_2010",
            "nsclc",
            "NCCN NSCLC 2010 Discussion Chunks",
        ),
        "SCLC_2010.md": (
            "nccn_sclc_2010",
            "sclc",
            "NCCN SCLC 2010 Discussion Chunks",
        ),
        "CSCO子宫内膜癌2023.md": (
            "csco_ucec_2023",
            "ucec",
            "CSCO 子宫内膜癌诊疗指南 2023",
        ),
        "CSCO鼻咽癌2022.md": (
            "csco_npc_2022",
            "npc",
            "CSCO 鼻咽癌诊疗指南 2022",
        ),
        "2025CSCO非小细胞肺癌诊疗指南.md": (
            "csco_nsclc_2025",
            "nsclc",
            "2025 CSCO 非小细胞肺癌诊疗指南",
        ),
        "2025CSCO小细胞肺癌诊疗指南.md": (
            "csco_sclc_2025",
            "sclc",
            "2025 CSCO 小细胞肺癌诊疗指南",
        ),
        "2025_CSCO_NSCLC_Guideline.md": (
            "csco_nsclc_2025",
            "nsclc",
            "2025 CSCO 非小细胞肺癌诊疗指南",
        ),
        "2025CSCO结直肠癌诊疗指南.md": (
            "csco_colorectal_2025",
            "colorectal",
            "2025 CSCO 结直肠癌诊疗指南",
        ),
        "2025_CSCO_Breast_Cancer_Guideline.md": (
            "csco_breast_2025",
            "breast",
            "2025 CSCO 乳腺癌诊疗指南",
        ),
        "2025_CSCO_Head_and_Neck_Cancer_Guideline.md": (
            "csco_head_neck_2025",
            "head_neck",
            "2025 CSCO 头颈部肿瘤诊疗指南",
        ),
        "2025_CSCO_Lymphoma_Guideline.md": (
            "csco_lymphoma_2025",
            "lymphoma",
            "2025 CSCO 淋巴瘤诊疗指南",
        ),
    }
    if filename in known:
        return known[filename]
    nccn_match = re.fullmatch(
        r"NCCN[ _-]+(NSCLC|SCLC)[ _-]+(20\d{2})\.md",
        filename,
        flags=re.IGNORECASE,
    )
    if nccn_match:
        cancer_type = nccn_match.group(1).lower()
        version = nccn_match.group(2)
        return (
            f"nccn_{cancer_type}_{version}",
            cancer_type,
            f"NCCN {cancer_type.upper()} {version}",
        )

    relative = path.relative_to(guideline_dir)
    source_id = _safe_source_id(relative.with_suffix("").as_posix())
    title = _first_heading(text) or path.stem
    return source_id, _infer_document_cancer_type(path, guideline_dir, text), title


def _safe_source_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-").lower()
    return normalized or "guideline"


def _infer_document_cancer_type(path: Path, guideline_dir: Path, text: str) -> str:
    haystack = f"{path.as_posix()} {text[:2000]}".lower()
    if "非小细胞" in haystack or "nsclc" in haystack:
        return "nsclc"
    if "小细胞" in haystack or "sclc" in haystack:
        return "sclc"
    if "子宫内膜" in haystack or "endometr" in haystack or "ucec" in haystack:
        return "ucec"
    if "鼻咽" in haystack or "nasopharyn" in haystack:
        return "npc"
    if "乳腺" in haystack or "breast" in haystack:
        return "breast"
    if (
        "结直肠" in haystack
        or "colorectal" in haystack
        or "colon" in haystack
        or "rectal" in haystack
    ):
        return "colorectal"
    if "头颈" in haystack or "head_and_neck" in haystack or "head and neck" in haystack:
        return "head_neck"
    if "淋巴瘤" in haystack or "lymphoma" in haystack:
        return "lymphoma"
    try:
        relative = path.relative_to(guideline_dir)
    except ValueError:
        relative = path
    if len(relative.parts) > 1:
        parent = relative.parts[0].strip().lower()
        if parent and not parent.startswith("."):
            return _safe_source_id(parent)
    return "general"


def _split_document_into_chunks(
    document: GuidelineDocument,
    *,
    chunk_mode: str,
) -> list[GuidelineChunk]:
    if chunk_mode == "chapter":
        return _split_into_planner_chapter_chunks(document)
    return _split_into_page_chunks(document)


def _split_into_page_chunks(document: GuidelineDocument) -> list[GuidelineChunk]:
    page_records: list[tuple[int | None, str]] = []
    current_page: int | None = None
    current_lines: list[str] = []
    page_pattern = re.compile(r"^##\s*第\s*(\d+)\s*页")

    def flush() -> None:
        if not current_lines:
            return
        text = "\n".join(current_lines).strip()
        if text:
            page_records.append((current_page, text))

    for line in document.text.splitlines():
        match = page_pattern.match(line.strip())
        if match:
            flush()
            current_page = int(match.group(1))
            current_lines = [line]
            continue
        current_lines.append(line)
    flush()

    chunks: list[GuidelineChunk] = []
    for page, page_text in page_records:
        heading = _first_heading(page_text)
        for index, text in enumerate(_split_long_text(page_text)):
            page_label = page if page is not None else 0
            chunk_id = f"{document.source_id}:p{page_label}:c{index + 1}"
            chunks.append(
                GuidelineChunk(
                    chunk_id=chunk_id,
                    source_id=document.source_id,
                    cancer_type=document.cancer_type,
                    title=document.title,
                    page=page,
                    heading=heading,
                    text=text,
                    page_end=page,
                    chunk_mode="page",
                )
            )
    return chunks


def _split_into_planner_chapter_chunks(document: GuidelineDocument) -> list[GuidelineChunk]:
    chunks: list[GuidelineChunk] = []
    for planner_chunk in chunk_guideline_file_for_planner(document.path):
        text = str(planner_chunk.text).strip()
        if not text:
            continue
        chunks.append(
            GuidelineChunk(
                chunk_id=planner_chunk.source_span_id,
                source_id=document.source_id,
                cancer_type=document.cancer_type,
                title=document.title,
                page=planner_chunk.page_start,
                heading=planner_chunk.h1_title,
                text=text,
                page_end=planner_chunk.page_end,
                chunk_mode="chapter",
                guideline_memory_id=planner_chunk.source_span_id,
                planner_guideline_id=planner_chunk.guideline_id,
                source_rule_ids=tuple(planner_chunk.source_rule_ids),
                source_span_ids=tuple(planner_chunk.source_span_ids),
            )
        )
    return chunks


def _split_long_text(text: str) -> list[str]:
    compact = text.strip()
    if len(compact) <= CHUNK_MAX_CHARS:
        return [compact]
    chunks: list[str] = []
    start = 0
    while start < len(compact):
        end = min(len(compact), start + CHUNK_MAX_CHARS)
        chunks.append(compact[start:end].strip())
        if end >= len(compact):
            break
        start = max(0, end - CHUNK_OVERLAP_CHARS)
    return [chunk for chunk in chunks if chunk]


def _first_heading(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def _embedding_config_from_env() -> EmbeddingConfig:
    return EmbeddingConfig(
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        base_url=os.environ.get("MEDCLAW_EMBEDDING_BASE_URL", DEFAULT_EMBEDDING_BASE_URL).rstrip("/"),
        model=os.environ.get("MEDCLAW_GUIDELINE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        dimensions=_bounded_int(
            os.environ.get("MEDCLAW_GUIDELINE_EMBEDDING_DIMENSIONS"),
            default=DEFAULT_EMBEDDING_DIMENSIONS,
            lower=64,
            upper=4096,
        ),
        batch_size=_bounded_int(
            os.environ.get("MEDCLAW_GUIDELINE_EMBEDDING_BATCH_SIZE"),
            default=10,
            lower=1,
            upper=10,
        ),
    )


def _rerank_config_from_env() -> RerankConfig:
    explicit_provider = os.environ.get("MEDCLAW_GUIDELINE_RERANK_PROVIDER")
    provider = (
        explicit_provider
        or os.environ.get("MEDCLAW_LLM_PROVIDER")
        or "local_openai"
    ).strip().lower()
    if provider not in {"local_openai", "qwen"}:
        if explicit_provider:
            raise RerankError(
                "MEDCLAW_GUIDELINE_RERANK_PROVIDER must be local_openai or qwen."
            )
        provider = "local_openai"

    if provider == "local_openai":
        timeout_name = "MEDCLAW_LOCAL_TIMEOUT_SEC"
        api_key = os.environ.get("MEDCLAW_LOCAL_API_KEY") or os.environ.get(
            "VLLM_API_KEY"
        )
        base_url = os.environ.get("MEDCLAW_LOCAL_BASE_URL", DEFAULT_LOCAL_BASE_URL)
        model = os.environ.get("MEDCLAW_LOCAL_MODEL", DEFAULT_LOCAL_MODEL)
        enable_thinking = False
    else:
        timeout_name = "MEDCLAW_QWEN_TIMEOUT_SEC"
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        base_url = os.environ.get("MEDCLAW_QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL)
        model = os.environ.get("MEDCLAW_QWEN_MODEL", DEFAULT_RERANK_MODEL)
        enable_thinking = _parse_bool(
            os.environ.get("MEDCLAW_QWEN_ENABLE_THINKING", "true")
        )

    timeout_sec = 120.0
    timeout_text = os.environ.get(timeout_name)
    if timeout_text:
        try:
            timeout_sec = float(timeout_text)
        except ValueError:
            timeout_sec = 120.0
    return RerankConfig(
        provider=provider,
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        model=model,
        timeout_sec=timeout_sec,
        enable_thinking=enable_thinking,
    )


def _maybe_llm_rerank(
    query: str,
    snippets: list[dict[str, Any]],
    *,
    max_snippets: int,
    rerank_mode: str,
    config: RerankConfig,
) -> tuple[list[dict[str, Any]], str, str | None, str | None]:
    if not snippets:
        return [], "not_needed", None, None
    if rerank_mode == "none":
        return _renumber_snippets(snippets[:max_snippets]), "disabled", None, None
    if not config.api_key:
        if rerank_mode == "llm":
            raise RerankError(
                f"The {config.provider} API key is required when rerank_mode='llm'."
            )
        return (
            _renumber_snippets(snippets[:max_snippets]),
            "skipped_no_api_key",
            config.model,
            f"LLM rerank skipped because the {config.provider} API key is not set.",
        )

    try:
        ranked_items, model = _llm_rerank(query, snippets, max_snippets, config)
        reranked = _apply_rerank(snippets, ranked_items, max_snippets=max_snippets)
        return reranked, "llm_reranked", model or config.model, None
    except Exception as exc:
        if rerank_mode == "llm":
            raise RerankError(f"LLM rerank failed: {exc}") from exc
        return (
            _renumber_snippets(snippets[:max_snippets]),
            "fallback_original_order",
            config.model,
            f"LLM rerank failed; used original retrieval order. Reason: {exc}",
        )


def _llm_rerank(
    query: str,
    snippets: list[dict[str, Any]],
    max_snippets: int,
    config: RerankConfig,
) -> tuple[list[dict[str, Any]], str | None]:
    client = _create_rerank_client(config)
    request: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the MedClaw core model reranking retrieved CSCO guideline "
                    "snippets for a medical benchmark. Rank snippets only by relevance "
                    "to the query and clinical decision support value. Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": _rerank_prompt(query, snippets, max_snippets),
            },
        ],
        "temperature": 0,
        "stream": False,
    }
    if config.provider == "qwen":
        request["extra_body"] = {"enable_thinking": config.enable_thinking}
    response = client.chat.completions.create(
        **request,
    )
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise RerankError("Rerank response did not contain choices.")
    message = choices[0].message
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RerankError("Rerank response did not contain text content.")
    data = _extract_json_object(content)
    return _parse_rerank_items(data), getattr(response, "model", None)


def _rerank_prompt(query: str, snippets: list[dict[str, Any]], max_snippets: int) -> str:
    candidates = [
        {
            "chunk_id": snippet.get("chunk_id"),
            "retrieval_rank": snippet.get("rank"),
            "guideline": snippet.get("guideline"),
            "page": snippet.get("page"),
            "page_end": snippet.get("page_end"),
            "heading": snippet.get("heading"),
            "chunk_mode": snippet.get("chunk_mode"),
            "guideline_memory_id": snippet.get("guideline_memory_id"),
            "vector_score": snippet.get("dense_score", snippet.get("score")),
            "lexical_score": snippet.get("lexical_score"),
            "matched_terms": snippet.get("matched_terms", []),
            "excerpt": snippet.get("excerpt"),
        }
        for snippet in snippets
    ]
    payload = {
        "query": query,
        "task": (
            f"Select and order the top {max_snippets} guideline snippets. Prefer snippets "
            "that directly support diagnosis, staging, molecular testing, treatment "
            "planning, or follow-up for the query. Penalize generic front matter."
        ),
        "output_schema": {
            "items": [
                {
                    "chunk_id": "candidate chunk_id",
                    "relevance_score": "integer 0-100",
                    "reason": "short Chinese reason",
                }
            ]
        },
        "candidates": candidates,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _extract_json_object(text: str) -> Any:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.S | re.I)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        start = stripped.find("[")
        end = stripped.rfind("]")
        if start >= 0 and end > start:
            return json.loads(stripped[start : end + 1])
        raise


def _parse_rerank_items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        raw_items = data
    elif isinstance(data, dict):
        raw_items = data.get("items") or data.get("ranked_items") or data.get("rankings")
        if raw_items is None:
            raw_items = data.get("ranked_chunk_ids") or data.get("chunk_ids")
    else:
        raw_items = None
    if not isinstance(raw_items, list):
        raise RerankError("Rerank JSON must contain an items or ranked_chunk_ids list.")

    items: list[dict[str, Any]] = []
    for raw in raw_items:
        if isinstance(raw, str):
            items.append({"chunk_id": raw})
            continue
        if not isinstance(raw, dict):
            continue
        chunk_id = raw.get("chunk_id") or raw.get("id")
        if not isinstance(chunk_id, str) or not chunk_id:
            continue
        score = raw.get("relevance_score", raw.get("score"))
        reason = raw.get("reason", raw.get("rationale"))
        item = {"chunk_id": chunk_id}
        if isinstance(score, (int, float)):
            item["rerank_score"] = float(score)
        if isinstance(reason, str):
            item["rerank_reason"] = reason[:300]
        items.append(item)
    if not items:
        raise RerankError("Rerank JSON did not contain any usable chunk_id.")
    return items


def _apply_rerank(
    snippets: list[dict[str, Any]],
    ranked_items: list[dict[str, Any]],
    *,
    max_snippets: int,
) -> list[dict[str, Any]]:
    by_id = {
        str(snippet.get("chunk_id")): snippet
        for snippet in snippets
        if snippet.get("chunk_id")
    }
    used: set[str] = set()
    ordered: list[dict[str, Any]] = []
    for item in ranked_items:
        chunk_id = str(item.get("chunk_id"))
        if chunk_id in used or chunk_id not in by_id:
            continue
        snippet = dict(by_id[chunk_id])
        if "rerank_score" in item:
            snippet["rerank_score"] = item["rerank_score"]
        if "rerank_reason" in item:
            snippet["rerank_reason"] = item["rerank_reason"]
        ordered.append(snippet)
        used.add(chunk_id)
        if len(ordered) >= max_snippets:
            break
    for snippet in snippets:
        chunk_id = str(snippet.get("chunk_id"))
        if chunk_id in used:
            continue
        ordered.append(dict(snippet))
        if len(ordered) >= max_snippets:
            break
    return _renumber_snippets(ordered[:max_snippets])


def _renumber_snippets(snippets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, snippet in enumerate(snippets, start=1):
        normalized = dict(snippet)
        normalized.setdefault("retrieval_rank", normalized.get("rank"))
        normalized["rank"] = index
        result.append(normalized)
    return result


def _create_rerank_client(config: RerankConfig) -> Any:
    from openai import OpenAI

    return OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout_sec,
    )


def _load_or_build_vector_index(
    chunks: list[GuidelineChunk],
    documents: list[GuidelineDocument],
    *,
    guideline_dir: Path,
    config: EmbeddingConfig,
    force_rebuild: bool,
) -> VectorIndex:
    corpus_hash = _corpus_hash(chunks, documents)
    cache_root = _embedding_cache_root(guideline_dir)
    cache_dir = cache_root / _cache_namespace(config, corpus_hash)
    if not force_rebuild:
        loaded = _try_load_vector_index(cache_dir, corpus_hash, config)
        if loaded is not None:
            return loaded

    if not config.api_key:
        raise EmbeddingCacheError(
            "No reusable guideline embedding cache was found and DASHSCOPE_API_KEY is not set."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings = _embed_texts([chunk.text for chunk in chunks], config)
    embeddings = _normalize_matrix(embeddings)
    _save_vector_cache(cache_dir, chunks, documents, embeddings, corpus_hash, config)
    return _load_vector_index(cache_dir, corpus_hash, config, cache_status="built")


def _embedding_cache_root(guideline_dir: Path) -> Path:
    override = os.environ.get("MEDCLAW_GUIDELINE_EMBEDDING_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return guideline_dir / ".embedding_cache"


def _cache_namespace(config: EmbeddingConfig, corpus_hash: str) -> str:
    model = re.sub(r"[^A-Za-z0-9._-]+", "_", config.model)
    return f"{model}_{config.dimensions}_{corpus_hash[:16]}"


def _try_load_vector_index(
    cache_dir: Path,
    corpus_hash: str,
    config: EmbeddingConfig,
) -> VectorIndex | None:
    try:
        return _load_vector_index(cache_dir, corpus_hash, config, cache_status="reused")
    except (EmbeddingCacheError, FileNotFoundError, ValueError, OSError):
        return None


def _load_vector_index(
    cache_dir: Path,
    corpus_hash: str,
    config: EmbeddingConfig,
    *,
    cache_status: str,
) -> VectorIndex:
    np = _import_numpy()
    manifest_path = cache_dir / "manifest.json"
    chunks_path = cache_dir / "chunks.json"
    embeddings_path = cache_dir / "embeddings.npy"
    if not manifest_path.is_file() or not chunks_path.is_file() or not embeddings_path.is_file():
        raise FileNotFoundError(f"Incomplete embedding cache: {cache_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("corpus_hash") != corpus_hash:
        raise EmbeddingCacheError("Embedding cache corpus hash does not match current guideline files.")
    if manifest.get("embedding_model") != config.model:
        raise EmbeddingCacheError("Embedding cache model does not match current configuration.")
    if int(manifest.get("embedding_dimensions", 0)) != config.dimensions:
        raise EmbeddingCacheError("Embedding cache dimensions do not match current configuration.")

    chunk_dicts = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = [_chunk_from_dict(item) for item in chunk_dicts]
    embeddings = np.load(embeddings_path)
    if len(chunks) != int(embeddings.shape[0]):
        raise EmbeddingCacheError("Embedding cache chunk count does not match embeddings.")

    faiss = _try_import_faiss()
    if faiss is not None:
        index_path = cache_dir / "index.faiss"
        if index_path.is_file():
            index = faiss.read_index(str(index_path))
        else:
            index = _build_faiss_index(embeddings, faiss)
            faiss.write_index(index, str(index_path))
        return VectorIndex(
            chunks=chunks,
            embeddings=embeddings,
            faiss_index=index,
            backend="faiss",
            cache_status=cache_status,
            cache_dir=cache_dir,
            corpus_hash=corpus_hash,
            manifest=manifest,
        )

    return VectorIndex(
        chunks=chunks,
        embeddings=embeddings,
        faiss_index=None,
        backend="numpy_vector",
        cache_status=cache_status,
        cache_dir=cache_dir,
        corpus_hash=corpus_hash,
        manifest=manifest,
    )


def _save_vector_cache(
    cache_dir: Path,
    chunks: list[GuidelineChunk],
    documents: list[GuidelineDocument],
    embeddings: Any,
    corpus_hash: str,
    config: EmbeddingConfig,
) -> None:
    np = _import_numpy()
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "chunks.json").write_text(
        json.dumps([_chunk_to_dict(chunk) for chunk in chunks], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    np.save(cache_dir / "embeddings.npy", embeddings)
    manifest = {
        "schema_version": "1.0",
        "corpus_hash": corpus_hash,
        "embedding_model": config.model,
        "embedding_dimensions": config.dimensions,
        "embedding_base_url": config.base_url,
        "embedding_aggregation": EMBEDDING_AGGREGATION,
        "embedding_segment_max_chars": EMBEDDING_SEGMENT_MAX_CHARS,
        "chunk_count": len(chunks),
        "chunk_modes": sorted({chunk.chunk_mode for chunk in chunks}),
        "source_documents": [
            {
                "source_id": document.source_id,
                "cancer_type": document.cancer_type,
                "title": document.title,
                "filename": document.path.name,
                "sha256": _sha256_file(document.path),
            }
            for document in documents
        ],
    }
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    faiss = _try_import_faiss()
    if faiss is not None:
        index = _build_faiss_index(embeddings, faiss)
        faiss.write_index(index, str(cache_dir / "index.faiss"))


def _embed_query_with_cache(
    query: str,
    cache_dir: Path,
    config: EmbeddingConfig,
) -> tuple[Any, str]:
    np = _import_numpy()
    cache_path = cache_dir / "query_embeddings.json"
    cache: dict[str, Any] = {}
    if cache_path.is_file():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    key = _hash_json(
        {
            "query": query,
            "embedding_model": config.model,
            "embedding_dimensions": config.dimensions,
        }
    )
    if key in cache:
        return _normalize_matrix(np.asarray([cache[key]], dtype="float32"))[0], "reused"
    if not config.api_key:
        raise EmbeddingCacheError(
            "Guideline document embeddings are cached, but this query has no cached embedding "
            "and DASHSCOPE_API_KEY is not set."
        )
    embedding = _embed_texts([query], config)[0]
    embedding = _normalize_matrix(np.asarray([embedding], dtype="float32"))[0]
    cache[key] = embedding.astype("float32").tolist()
    cache_path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return embedding, "built"


def _embed_texts(texts: Sequence[str], config: EmbeddingConfig) -> Any:
    if not config.api_key:
        raise EmbeddingCacheError("DASHSCOPE_API_KEY is required to create missing embeddings.")
    np = _import_numpy()
    segment_inputs: list[str] = []
    segment_owners: list[int] = []
    for owner, source_text in enumerate(texts):
        for segment in _split_embedding_input(source_text):
            segment_inputs.append(segment)
            segment_owners.append(owner)

    client = _create_embedding_client(config)
    segment_vectors: list[list[float]] = []
    for start in range(0, len(segment_inputs), config.batch_size):
        batch = segment_inputs[start : start + config.batch_size]
        try:
            response = client.embeddings.create(
                model=config.model,
                input=batch,
                dimensions=config.dimensions,
                encoding_format="float",
            )
        except Exception as exc:
            raise EmbeddingCacheError(f"DashScope embedding request failed: {exc}") from exc
        data = getattr(response, "data", None)
        if data is None:
            raise EmbeddingCacheError("DashScope embedding response did not contain data.")
        for item in data:
            embedding = getattr(item, "embedding", None)
            if embedding is None and isinstance(item, dict):
                embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise EmbeddingCacheError("DashScope embedding item did not contain a vector.")
            segment_vectors.append([float(value) for value in embedding])
    if len(segment_vectors) != len(segment_inputs):
        raise EmbeddingCacheError(
            f"Expected {len(segment_inputs)} segment embeddings, "
            f"received {len(segment_vectors)}."
        )
    segment_array = np.asarray(segment_vectors, dtype="float32")
    if segment_array.ndim != 2 or segment_array.shape[1] != config.dimensions:
        raise EmbeddingCacheError(
            f"Embedding shape {segment_array.shape} does not match configured "
            f"dimensions {config.dimensions}."
        )
    aggregated = np.zeros((len(texts), config.dimensions), dtype="float32")
    counts = np.zeros((len(texts), 1), dtype="float32")
    for owner, vector in zip(segment_owners, segment_array):
        aggregated[owner] += vector
        counts[owner, 0] += 1.0
    counts[counts == 0] = 1.0
    return aggregated / counts


def _split_embedding_input(
    text: str,
    *,
    max_chars: int = EMBEDDING_SEGMENT_MAX_CHARS,
) -> list[str]:
    """Split one retrieval unit for embedding while preserving one pooled vector."""

    source = str(text).strip()
    if not source:
        return ["[empty]"]
    segments: list[str] = []
    start = 0
    while start < len(source):
        end = min(start + max_chars, len(source))
        if end < len(source):
            search_start = start + max_chars // 2
            candidates = (
                source.rfind("\n\n", search_start, end),
                source.rfind("\n", search_start, end),
                source.rfind(" ", search_start, end),
            )
            boundary = max(candidates)
            if boundary > start:
                end = boundary
        segment = source[start:end].strip()
        if segment:
            segments.append(segment)
        start = end
    return segments or ["[empty]"]


def _create_embedding_client(config: EmbeddingConfig) -> Any:
    from openai import OpenAI

    return OpenAI(api_key=config.api_key, base_url=config.base_url)


def _vector_search(
    vector_index: VectorIndex,
    query: str,
    *,
    query_embedding: Any,
    selected_type: str,
    selected_source_ids: set[str] | None = None,
    max_snippets: int,
) -> list[dict[str, Any]]:
    np = _import_numpy()
    query_vector = np.asarray([query_embedding], dtype="float32")
    if vector_index.faiss_index is not None:
        scores, indices = vector_index.faiss_index.search(
            query_vector,
            len(vector_index.chunks),
        )
        raw_hits = [
            (float(score), int(index))
            for score, index in zip(scores[0].tolist(), indices[0].tolist())
            if index >= 0
        ]
    else:
        scores = vector_index.embeddings @ query_vector[0]
        raw_hits = [
            (float(score), int(index))
            for index, score in enumerate(scores.tolist())
        ]
        raw_hits.sort(key=lambda item: item[0], reverse=True)

    terms = _query_terms(query)
    snippets: list[dict[str, Any]] = []
    for dense_score, index in raw_hits:
        chunk = vector_index.chunks[index]
        if selected_type != "all" and chunk.cancer_type != selected_type:
            continue
        if selected_source_ids is not None and chunk.source_id not in selected_source_ids:
            continue
        lexical_score, matched_terms = _score_chunk(chunk, terms)
        snippets.append(
            _snippet_from_chunk(
                chunk,
                rank=len(snippets) + 1,
                score=dense_score,
                matched_terms=matched_terms,
                excerpt_terms=matched_terms or terms,
                dense_score=dense_score,
                lexical_score=lexical_score,
            )
        )
        if len(snippets) >= max_snippets:
            break
    return snippets


def _lexical_search(
    chunks: Iterable[GuidelineChunk],
    query: str,
    *,
    max_snippets: int,
) -> list[dict[str, Any]]:
    terms = _query_terms(query)
    scored = []
    for chunk in chunks:
        score, matched_terms = _score_chunk(chunk, terms)
        if score <= 0:
            continue
        scored.append((score, chunk, matched_terms))
    scored.sort(key=lambda item: (-item[0], item[1].source_id, item[1].page or 0))
    return [
        _snippet_from_chunk(
            chunk,
            rank=index + 1,
            score=score,
            matched_terms=matched_terms,
            excerpt_terms=matched_terms or terms,
            dense_score=None,
            lexical_score=score,
        )
        for index, (score, chunk, matched_terms) in enumerate(scored[:max_snippets])
    ]


def _snippet_from_chunk(
    chunk: GuidelineChunk,
    *,
    rank: int,
    score: float,
    matched_terms: list[str],
    excerpt_terms: list[str],
    dense_score: float | None,
    lexical_score: float,
) -> dict[str, Any]:
    snippet = {
        "rank": rank,
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "guideline": chunk.title,
        "cancer_type": chunk.cancer_type,
        "page": chunk.page,
        "page_end": chunk.page_end,
        "heading": chunk.heading,
        "chunk_mode": chunk.chunk_mode,
        "guideline_memory_id": chunk.guideline_memory_id,
        "planner_guideline_id": chunk.planner_guideline_id,
        "source_rule_ids": list(chunk.source_rule_ids)[:12],
        "source_span_ids": list(chunk.source_span_ids)[:12],
        "score": round(float(score), 6),
        "lexical_score": round(float(lexical_score), 3),
        "matched_terms": matched_terms[:12],
        "excerpt": _excerpt(chunk.text, excerpt_terms),
    }
    if dense_score is not None:
        snippet["dense_score"] = round(float(dense_score), 6)
    return snippet


def _query_terms(query: str) -> list[str]:
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9+./_-]*|[\u4e00-\u9fff]+", query):
        token = token.strip()
        if not token:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", token) and len(token) > 4:
            terms.extend(_cjk_ngrams(token))
        else:
            terms.append(token)
    lower_query = query.lower()
    if any(term in query for term in ("分期", "分级", "TNM")):
        terms.extend(["分期", "TNM", "AJCC", "PET", "脑", "骨", "影像"])
    if any(
        term in lower_query
        for term in (
            "分子",
            "基因",
            "biomarker",
            "egfr",
            "alk",
            "pd-l1",
            "mmr",
            "msi",
            "tmb",
            "pole",
        )
    ):
        terms.extend(
            [
                "分子",
                "检测",
                "生物标志物",
                "突变",
                "融合",
                "扩增",
                "表达",
                "分子分型",
                "MSI",
                "MMR",
                "TMB",
            ]
        )
    if any(term in query for term in ("治疗", "一线", "辅助", "新辅助", "免疫", "靶向")):
        terms.extend(["治疗", "一线", "辅助", "新辅助", "免疫", "靶向", "推荐"])
    if any(term in query for term in ("病理", "免疫组化", "IHC")):
        terms.extend(["病理", "免疫组化", "IHC", "TTF-1", "Napsin", "p40", "神经内分泌"])
    return _unique_preserve_order(terms)


def _cjk_ngrams(text: str) -> list[str]:
    grams: list[str] = []
    max_size = min(6, len(text))
    for size in range(2, max_size + 1):
        for index in range(0, len(text) - size + 1):
            grams.append(text[index : index + size])
    return grams


def _score_chunk(chunk: GuidelineChunk, terms: list[str]) -> tuple[float, list[str]]:
    haystack = _normalize_for_match(chunk.text)
    heading = _normalize_for_match(chunk.heading)
    score = 0.0
    matched: list[str] = []
    for term in terms:
        normalized = _normalize_for_match(term)
        if not normalized:
            continue
        count = haystack.count(normalized)
        if count <= 0:
            continue
        matched.append(term)
        weight = 1.0 + min(len(normalized), 12) / 6.0
        score += min(count, 5) * weight
        if normalized in heading:
            score += 3.0
    return score, _unique_preserve_order(matched)


def _excerpt(text: str, terms: list[str]) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= MAX_EXCERPT_CHARS:
        return compact
    lower = _normalize_for_match(compact)
    best_index = -1
    for term in terms:
        normalized = _normalize_for_match(term)
        if not normalized:
            continue
        index = lower.find(normalized)
        if index >= 0:
            best_index = index
            break
    if best_index < 0:
        best_index = 0
    start = max(0, best_index - MAX_EXCERPT_CHARS // 3)
    end = min(len(compact), start + MAX_EXCERPT_CHARS)
    start = max(0, end - MAX_EXCERPT_CHARS)
    excerpt = compact[start:end]
    if start > 0:
        excerpt = "..." + excerpt
    if end < len(compact):
        excerpt += "..."
    return excerpt


def _build_faiss_index(embeddings: Any, faiss: Any) -> Any:
    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(embeddings)
    return index


def _normalize_matrix(array: Any) -> Any:
    np = _import_numpy()
    matrix = np.asarray(array, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _corpus_hash(chunks: list[GuidelineChunk], documents: list[GuidelineDocument]) -> str:
    return _hash_json(
        {
            "documents": [
                {
                    "source_id": document.source_id,
                    "filename": document.path.name,
                    "sha256": _sha256_file(document.path),
                }
                for document in documents
            ],
            "chunking": {
                "chunk_max_chars": CHUNK_MAX_CHARS,
                "chunk_overlap_chars": CHUNK_OVERLAP_CHARS,
                "chunk_modes": sorted({chunk.chunk_mode for chunk in chunks}),
                "embedding_aggregation": EMBEDDING_AGGREGATION,
                "embedding_segment_max_chars": EMBEDDING_SEGMENT_MAX_CHARS,
            },
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "chunk_mode": chunk.chunk_mode,
                    "text_sha256": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
                }
                for chunk in chunks
            ],
        }
    )


def _chunk_to_dict(chunk: GuidelineChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "source_id": chunk.source_id,
        "cancer_type": chunk.cancer_type,
        "title": chunk.title,
        "page": chunk.page,
        "page_end": chunk.page_end,
        "heading": chunk.heading,
        "text": chunk.text,
        "chunk_mode": chunk.chunk_mode,
        "guideline_memory_id": chunk.guideline_memory_id,
        "planner_guideline_id": chunk.planner_guideline_id,
        "source_rule_ids": list(chunk.source_rule_ids),
        "source_span_ids": list(chunk.source_span_ids),
    }


def _chunk_from_dict(value: dict[str, Any]) -> GuidelineChunk:
    return GuidelineChunk(
        chunk_id=str(value["chunk_id"]),
        source_id=str(value["source_id"]),
        cancer_type=str(value["cancer_type"]),
        title=str(value["title"]),
        page=value.get("page") if isinstance(value.get("page"), int) else None,
        heading=str(value.get("heading") or ""),
        text=str(value["text"]),
        page_end=value.get("page_end") if isinstance(value.get("page_end"), int) else None,
        chunk_mode=str(value.get("chunk_mode") or "chapter"),
        guideline_memory_id=(
            str(value["guideline_memory_id"])
            if value.get("guideline_memory_id")
            else None
        ),
        planner_guideline_id=(
            str(value["planner_guideline_id"])
            if value.get("planner_guideline_id")
            else None
        ),
        source_rule_ids=tuple(str(item) for item in value.get("source_rule_ids", [])),
        source_span_ids=tuple(str(item) for item in value.get("source_span_ids", [])),
    )


def _import_numpy() -> Any:
    try:
        import numpy as np
    except Exception as exc:
        raise EmbeddingCacheError(
            "numpy is required for guideline vector retrieval. Install requirements.txt."
        ) from exc
    return np


def _try_import_faiss() -> Any | None:
    try:
        import faiss  # type: ignore[import-not-found]
    except Exception:
        return None
    return faiss


def _normalize_for_match(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _bounded_int(value: Any, *, default: int, lower: int, upper: int) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        integer = default
    return max(lower, min(upper, integer))


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
