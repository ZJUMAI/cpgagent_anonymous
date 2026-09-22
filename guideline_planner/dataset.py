"""Training-data construction for latent guideline-memory planner tasks."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from guideline_planner.constants import DEFAULT_TASK_RATIOS, TASK_TOKENS
from guideline_planner.io_utils import read_jsonl, write_json, write_jsonl


def build_training_data(
    chunks: str | Path | Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    task_ratios: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build the clean V2 Memory Encoder task families.

    PLAN supervision deliberately does not originate from guideline chunks. It
    must be built from validated ``planner_trajectory.v2`` transitions instead.
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    legacy_plan = output / "plan.jsonl"
    if legacy_plan.is_file():
        legacy_plan.unlink()
    records = _load_chunks(chunks)
    ratios = normalize_task_ratios(task_ratios or DEFAULT_TASK_RATIOS)

    ae = [_autoencoding_example(chunk) for chunk in records]
    retrieval = [_retrieval_example(chunk, index, records) for index, chunk in enumerate(records)]
    continuation = [
        _continuation_example(chunk, _next_chunk(index, records))
        for index, chunk in enumerate(records)
    ]
    task_map = {
        "AE": ae,
        "RETRIEVE": retrieval,
        "CONTINUE": continuation,
    }
    for task, examples in task_map.items():
        write_jsonl(output / f"{task.lower()}.jsonl", examples)

    mixed = [
        example
        for task in ("AE", "RETRIEVE", "CONTINUE")
        for example in task_map[task]
    ]
    write_jsonl(output / "mixed_train.jsonl", mixed)
    manifest = {
        "schema_version": "memory_training_data.v2",
        "training_stage": "memory_encoder",
        "chunk_count": len(records),
        "task_counts": {task: len(items) for task, items in task_map.items()},
        "task_sampling_ratios": ratios,
        "sampling_policy": "exact_task_schedule_with_sqrt_cancer_guideline_groups",
        "plan_supervision": "excluded; use planner_trajectory.v2",
        "files": {
            "AE": str(output / "ae.jsonl"),
            "RETRIEVE": str(output / "retrieve.jsonl"),
            "CONTINUE": str(output / "continue.jsonl"),
            "mixed": str(output / "mixed_train.jsonl"),
        },
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def normalize_task_ratios(values: Mapping[str, float]) -> dict[str, float]:
    ratios = {
        task: float(values.get(task, DEFAULT_TASK_RATIOS[task]))
        for task in DEFAULT_TASK_RATIOS
    }
    total = sum(max(value, 0.0) for value in ratios.values())
    if total <= 0:
        return dict(DEFAULT_TASK_RATIOS)
    return {task: max(value, 0.0) / total for task, value in ratios.items()}


def _load_chunks(chunks: str | Path | Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(chunks, (str, Path)):
        return read_jsonl(Path(chunks))
    return [dict(chunk) for chunk in chunks]


def _autoencoding_example(chunk: Mapping[str, Any]) -> dict[str, Any]:
    target = {
        "guideline_memory_id": chunk["source_span_id"],
        "guideline_id": chunk["guideline_id"],
        "version": chunk["version"],
        "cancer_type": chunk["cancer_type"],
        "chapter": chunk["chapter"],
        "section": chunk["section"],
        "source_rule_ids": chunk.get("source_rule_ids", []),
        "structured_guideline_content": _section_summary(chunk["text"], max_chars=900),
    }
    return {
        "task": "AE",
        "task_token": TASK_TOKENS["AE"],
        "guideline_memory_id": chunk["source_span_id"],
        "guideline_id": chunk["guideline_id"],
        "version": chunk["version"],
        "cancer_type": chunk["cancer_type"],
        "encoder_text": chunk["text"],
        "prompt": "请将该指南 section 重构为结构化 guideline JSON。",
        "target": json.dumps(target, ensure_ascii=False, sort_keys=True),
    }


def _retrieval_example(
    chunk: Mapping[str, Any],
    index: int,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    strong = [chunk["source_span_id"]]
    source_rules = set(chunk.get("source_rule_ids", []))
    weak = [
        candidate["source_span_id"]
        for candidate in _same_cancer(records, chunk)
        if candidate["source_span_id"] != chunk["source_span_id"]
        and (
            bool(source_rules & set(candidate.get("source_rule_ids", [])))
            or (
                candidate.get("guideline_id") == chunk.get("guideline_id")
                and candidate.get("chapter") == chunk.get("chapter")
            )
        )
    ][:2]
    hard = [
        candidate["source_span_id"]
        for candidate in _same_cancer(records, chunk)
        if candidate["source_span_id"] != chunk["source_span_id"]
        and candidate.get("version") == chunk.get("version")
        and not (source_rules & set(candidate.get("source_rule_ids", [])))
    ][:3]
    easy = [
        candidate["source_span_id"]
        for candidate in records
        if candidate.get("cancer_type") != chunk.get("cancer_type")
    ][:3]
    query = f"{chunk['cancer_type']} {chunk['h1_title']} {', '.join(_keywords(chunk['text'])[:8])}"
    return {
        "task": "RETRIEVE",
        "task_token": TASK_TOKENS["RETRIEVE"],
        "guideline_memory_id": chunk["source_span_id"],
        "guideline_id": chunk["guideline_id"],
        "version": chunk["version"],
        "cancer_type": chunk["cancer_type"],
        "query": query.strip(),
        "strong_positive": strong,
        "weak_positive": weak,
        "hard_negative": hard,
        "easy_negative": easy,
        "positive_slot_ids": strong + weak,
        "negative_slot_ids": hard + easy,
        "encoder_text": chunk["text"],
        "prompt": "选择最相关的 latent guideline memory slot。",
        "target": chunk["source_span_id"],
        "record_index": index,
    }


def _continuation_example(
    chunk: Mapping[str, Any],
    next_chunk: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if next_chunk is None:
        target = {
            "next_clinical_phase": "complete_or_followup",
            "next_pathway": None,
            "next_section_summary": "当前指南 topic 后没有同指南的下一个一级标题 section。",
        }
    else:
        target = {
            "next_clinical_phase": _phase_from_title(next_chunk["h1_title"]),
            "next_pathway": next_chunk["h1_title"],
            "next_section_summary": _section_summary(next_chunk["text"], max_chars=420),
            "next_guideline_memory_id": next_chunk["source_span_id"],
        }
    return {
        "task": "CONTINUE",
        "task_token": TASK_TOKENS["CONTINUE"],
        "guideline_memory_id": chunk["source_span_id"],
        "guideline_id": chunk["guideline_id"],
        "version": chunk["version"],
        "cancer_type": chunk["cancer_type"],
        "encoder_text": chunk["text"],
        "prompt": "根据当前指南 section，预测下一 clinical phase / pathway。",
        "target": json.dumps(target, ensure_ascii=False, sort_keys=True),
    }


def _same_cancer(records: list[dict[str, Any]], chunk: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [record for record in records if record.get("cancer_type") == chunk.get("cancer_type")]


def _next_chunk(index: int, records: list[dict[str, Any]]) -> dict[str, Any] | None:
    current = records[index]
    for candidate in records[index + 1 :]:
        if candidate.get("guideline_id") == current.get("guideline_id"):
            return candidate
    return None


def _keywords(text: str) -> list[str]:
    needles = re.findall(r"[\u4e00-\u9fffA-Za-z0-9\-+/]{2,}", text)
    stop = {"指南", "推荐", "患者", "治疗", "诊断", "可以", "进行", "以及", "the", "and"}
    seen: set[str] = set()
    result: list[str] = []
    for item in needles:
        if item in stop or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _topic_overlap(left: str, right: str) -> int:
    return len(set(_keywords(left)) & set(_keywords(right)))


def _section_summary(text: str, *, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:max_chars]


def _phase_from_title(title: str) -> str:
    if any(word in title for word in ("随访", "复查", "监测")):
        return "followup"
    if any(word in title for word in ("复发", "转移", "晚期", "解救")):
        return "advanced_or_recurrent_treatment"
    if any(word in title for word in ("治疗", "辅助", "新辅助", "放疗", "化疗", "免疫", "靶向")):
        return "treatment_planning"
    if any(word in title for word in ("分期", "诊断", "检查", "病理", "影像", "检测")):
        return "diagnostic_workup"
    return "guideline_review"
