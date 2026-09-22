"""LLM extraction of structured benchmark case facts from integrated reports."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping


class ReportExtractionError(ValueError):
    """Raised when an LLM extraction response cannot be parsed."""


@dataclass(frozen=True)
class LLMReportExtractor:
    """Extract clinical and molecular case JSON from a full text report."""

    llm_client: Any

    def extract(self, *, case_id: str, report_text: str) -> dict[str, Any]:
        """Call the LLM and return normalized extraction JSON."""

        completion = self.llm_client.complete(
            messages=_messages(case_id=case_id, report_text=report_text),
            tools=[],
            tool_choice="none",
        )
        content = completion.message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ReportExtractionError("Report extractor response did not contain text.")
        data = _extract_json_object(content)
        if not isinstance(data, dict):
            raise ReportExtractionError("Report extractor JSON must be an object.")
        return _normalize_extraction(
            data,
            model=completion.model,
            usage=completion.usage,
        )


def _messages(*, case_id: str, report_text: str) -> list[dict[str, str]]:
    schema = {
        "clinical": {
            "project_id": "string or null",
            "age": "number in years or null",
            "sex": "female|male|other|not_available",
            "race": "string or null",
            "ethnicity": "string or null",
            "country": "string or null",
            "primary_site": "string or null",
            "disease_type": "string or null",
            "primary_diagnosis": "string or null",
            "diagnosis_age_days": "number or null",
            "tumor_classification": "string or null",
            "diagnosis_is_primary_disease": "boolean|string|null",
            "prior_malignancy": "string or boolean or null",
            "prior_treatment": "string or boolean or null",
            "smoking_status": "string or null",
            "pack_years": "number or null",
            "tumor_location": "string or null",
            "tumor_size_cm": "number or null",
            "histologic_grade": "string or null",
            "surgical_margins": "string or null",
            "pleural_involvement": "string or null",
            "lymphovascular_invasion": "string or null",
            "perineural_invasion": "string or null",
            "lymph_node_status": "string or null",
            "pathologic_t_stage": "string or null",
            "pathologic_n_stage": "string or null",
            "pathologic_m_stage": "string or null",
            "pathologic_stage": "string or null",
            "vital_status_at_last_follow_up": "alive|dead|not_available",
            "lost_to_follow_up": "boolean|null",
        },
        "treatment": {
            "treatments": [
                {
                    "treatment_type": "string",
                    "treatment_type_english": "string or null",
                    "received": "yes|no|unknown",
                    "intent": "string or null",
                    "site": "string or null",
                    "evidence": "short quote or paraphrase",
                }
            ]
        },
        "follow_up": {
            "vital_status": "alive|dead|not_available",
            "last_follow_up_day": "number or null",
            "disease_status_at_last_follow_up": "string or null",
            "progression": {
                "occurred": "boolean|string|null",
                "type": "string or null",
                "evidence": "string or null",
                "day": "number or null",
            },
        },
        "molecular": {
            "molecular_subtype": "string or null",
            "summary": "string or null",
            "high_amplification_genes": ["gene symbols"],
            "copy_number_summary": "string or null",
            "biomarkers": {
                "EGFR": {
                    "alteration_status": "positive|negative|not_reported|uncertain",
                    "mutation": "string or null",
                    "cnv": "string or null",
                    "rna_expression": "string or number or null",
                    "interpretation": "string or null",
                    "evidence": "short quote or paraphrase",
                }
            },
        },
        "warnings": ["uncertain or conflicting extraction notes"],
    }
    biomarker_list = (
        "EGFR, ALK, KRAS, BRAF, ROS1, RET, MET, ERBB2, NTRK1, NTRK2, "
        "NTRK3, TP53, STK11, KEAP1, SMARCA4, RB1, CDKN2A, PIK3CA, NF1, ATM, "
        "PD-L1, TMB"
    )
    system = (
        "You are a careful medical data extraction engine for a research benchmark. "
        "Extract only facts explicitly supported by the supplied report. "
        "Do not infer unavailable biomarkers or guideline recommendations. "
        "Return JSON only, with no Markdown."
    )
    user = {
        "case_id": case_id,
        "task": (
            "从完整病例报告中抽取 case-level clinical、treatment、follow_up 和 "
            "molecular/biomarker 结构化信息。guideline 暂时不要生成。"
        ),
        "biomarkers_to_extract": biomarker_list,
        "missing_value_rule": (
            "如果报告没有明确给出某字段，填 null 或 not_available；不要凭常识补全。"
        ),
        "required_schema": schema,
        "report_text": report_text,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def _normalize_extraction(
    data: Mapping[str, Any],
    *,
    model: str | None,
    usage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = {
        "status": "success",
        "method": "llm",
        "model": model,
        "usage": dict(usage) if isinstance(usage, Mapping) else None,
        "clinical": _object(data.get("clinical")),
        "treatment": _object(data.get("treatment")),
        "follow_up": _object(data.get("follow_up")),
        "molecular": _object(data.get("molecular")),
        "warnings": _string_list(data.get("warnings")),
    }
    if not result["clinical"] and not result["molecular"]:
        raise ReportExtractionError(
            "Report extractor JSON must contain at least clinical or molecular data."
        )
    return result


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ReportExtractionError("No JSON object found in extractor response.")
        try:
            data = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ReportExtractionError(
                f"Invalid extractor JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
            ) from exc
    if not isinstance(data, dict):
        raise ReportExtractionError("Extractor response JSON must be an object.")
    return data


def _object(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str) and value:
        return [value]
    return []
