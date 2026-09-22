"""Rule-based extraction of patient events from TCGA-style markdown reports."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from medclaw.trajectory.schema import PatientEvent, STEP_BY_STAGE, Stage

NA_VALUES = {
    "",
    "na",
    "n/a",
    "none",
    "unknown",
    "not reported",
    "not_available",
    "missing",
}

TABLE_ROW_RE = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*(.*?)\s*\|$")


def extract_patient_events(
    case_id: str,
    report_path: Path | str,
    *,
    clinical_path: Path | str | None = None,
) -> list[PatientEvent]:
    """Extract a deterministic event table from a report and optional clinical JSON.

    Structured clinical JSON is authoritative for identity and staging. When the
    report also exposes those facts, contradictions fail closed instead of
    silently producing a mixed-patient reference trajectory.
    """

    path = Path(report_path)
    report_text = path.read_text(encoding="utf-8")
    fields = _parse_markdown_fields(report_text)
    if clinical_path is not None:
        clinical = _read_clinical(case_id, Path(clinical_path))
        _validate_report_consistency(report_text, fields, clinical)
        fields = _supplement_free_text_fields(fields, report_text)
        fields = _merge_clinical_fields(fields, clinical)
    if not fields:
        return _extract_free_text_events(case_id, report_text, source_name=path.name)
    events: list[PatientEvent] = []

    def add(
        stage: Stage,
        event_type: str,
        field_names: Iterable[str],
        content: str,
        *,
        attributes: dict[str, object] | None = None,
    ) -> None:
        evidence_parts = []
        for name in field_names:
            for value in fields.get(name, []):
                evidence_parts.append(f"{name} = {value}")
        evidence_span = "; ".join(evidence_parts) if evidence_parts else content
        events.append(
            PatientEvent(
                event_id=f"e{len(events)}",
                case_id=case_id,
                stage=stage,
                event_type=event_type,
                content=content,
                source="patient_report",
                available_at_step=STEP_BY_STAGE[stage],
                evidence_span=evidence_span,
                attributes=dict(attributes or {}),
            )
        )

    age = _first(fields, "demographic.age_at_index")
    sex = _first(fields, "demographic.gender")
    race = _first(fields, "demographic.race")
    primary_site = _first(fields, "cases.primary_site")
    disease_type = _first(fields, "cases.disease_type")
    if any([age, sex, race, primary_site, disease_type]):
        content = ", ".join(
            part
            for part in [
                f"{age}-year-old" if age else "",
                sex or "",
                race or "",
                primary_site or "",
                disease_type or "",
            ]
            if part
        )
        add(
            "baseline",
            "demographic_and_disease_context",
            [
                "demographic.age_at_index",
                "demographic.gender",
                "demographic.race",
                "cases.primary_site",
                "cases.disease_type",
            ],
            content,
            attributes={
                "age": _int_or_none(age),
                "sex": sex,
                "race": race,
                "primary_site": primary_site,
                "disease_type": disease_type,
            },
        )

    diagnosis = _first(fields, "diagnoses.primary_diagnosis")
    if diagnosis:
        add(
            "diagnosis_confirmation",
            "pathology",
            ["diagnoses.primary_diagnosis"],
            diagnosis,
            attributes={"diagnosis": diagnosis},
        )

    t_stage = _first(fields, "diagnoses.ajcc_pathologic_t")
    n_stage = _first(fields, "diagnoses.ajcc_pathologic_n")
    m_stage = _first(fields, "diagnoses.ajcc_pathologic_m")
    stage_group = _first(fields, "diagnoses.ajcc_pathologic_stage")
    if any([t_stage, n_stage, m_stage, stage_group]):
        parts = [part for part in [t_stage, n_stage, m_stage, stage_group] if part]
        add(
            "staging",
            "tnm",
            [
                "diagnoses.ajcc_pathologic_t",
                "diagnoses.ajcc_pathologic_n",
                "diagnoses.ajcc_pathologic_m",
                "diagnoses.ajcc_pathologic_stage",
            ],
            " ".join(parts),
            attributes={
                "T": t_stage,
                "N": n_stage,
                "M": m_stage,
                "stage_group": stage_group,
            },
        )

    molecular_events = _molecular_events(fields)
    for item in molecular_events:
        add(
            "biomarker_assessment",
            "molecular_test",
            item["fields"],
            item["content"],
            attributes=item["attributes"],
        )

    for treatment in _treatment_events(fields):
        add(
            "treatment_observed",
            "therapy",
            treatment["fields"],
            treatment["content"],
            attributes=treatment["attributes"],
        )

    vital_status = _first(fields, "demographic.vital_status")
    days_to_death = _first(fields, "demographic.days_to_death")
    if vital_status or days_to_death:
        content = ", ".join(
            part
            for part in [
                f"vital status: {vital_status}" if vital_status else "",
                f"days to death: {days_to_death}" if days_to_death else "",
            ]
            if part
        )
        add(
            "followup_or_outcome",
            "outcome",
            ["demographic.vital_status", "demographic.days_to_death"],
            content,
            attributes={"vital_status": vital_status, "days_to_death": days_to_death},
        )

    return events


def _parse_markdown_fields(text: str) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = defaultdict(list)
    for line in text.splitlines():
        match = TABLE_ROW_RE.match(line.strip())
        if not match:
            continue
        field, value = match.group(1), _strip_markdown_value(match.group(2))
        if field == "字段" or _is_na(value):
            continue
        if value not in fields[field]:
            fields[field].append(value)
    return dict(fields)


def _extract_free_text_events(
    case_id: str, text: str, *, source_name: str
) -> list[PatientEvent]:
    events: list[PatientEvent] = []

    def add(
        stage: Stage,
        event_type: str,
        content: str,
        evidence_span: str,
        attributes: dict[str, object] | None = None,
    ) -> None:
        events.append(
            PatientEvent(
                event_id=f"e{len(events)}",
                case_id=case_id,
                stage=stage,
                event_type=event_type,
                content=content,
                source=f"patient_report:{source_name}",
                available_at_step=STEP_BY_STAGE[stage],
                evidence_span=evidence_span,
                attributes=dict(attributes or {}),
            )
        )

    age_match = re.search(r"(\d+)\s*岁", text)
    if age_match is None:
        age_match = re.search(r"\b(\d{1,3})-year-old\b", text, re.I)
    sex = _free_text_sex(text)
    if age_match or sex or "肺" in text:
        age = _int_or_none(age_match.group(1)) if age_match else None
        primary_site = "Bronchus and lung" if "肺" in text else None
        content = ", ".join(
            part
            for part in [
                f"{age}-year-old" if age is not None else "",
                sex or "",
                primary_site or "",
            ]
            if part
        )
        add(
            "baseline",
            "demographic_and_disease_context",
            content or "baseline context reported",
            _short_evidence(text, age_match.start() if age_match else 0),
            {"age": age, "sex": sex, "primary_site": primary_site},
        )

    diagnosis_match = re.search(r"(Adenocarcinoma[^。\n]*|肺腺癌)", text, re.I)
    if diagnosis_match:
        diagnosis = diagnosis_match.group(1).strip()
        add(
            "diagnosis_confirmation",
            "pathology",
            diagnosis,
            _short_evidence(text, diagnosis_match.start()),
            {"diagnosis": diagnosis},
        )

    tnm_match = re.search(
        r"\b(p?T[0-9X][a-c]?)\s*(p?N[0-9X][a-c]?)\s*(p?M[0-9X][a-c]?)\b",
        text,
        re.I,
    )
    if tnm_match:
        t_value, n_value, m_value = tnm_match.groups()
        add(
            "staging",
            "tnm",
            f"{t_value} {n_value} {m_value}",
            _short_evidence(text, tnm_match.start()),
            {"T": t_value, "N": n_value, "M": m_value, "stage_group": None},
        )

    treatment_section = _free_text_section(text, "治疗信息")
    for name, keyword in [
        ("Surgery, NOS", "手术"),
        ("Radiation Therapy, NOS", "放射治疗"),
        ("Pharmaceutical Therapy, NOS", "药物治疗"),
    ]:
        index = treatment_section.find(keyword)
        if index < 0:
            continue
        status = "recorded_as_no" if "未接受该疗法" in treatment_section else "observed_or_recorded"
        add(
            "treatment_observed",
            "therapy",
            name,
            _short_evidence(treatment_section, index),
            {
                "treatment_type": name,
                "status": status,
                "note": "free-text treatment section",
            },
        )

    outcome_match = re.search(r"(生存状态为存活|vital status[^。\n]*alive|Alive)", text, re.I)
    if outcome_match:
        add(
            "followup_or_outcome",
            "outcome",
            "vital status: Alive",
            _short_evidence(text, outcome_match.start()),
            {"vital_status": "Alive", "days_to_death": None},
        )

    return events


def _short_evidence(text: str, start: int, *, radius: int = 120) -> str:
    left = max(0, start - radius)
    right = min(len(text), start + radius)
    return " ".join(text[left:right].split())


def _free_text_section(text: str, heading: str) -> str:
    pattern = re.compile(rf"#+\s*{re.escape(heading)}\s*(.*?)(?=\n#+\s|\Z)", re.DOTALL)
    match = pattern.search(text)
    return match.group(1) if match else text


def _strip_markdown_value(value: str) -> str:
    return value.strip().strip("`").strip()


CLINICAL_FIELD_MAP = {
    "age": "demographic.age_at_index",
    "sex": "demographic.gender",
    "race": "demographic.race",
    "primary_site": "cases.primary_site",
    "disease_type": "cases.disease_type",
    "primary_diagnosis": "diagnoses.primary_diagnosis",
    "pathologic_t_stage": "diagnoses.ajcc_pathologic_t",
    "pathologic_n_stage": "diagnoses.ajcc_pathologic_n",
    "pathologic_m_stage": "diagnoses.ajcc_pathologic_m",
    "pathologic_stage": "diagnoses.ajcc_pathologic_stage",
}


def _read_clinical(case_id: str, path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Clinical JSON does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Clinical JSON must contain an object: {path}")
    clinical_case_id = str(data.get("case_id") or "").strip()
    if clinical_case_id and clinical_case_id != case_id:
        raise ValueError(
            f"Clinical case_id mismatch: expected {case_id}, found {clinical_case_id}"
        )
    return data


def _merge_clinical_fields(
    fields: dict[str, list[str]], clinical: dict[str, object]
) -> dict[str, list[str]]:
    merged = {key: list(values) for key, values in fields.items()}
    for clinical_key, report_key in CLINICAL_FIELD_MAP.items():
        value = clinical.get(clinical_key)
        if value is None or _is_na(str(value)):
            continue
        normalized = _normalize_sex(str(value)) if clinical_key == "sex" else str(value).strip()
        if normalized:
            merged[report_key] = [normalized]
    return merged


def _supplement_free_text_fields(
    fields: dict[str, list[str]], text: str
) -> dict[str, list[str]]:
    supplemented = {key: list(values) for key, values in fields.items()}
    age = re.search(r"(\d{1,3})\s*岁", text) or re.search(
        r"\b(\d{1,3})-year-old\b", text, re.I
    )
    if age:
        supplemented.setdefault("demographic.age_at_index", [age.group(1)])
    sex = _free_text_sex(text)
    if sex:
        supplemented.setdefault("demographic.gender", [sex])
    tnm = re.search(
        r"\b(p?T[0-9X][a-c]?)\s*[, /-]*\s*(p?N[0-9X][a-c]?)\s*[, /-]*\s*(p?M[0-9X][a-c]?)\b",
        text,
        re.I,
    )
    if tnm:
        for key, value in zip(
            (
                "diagnoses.ajcc_pathologic_t",
                "diagnoses.ajcc_pathologic_n",
                "diagnoses.ajcc_pathologic_m",
            ),
            tnm.groups(),
            strict=True,
        ):
            supplemented.setdefault(key, [value])
    return supplemented


def _validate_report_consistency(
    text: str,
    fields: dict[str, list[str]],
    clinical: dict[str, object],
) -> None:
    report_sex = _normalize_sex(_first(fields, "demographic.gender")) or _free_text_sex(text)
    clinical_sex = _normalize_sex(str(clinical.get("sex") or ""))
    if report_sex and clinical_sex and report_sex != clinical_sex:
        raise ValueError(
            f"Sex conflict between report ({report_sex}) and clinical JSON ({clinical_sex})"
        )

    report_tnm = {
        "T": _first(fields, "diagnoses.ajcc_pathologic_t"),
        "N": _first(fields, "diagnoses.ajcc_pathologic_n"),
        "M": _first(fields, "diagnoses.ajcc_pathologic_m"),
    }
    if not any(report_tnm.values()):
        match = re.search(
            r"\b(p?T[0-9X][a-c]?)\s*[, /-]*\s*(p?N[0-9X][a-c]?)\s*[, /-]*\s*(p?M[0-9X][a-c]?)\b",
            text,
            re.I,
        )
        if match:
            report_tnm = dict(zip(("T", "N", "M"), match.groups(), strict=True))
    clinical_tnm = {
        "T": clinical.get("source_pathologic_t_stage") or clinical.get("pathologic_t_stage"),
        "N": clinical.get("source_pathologic_n_stage") or clinical.get("pathologic_n_stage"),
        "M": clinical.get("source_pathologic_m_stage") or clinical.get("pathologic_m_stage"),
    }
    for key in ("T", "N", "M"):
        report_value = _normalize_tnm(report_tnm.get(key))
        clinical_value = _normalize_tnm(clinical_tnm.get(key))
        if report_value and clinical_value and report_value != clinical_value:
            raise ValueError(
                f"{key}-stage conflict between report ({report_value}) and "
                f"clinical JSON ({clinical_value})"
            )


def _free_text_sex(text: str) -> str | None:
    # Prefer explicit patient-summary phrases. Word boundaries deliberately do
    # not match the feature name ``is_female`` because underscore is a word char.
    patterns = (
        r"\d{1,3}\s*岁[^。\n]{0,40}?(男性|女性)",
        r"\b\d{1,3}-year-old\s+(?:[a-z-]+\s+){0,4}(male|female)\b",
        r"\b(?:patient|case)\b[^.\n]{0,100}?\b(male|female)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return _normalize_sex(match.group(1))
    feature = re.search(r"\|\s*is_female\s*\|\s*([01](?:\.0+)?)\s*\|", text, re.I)
    if feature:
        return "female" if float(feature.group(1)) == 1.0 else "male"
    return None


def _normalize_sex(value: str | None) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"female", "女性", "f", "1", "1.0"}:
        return "female"
    if normalized in {"male", "男性", "m", "0", "0.0"}:
        return "male"
    return None


def _normalize_tnm(value: object) -> str | None:
    normalized = re.sub(r"\s+", "", str(value or "")).upper()
    if normalized.startswith("P"):
        normalized = normalized[1:]
    return normalized or None


def _first(fields: dict[str, list[str]], name: str) -> str | None:
    values = [value for value in fields.get(name, []) if not _is_na(value)]
    return values[0] if values else None


def _is_na(value: str | None) -> bool:
    return value is None or value.strip().lower() in NA_VALUES


def _int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _molecular_events(fields: dict[str, list[str]]) -> list[dict[str, object]]:
    gene_values = fields.get("molecular_tests.gene_symbol", [])
    result_values = fields.get("molecular_tests.test_result", [])
    events = []
    for index, gene in enumerate(gene_values):
        if _is_na(gene):
            continue
        result = result_values[index] if index < len(result_values) else None
        content = f"{gene}: {result}" if result and not _is_na(result) else gene
        events.append(
            {
                "fields": ["molecular_tests.gene_symbol", "molecular_tests.test_result"],
                "content": content,
                "attributes": {"gene": gene, "result": result},
            }
        )
    return events


def _treatment_events(fields: dict[str, list[str]]) -> list[dict[str, object]]:
    treatment_types = fields.get("treatments.treatment_type", [])
    treatment_intents = fields.get("treatments.treatment_intent_type", [])
    treatment_given = fields.get("treatments.treatment_or_therapy", [])
    events = []
    seen = set()
    for index, treatment_type in enumerate(treatment_types):
        if _is_na(treatment_type):
            continue
        intent = treatment_intents[index] if index < len(treatment_intents) else None
        given = treatment_given[index] if index < len(treatment_given) else None
        key = (treatment_type, intent, given)
        if key in seen:
            continue
        seen.add(key)
        content = treatment_type
        if intent and not _is_na(intent):
            content = f"{intent} {content}"
        status = _treatment_status(given)
        note = f"treatment_or_therapy = {given}" if given and not _is_na(given) else None
        events.append(
            {
                "fields": [
                    "treatments.treatment_type",
                    "treatments.treatment_intent_type",
                    "treatments.treatment_or_therapy",
                ],
                "content": content,
                "attributes": {
                    "treatment_type": treatment_type,
                    "intent": intent,
                    "treatment_or_therapy": given,
                    "status": status,
                    "note": note,
                },
            }
        )
    return events


def _treatment_status(value: str | None) -> str:
    if value is None or _is_na(value):
        return "recorded_unknown"
    if value.strip().lower() in {"no", "false", "0"}:
        return "recorded_as_no"
    if value.strip().lower() in {"yes", "true", "1"}:
        return "observed_or_recorded"
    return "recorded_unknown"
