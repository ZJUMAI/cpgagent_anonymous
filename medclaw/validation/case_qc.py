"""Case-level quality control and data completeness audit."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from medclaw.utils import read_yaml

CANCER_ALIASES = {
    "lung_cancer": (
        "lung",
        "luad",
        "nsclc",
        "sclc",
        "bronchus",
        "肺",
        "支气管",
        "adenocarcinoma",
        "squamous",
        "小细胞",
        "非小细胞",
    ),
    "endometrial_cancer": (
        "endometrial",
        "ucec",
        "子宫内膜",
        "子宫体",
    ),
    "nasopharyngeal_cancer": (
        "nasopharyngeal",
        "npc",
        "鼻咽",
    ),
}

LUNG_BIOMARKERS = (
    "EGFR",
    "ALK",
    "ROS1",
    "BRAF",
    "MET_exon14",
    "RET",
    "NTRK",
    "KRAS",
    "PD_L1",
)

LUNG_IMAGING_FIELDS = (
    "CT_chest",
    "PET_CT",
    "brain_MRI",
    "nodal_assessment",
)

UCEC_BIOMARKERS = (
    "MMR",
    "POLE",
    "TP53",
    "ER",
    "PR",
    "p53",
    "MSI",
)

NPC_BIOMARKERS = (
    "EBV_DNA",
    "VCA_IgA",
    "EBNA1_IgA",
    "plasma_EBV",
)

TASK_BY_CLASS = {
    "A_complete_modern": "modern_diagnostic_reasoning",
    "B_usable_incomplete": "modern_missing_information_audit",
    "C_skeleton_only": "synthetic_case_generation_seed",
    "D_exclude": "exclude_or_manual_review",
}


@dataclass
class QCResult:
    case_id: str
    cancer_type: str
    diagnosis_year: int | None
    source_region: str | None
    qc_class: str
    scores: dict[str, int]
    present_fields: list[str]
    missing_critical_fields: list[str]
    recommended_task_type: str
    notes: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CaseRecord:
    """Normalized case payload assembled from one or more source files."""

    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        self.text_parts: list[str] = []
        self.structured: dict[str, Any] = {}
        self.source_files: list[str] = []

    @property
    def text(self) -> str:
        return "\n".join(self.text_parts)

    def add_text(self, text: str, source: str) -> None:
        if text.strip():
            self.text_parts.append(text)
            self.source_files.append(source)

    def add_structured(self, data: dict[str, Any], source: str) -> None:
        self.structured.update(data)
        self.source_files.append(source)


def _normalize(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NA", "N/A", "NOT REPORTED", "UNKNOWN", "--", "'--"}:
        return None
    return text


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(p.lower() in lower for p in patterns)


def detect_cancer_type(text: str, structured: dict[str, Any]) -> str | None:
    clinical = _normalize(structured.get("clinical.cancer") or structured.get("cancer"))
    candidates = " ".join(filter(None, [text, clinical or ""]))
    scores: dict[str, int] = defaultdict(int)
    for cancer_type, aliases in CANCER_ALIASES.items():
        for alias in aliases:
            if alias.lower() in candidates.lower():
                scores[cancer_type] += 1
    if not scores:
        return None
    return max(scores, key=scores.get)


def _extract_age(text: str, structured: dict[str, Any]) -> int | None:
    for key in ("demographic.age_at_index", "age", "age_at_index"):
        raw = _normalize(structured.get(key))
        if raw and raw.isdigit():
            return int(raw)
    match = re.search(r"(\d{1,3})\s*岁", text)
    if match:
        return int(match.group(1))
    match = re.search(r"age[:\s]+(\d{1,3})", text, re.I)
    return int(match.group(1)) if match else None


def _extract_sex(text: str, structured: dict[str, Any]) -> str | None:
    gender = _normalize(structured.get("demographic.gender") or structured.get("gender"))
    if gender:
        return gender
    if re.search(r"\b女性\b|\bfemale\b", text, re.I):
        return "female"
    if re.search(r"\b男性\b|\bmale\b", text, re.I):
        return "male"
    return None


def _extract_diagnosis_year(text: str, structured: dict[str, Any]) -> int | None:
    for key in ("diagnoses.year_of_diagnosis", "year_of_diagnosis", "diagnosis_year"):
        raw = _normalize(structured.get(key))
        if raw and raw.isdigit():
            return int(raw)
    match = re.search(r"诊断年份[：:]\s*(\d{4})", text)
    if match:
        return int(match.group(1))
    match = re.search(r"year_of_diagnosis[:\s]+(\d{4})", text, re.I)
    return int(match.group(1)) if match else None


def _extract_source_region(text: str, structured: dict[str, Any]) -> str | None:
    country = _normalize(
        structured.get("demographic.country_of_residence_at_enrollment")
        or structured.get("country")
    )
    if country:
        if "united states" in country.lower() or country in {"美国", "US", "USA"}:
            return "US"
        if "china" in country.lower() or country in {"中国", "CN"}:
            return "China"
        return country
    if re.search(r"美国|united states|\bUS\b", text, re.I):
        return "US"
    if re.search(r"中国|china|\bCN\b", text, re.I):
        return "China"
    return None


def _extract_histology(text: str, structured: dict[str, Any]) -> str | None:
    diagnosis = _normalize(structured.get("diagnoses.primary_diagnosis"))
    if diagnosis:
        return diagnosis
    patterns = (
        r"诊断[：:]\s*([^。\n]+)",
        r"histology[：:]\s*([^。\n]+)",
        r"(腺癌|鳞癌|小细胞癌|大细胞癌|腺鳞癌|神经内分泌)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            return match.group(1).strip()
    return None


def _extract_primary_site(text: str, structured: dict[str, Any]) -> str | None:
    site = _normalize(
        structured.get("cases.primary_site")
        or structured.get("diagnoses.tissue_or_organ_of_origin")
        or structured.get("primary_site")
    )
    if site:
        return site
    match = re.search(r"原发部位[：:]\s*([^。\n]+)", text)
    return match.group(1).strip() if match else None


def _extract_stage(text: str, structured: dict[str, Any]) -> dict[str, str | None]:
    stage: dict[str, str | None] = {
        "pathologic_T": _normalize(structured.get("diagnoses.ajcc_pathologic_t")),
        "pathologic_N": _normalize(structured.get("diagnoses.ajcc_pathologic_n")),
        "pathologic_M": _normalize(structured.get("diagnoses.ajcc_pathologic_m")),
        "stage_group": _normalize(structured.get("diagnoses.ajcc_pathologic_stage")),
    }
    tnm = re.search(
        r"([Tt]\d+[a-cA-C]?)\s*([Nn]\d+[a-cA-C]?)\s*([Mm]\d+[a-cA-C]?)",
        text,
    )
    if tnm:
        stage["pathologic_T"] = stage["pathologic_T"] or tnm.group(1).upper()
        stage["pathologic_N"] = stage["pathologic_N"] or tnm.group(2).upper()
        stage["pathologic_M"] = stage["pathologic_M"] or tnm.group(3).upper()
    group = re.search(r"(?:AJCC|病理分期|FIGO|TNM)[^。\n]{0,20}([IVX]+|[0-9]+)\s*期", text, re.I)
    if group:
        stage["stage_group"] = stage["stage_group"] or group.group(1)
    return stage


def _extract_smoking(text: str, structured: dict[str, Any]) -> bool:
    if _normalize(structured.get("exposure.smoking_status")):
        return True
    return bool(re.search(r"吸烟|烟草|包年|smoking|tobacco|pack[- ]year", text, re.I))


def _extract_performance_status(text: str) -> bool:
    return bool(re.search(r"\bPS\s*[=:]?\s*[0-4]\b|Karnofsky|ECOG|体能状态", text, re.I))


def _extract_pathology_evidence(text: str) -> bool:
    return bool(
        re.search(
            r"切除|活检|肺叶切除|楔形切除|lobectomy|biopsy|resection|病理报告",
            text,
            re.I,
        )
    )


def _extract_nsclc_sclc(text: str) -> str | None:
    if re.search(r"小细胞|SCLC|small cell", text, re.I):
        return "SCLC"
    if re.search(r"非小细胞|NSCLC|腺癌|鳞癌|大细胞", text, re.I):
        return "NSCLC"
    return None


def _extract_imaging(text: str) -> dict[str, bool]:
    return {
        "CT_chest": bool(re.search(r"胸部\s*CT|胸\s*CT|chest\s*CT|CT\s*chest", text, re.I)),
        "PET_CT": bool(re.search(r"PET[-/ ]?CT|PET扫描", text, re.I)),
        "brain_MRI": bool(
            re.search(r"脑\s*(MRI|磁共振|CT)|brain\s*(MRI|CT)|头颅", text, re.I)
        ),
        "nodal_assessment": bool(
            re.search(r"淋巴结|纵隔|EBUS|EUS|纵隔镜|nodal|mediastin", text, re.I)
        ),
    }


def _extract_biomarker_status(text: str, genes: tuple[str, ...]) -> dict[str, bool]:
    present: dict[str, bool] = {}
    table_gene_aliases = {
        "MET_exon14": ("MET",),
        "NTRK": ("NTRK1", "NTRK2", "NTRK3"),
        "PD_L1": ("PD-L1", "PDL1"),
    }
    for gene in genes:
        patterns = [gene.replace("_", r"[\s-]?")]
        if gene == "MET_exon14":
            patterns.extend([r"MET\s*14", r"MET\s*外显子\s*14", r"MET\s*exon\s*14"])
        if gene == "PD_L1":
            patterns.extend([r"PD[- ]?L1", r"PDL1"])
        if gene == "NTRK":
            patterns.extend([r"NTRK[123]?", r"NTRK"])
        if gene == "MMR":
            patterns.extend([r"MMR", r"dMMR", r"pMMR", r"错配修复"])
        if gene in {"EBV_DNA", "plasma_EBV"}:
            patterns.extend([r"EBV\s*DNA", r"血浆\s*EBV"])
        found = any(re.search(p, text, re.I) for p in patterns)
        if not found:
            aliases = table_gene_aliases.get(gene, (gene,))
            for alias in aliases:
                if re.search(rf"\|\s*{re.escape(alias)}\s*\|", text, re.I):
                    found = True
                    break
        present[gene] = found
    return present


def _extract_treatment_context(text: str, structured: dict[str, Any]) -> bool:
    if _normalize(structured.get("treatments.treatment_type")):
        return True
    return bool(
        re.search(
            r"治疗|化疗|放疗|手术|靶向|免疫|recurrent|metastatic|复发|转移|新辅助|辅助",
            text,
            re.I,
        )
    )


def _extract_outcome(text: str, structured: dict[str, Any]) -> bool:
    if _normalize(structured.get("demographic.vital_status")):
        return True
    return bool(re.search(r"生存|结局|随访|vital|survival|outcome|失访", text, re.I))


def extract_fields(record: CaseRecord) -> dict[str, Any]:
    text = record.text
    structured = record.structured
    stage = _extract_stage(text, structured)
    imaging = _extract_imaging(text)
    cancer_type = detect_cancer_type(text, structured) or "unknown"

    biomarker_genes = LUNG_BIOMARKERS
    if cancer_type == "endometrial_cancer":
        biomarker_genes = UCEC_BIOMARKERS
    elif cancer_type == "nasopharyngeal_cancer":
        biomarker_genes = NPC_BIOMARKERS

    biomarkers = _extract_biomarker_status(text, biomarker_genes)
    histology = _extract_histology(text, structured)
    primary_site = _extract_primary_site(text, structured)

    fields: dict[str, Any] = {
        "cancer_type": cancer_type,
        "age": _extract_age(text, structured),
        "sex": _extract_sex(text, structured),
        "diagnosis_year": _extract_diagnosis_year(text, structured),
        "source_region": _extract_source_region(text, structured),
        "primary_site": primary_site,
        "primary_diagnosis": _normalize(structured.get("diagnoses.primary_diagnosis")),
        "histology": histology,
        "primary_site_lung": _contains_any(
            " ".join(filter(None, [primary_site or "", text])),
            ("lung", "bronchus", "肺", "支气管"),
        ),
        "nsclc_sclc": _extract_nsclc_sclc(text),
        "smoking_history": _extract_smoking(text, structured),
        "performance_status": _extract_performance_status(text),
        "pathology_evidence": _extract_pathology_evidence(text),
        "treatment_context": _extract_treatment_context(text, structured),
        "outcome": _extract_outcome(text, structured),
        "case_id": record.case_id,
        **stage,
        **imaging,
        **biomarkers,
    }
    return fields


def detect_conflicts(fields: dict[str, Any], text: str) -> list[str]:
    conflicts: list[str] = []
    cancer_type = fields.get("cancer_type")
    if cancer_type == "lung_cancer" and fields.get("primary_site") and not fields.get("primary_site_lung"):
        conflicts.append("Primary site does not appear to be lung/bronchus.")
    if fields.get("nsclc_sclc") == "SCLC" and fields.get("histology") and "腺癌" in fields["histology"]:
        conflicts.append("Histology suggests adenocarcinoma but NSCLC/SCLC label is SCLC.")
    if re.search(r"原发部位不清|癌种不明|diagnosis conflict", text, re.I):
        conflicts.append("Report explicitly flags unclear diagnosis or primary site.")
    t, n, m = fields.get("pathologic_T"), fields.get("pathologic_N"), fields.get("pathologic_M")
    stage_group = fields.get("stage_group")
    if t and n and m and stage_group:
        if stage_group.upper() in {"IV", "4"} and m.upper().startswith("M0"):
            conflicts.append("Stage group suggests metastatic disease but M is M0.")
    return conflicts


def score_lung_case(fields: dict[str, Any], conflicts: list[str]) -> tuple[dict[str, int], list[str], list[str], list[str]]:
    present: list[str] = []
    missing: list[str] = []
    notes: list[str] = []

    def mark(name: str, ok: bool, critical: bool = True) -> None:
        if ok:
            present.append(name)
        elif critical:
            missing.append(name)

    mark("case_id", bool(fields.get("case_id")), critical=False)
    mark("primary_site", bool(fields.get("primary_site_lung")))
    mark("primary_diagnosis", bool(fields.get("primary_diagnosis") or fields.get("histology")))
    mark("histology", bool(fields.get("histology")))

    identity = 0
    if fields.get("primary_site_lung"):
        identity += 1
    if fields.get("primary_diagnosis") or fields.get("histology"):
        identity += 1

    mark("nsclc_sclc", bool(fields.get("nsclc_sclc")))
    mark("pathology_evidence", bool(fields.get("pathology_evidence")))
    pathology = int(bool(fields.get("nsclc_sclc"))) + int(bool(fields.get("pathology_evidence")))

    for key in ("pathologic_T", "pathologic_N", "pathologic_M", "stage_group"):
        mark(key, bool(fields.get(key)))
    stage_parts = sum(bool(fields.get(k)) for k in ("pathologic_T", "pathologic_N", "pathologic_M", "stage_group"))
    stage = min(3, stage_parts)

    for key in LUNG_IMAGING_FIELDS:
        mark(key, bool(fields.get(key)), critical=key in {"CT_chest", "PET_CT", "brain_MRI"})
    imaging = min(2, sum(bool(fields.get(k)) for k in LUNG_IMAGING_FIELDS))

    biomarker_present = [g for g in LUNG_BIOMARKERS if fields.get(g)]
    for gene in LUNG_BIOMARKERS:
        mark(gene, bool(fields.get(gene)))
    biomarker = min(3, len(biomarker_present))

    mark("age", bool(fields.get("age")), critical=False)
    mark("sex", bool(fields.get("sex")), critical=False)
    mark("smoking_history", bool(fields.get("smoking_history")), critical=False)
    mark("performance_status", bool(fields.get("performance_status")), critical=False)
    context = int(
        bool(fields.get("smoking_history"))
        or bool(fields.get("performance_status"))
        or bool(fields.get("treatment_context"))
    )

    consistency = 3
    if conflicts:
        consistency = max(0, 3 - len(conflicts))
        notes.extend(conflicts)

    if biomarker == 0:
        notes.append(
            "Molecular testing is absent or incomplete; do not use as complete modern diagnostic case unless augmented."
        )
    elif biomarker < 3 and not fields.get("PD_L1"):
        notes.append("Partial molecular panel available; clinical PD-L1/IHC status may still be missing.")

    if stage >= 2 and imaging <= 1:
        notes.append("Diagnosis and pathologic stage are usable; modern imaging workup may be incomplete.")

    scores = {
        "identity_score": identity,
        "pathology_score": pathology,
        "stage_score": stage,
        "imaging_score": imaging,
        "biomarker_score": biomarker,
        "context_score": context,
        "consistency_score": consistency,
    }
    return scores, present, missing, notes


def score_endometrial_case(
    fields: dict[str, Any], conflicts: list[str]
) -> tuple[dict[str, int], list[str], list[str], list[str]]:
    present: list[str] = []
    missing: list[str] = []
    notes: list[str] = []

    def mark(name: str, ok: bool, critical: bool = True) -> None:
        if ok:
            present.append(name)
        elif critical:
            missing.append(name)

    mark("primary_site", _contains_any(fields.get("primary_site") or "", ("子宫", "endometr")))
    mark("histology", bool(fields.get("histology")))
    identity = int(bool(fields.get("histology"))) + int(
        _contains_any(fields.get("primary_site") or "", ("子宫", "endometr"))
    )

    mark("pathology_evidence", bool(fields.get("pathology_evidence")))
    figo = bool(fields.get("stage_group")) or bool(re.search(r"FIGO|I+\s*期", fields.get("histology") or ""))
    mark("stage_group", figo)
    pathology = int(bool(fields.get("pathology_evidence"))) + int(figo)

    stage = min(3, sum(bool(fields.get(k)) for k in ("pathologic_T", "pathologic_N", "pathologic_M", "stage_group")))
    imaging = min(
        2,
        int(bool(fields.get("CT_chest"))) + int(_contains_any(fields.get("primary_site") or "", ("MRI", "磁共振"))),
    )
    biomarker = min(3, sum(bool(fields.get(g)) for g in UCEC_BIOMARKERS))
    for gene in UCEC_BIOMARKERS:
        mark(gene, bool(fields.get(gene)))
    context = int(bool(fields.get("age")) or bool(fields.get("sex")) or bool(fields.get("treatment_context")))
    consistency = max(0, 3 - len(conflicts))
    notes.extend(conflicts)
    scores = {
        "identity_score": identity,
        "pathology_score": pathology,
        "stage_score": stage,
        "imaging_score": imaging,
        "biomarker_score": biomarker,
        "context_score": context,
        "consistency_score": consistency,
    }
    return scores, present, missing, notes


def score_npc_case(
    fields: dict[str, Any], conflicts: list[str]
) -> tuple[dict[str, int], list[str], list[str], list[str]]:
    present: list[str] = []
    missing: list[str] = []
    notes: list[str] = list(conflicts)

    def mark(name: str, ok: bool, critical: bool = True) -> None:
        if ok:
            present.append(name)
        elif critical:
            missing.append(name)

    npc_site = _contains_any(fields.get("primary_site") or "", ("鼻咽", "nasopharyn"))
    mark("primary_site", npc_site)
    mark("histology", bool(fields.get("histology")))
    identity = int(npc_site) + int(bool(fields.get("histology")))

    mark("pathology_evidence", bool(fields.get("pathology_evidence")))
    pathology = int(bool(fields.get("pathology_evidence"))) + int(bool(fields.get("histology")))
    stage = min(3, sum(bool(fields.get(k)) for k in ("pathologic_T", "pathologic_N", "pathologic_M", "stage_group")))
    imaging = min(2, sum(bool(fields.get(k)) for k in LUNG_IMAGING_FIELDS))
    biomarker = min(3, sum(bool(fields.get(g)) for g in NPC_BIOMARKERS))
    for gene in NPC_BIOMARKERS:
        mark(gene, bool(fields.get(gene)))
    context = int(bool(fields.get("age")) or bool(fields.get("sex")) or bool(fields.get("treatment_context")))
    consistency = max(0, 3 - len(conflicts))
    scores = {
        "identity_score": identity,
        "pathology_score": pathology,
        "stage_score": stage,
        "imaging_score": imaging,
        "biomarker_score": biomarker,
        "context_score": context,
        "consistency_score": consistency,
    }
    return scores, present, missing, notes


def classify_case(fields: dict[str, Any], scores: dict[str, int], conflicts: list[str]) -> str:
    if fields.get("cancer_type") == "unknown":
        return "D_exclude"
    if scores["identity_score"] == 0:
        return "D_exclude"
    if scores["consistency_score"] <= 1 and conflicts:
        return "D_exclude"
    if fields.get("cancer_type") == "lung_cancer" and not fields.get("primary_site_lung"):
        return "D_exclude"

    total_core = (
        scores["identity_score"]
        + scores["pathology_score"]
        + scores["stage_score"]
        + scores["consistency_score"]
    )

    if scores["pathology_score"] <= 1 and scores["stage_score"] <= 1 and total_core <= 4:
        if scores["identity_score"] >= 1:
            return "C_skeleton_only"

    if (
        scores["identity_score"] >= 2
        and scores["pathology_score"] >= 2
        and scores["stage_score"] >= 2
        and scores["consistency_score"] >= 2
        and scores["biomarker_score"] >= 2
        and scores["imaging_score"] >= 1
    ):
        modern_imaging_complete = fields.get("CT_chest") or (
            fields.get("PET_CT") and fields.get("brain_MRI")
        )
        modern_molecular_complete = fields.get("PD_L1")
        if modern_imaging_complete and modern_molecular_complete:
            return "A_complete_modern"
        return "B_usable_incomplete"

    if scores["identity_score"] >= 1 and scores["pathology_score"] >= 1:
        return "B_usable_incomplete"

    return "D_exclude"


def score_case(record: CaseRecord) -> QCResult:
    fields = extract_fields(record)
    conflicts = detect_conflicts(fields, record.text)
    cancer_type = fields.get("cancer_type") or "unknown"

    if cancer_type == "endometrial_cancer":
        scores, present, missing, score_notes = score_endometrial_case(fields, conflicts)
    elif cancer_type == "nasopharyngeal_cancer":
        scores, present, missing, score_notes = score_npc_case(fields, conflicts)
    else:
        scores, present, missing, score_notes = score_lung_case(fields, conflicts)

    qc_class = classify_case(fields, scores, conflicts)
    notes = list(score_notes)
    if qc_class == "B_usable_incomplete" and scores["stage_score"] >= 2:
        if "Diagnosis and pathologic stage are usable." not in notes:
            notes.append("Diagnosis and pathologic stage are usable.")
    if qc_class == "B_usable_incomplete":
        if not fields.get("PD_L1"):
            notes.append(
                "Clinical PD-L1/IHC status missing; research-grade molecular data alone is not enough for a complete modern case."
            )
        if not fields.get("CT_chest") and not fields.get("PET_CT"):
            notes.append("Modern imaging workup (CT/PET/brain MRI) is incomplete in the chart.")
    if scores["biomarker_score"] == 0:
        notes.append(
            "Molecular testing is absent; do not use as complete modern diagnostic case unless augmented."
        )
    if not record.text_parts and not fields.get("primary_diagnosis"):
        notes.append("No usable case narrative or structured report found.")
        qc_class = "D_exclude"

    deduped_notes: list[str] = []
    seen_notes: set[str] = set()
    for note in notes:
        if note not in seen_notes:
            deduped_notes.append(note)
            seen_notes.add(note)
    notes = deduped_notes

    return QCResult(
        case_id=record.case_id,
        cancer_type=cancer_type,
        diagnosis_year=fields.get("diagnosis_year"),
        source_region=fields.get("source_region"),
        qc_class=qc_class,
        scores=scores,
        present_fields=sorted(set(present)),
        missing_critical_fields=sorted(set(missing)),
        recommended_task_type=TASK_BY_CLASS[qc_class],
        notes=notes,
        conflicts=conflicts,
        source_files=sorted(set(record.source_files)),
    )


class CaseQCRunner:
    """Discover cases and run completeness audit."""

    def __init__(
        self,
        *,
        reports_root: Path | None = None,
        molecular_root: Path | None = None,
        clinical_tsv: Path | None = None,
        cases_root: Path | None = None,
    ) -> None:
        self.reports_root = Path(reports_root).resolve() if reports_root else None
        self.molecular_root = Path(molecular_root).resolve() if molecular_root else None
        self.clinical_tsv = Path(clinical_tsv).resolve() if clinical_tsv else None
        self.cases_root = Path(cases_root).resolve() if cases_root else None
        self._clinical_index: dict[str, dict[str, Any]] | None = None

    def _load_clinical_index(self) -> dict[str, dict[str, Any]]:
        if self._clinical_index is not None:
            return self._clinical_index
        index: dict[str, dict[str, Any]] = {}
        if self.clinical_tsv and self.clinical_tsv.is_file():
            with self.clinical_tsv.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    case_id = _normalize(row.get("cases.submitter_id"))
                    if case_id:
                        index[case_id] = row
        self._clinical_index = index
        return index

    def discover_case_ids(self) -> list[str]:
        case_ids: set[str] = set()
        if self.reports_root and self.reports_root.is_dir():
            for child in self.reports_root.iterdir():
                if child.is_dir():
                    case_ids.add(child.name)
        if self.molecular_root and self.molecular_root.is_dir():
            for child in self.molecular_root.iterdir():
                if child.is_dir():
                    case_ids.add(child.name)
        if self.cases_root and self.cases_root.is_dir():
            for child in self.cases_root.iterdir():
                if child.is_dir() and (child / "case.yaml").is_file():
                    case_ids.add(child.name)
        return sorted(case_ids)

    def load_case(self, case_id: str) -> CaseRecord:
        record = CaseRecord(case_id)
        if self.reports_root:
            report_dir = self.reports_root / case_id
            for name in (f"{case_id}_report_zh.md", f"{case_id}_report_en.md", f"{case_id}_full_report_zh.md"):
                path = report_dir / name
                if path.is_file():
                    record.add_text(path.read_text(encoding="utf-8"), str(path))
                    break
            if not record.text_parts:
                for path in sorted(report_dir.glob("*.md")):
                    record.add_text(path.read_text(encoding="utf-8"), str(path))

        if self.molecular_root:
            mol_dir = self.molecular_root / case_id
            for path in sorted(mol_dir.glob("*molecular*.md")):
                record.add_text(path.read_text(encoding="utf-8"), str(path))

        clinical = self._load_clinical_index().get(case_id)
        if clinical:
            record.add_structured(clinical, str(self.clinical_tsv))

        if self.cases_root:
            yaml_path = self.cases_root / case_id / "case.yaml"
            if yaml_path.is_file():
                data = read_yaml(yaml_path)
                if isinstance(data, dict):
                    flat: dict[str, Any] = {"case_id": case_id}
                    clinical_block = data.get("clinical") or {}
                    if isinstance(clinical_block, dict):
                        for key, value in clinical_block.items():
                            flat[f"clinical.{key}"] = value
                    record.add_structured(flat, str(yaml_path))
                    record.add_text(yaml_path.read_text(encoding="utf-8"), str(yaml_path))

        return record

    def run(self, case_ids: list[str] | None = None) -> list[QCResult]:
        ids = case_ids or self.discover_case_ids()
        return [score_case(self.load_case(case_id)) for case_id in ids]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary(results: list[QCResult]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_class = Counter(r.qc_class for r in results)
    by_cancer: dict[str, Counter[str]] = defaultdict(Counter)
    missing_counter: Counter[str] = Counter()
    conflict_counter: Counter[str] = Counter()

    for result in results:
        by_cancer[result.cancer_type][result.qc_class] += 1
        for field_name in result.missing_critical_fields:
            missing_counter[field_name] += 1
        for conflict in result.conflicts:
            conflict_counter[conflict] += 1

    summary_rows = [
        {"metric": "total_cases", "value": len(results)},
        {"metric": "A_complete_modern", "value": by_class["A_complete_modern"]},
        {"metric": "B_usable_incomplete", "value": by_class["B_usable_incomplete"]},
        {"metric": "C_skeleton_only", "value": by_class["C_skeleton_only"]},
        {"metric": "D_exclude", "value": by_class["D_exclude"]},
    ]
    for field_name, count in missing_counter.most_common(20):
        summary_rows.append({"metric": f"missing::{field_name}", "value": count})
    for conflict, count in conflict_counter.most_common(10):
        summary_rows.append({"metric": f"conflict::{conflict}", "value": count})

    cancer_rows: list[dict[str, Any]] = []
    for cancer_type in sorted(by_cancer):
        counts = by_cancer[cancer_type]
        cancer_rows.append(
            {
                "cancer_type": cancer_type,
                "total": sum(counts.values()),
                "A_complete_modern": counts["A_complete_modern"],
                "B_usable_incomplete": counts["B_usable_incomplete"],
                "C_skeleton_only": counts["C_skeleton_only"],
                "D_exclude": counts["D_exclude"],
            }
        )
    return summary_rows, cancer_rows


def write_qc_outputs(results: list[QCResult], output_dir: Path) -> dict[str, Path]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [r.to_dict() for r in results]
    paths = {
        "qc_report": output_dir / "qc_report.jsonl",
        "candidate_complete": output_dir / "candidate_complete_cases.jsonl",
        "candidate_incomplete": output_dir / "candidate_incomplete_cases.jsonl",
        "excluded": output_dir / "excluded_cases.jsonl",
    }
    _write_jsonl(paths["qc_report"], rows)
    _write_jsonl(
        paths["candidate_complete"],
        [r for r in rows if r["qc_class"] == "A_complete_modern"],
    )
    _write_jsonl(
        paths["candidate_incomplete"],
        [r for r in rows if r["qc_class"] == "B_usable_incomplete"],
    )
    _write_jsonl(
        paths["excluded"],
        [r for r in rows if r["qc_class"] in {"C_skeleton_only", "D_exclude"}],
    )

    summary_rows, cancer_rows = build_summary(results)
    summary_path = output_dir / "qc_summary.csv"
    cancer_path = output_dir / "qc_by_cancer_type.csv"
    _write_csv(summary_path, ["metric", "value"], summary_rows)
    _write_csv(
        cancer_path,
        [
            "cancer_type",
            "total",
            "A_complete_modern",
            "B_usable_incomplete",
            "C_skeleton_only",
            "D_exclude",
        ],
        cancer_rows,
    )
    paths["qc_summary"] = summary_path
    paths["qc_by_cancer_type"] = cancer_path
    return paths


def run_case_qc(
    *,
    output_dir: Path,
    reports_root: Path | None = None,
    molecular_root: Path | None = None,
    clinical_tsv: Path | None = None,
    cases_root: Path | None = None,
    case_ids: list[str] | None = None,
) -> list[QCResult]:
    runner = CaseQCRunner(
        reports_root=reports_root,
        molecular_root=molecular_root,
        clinical_tsv=clinical_tsv,
        cases_root=cases_root,
    )
    results = runner.run(case_ids)
    write_qc_outputs(results, output_dir)
    return results
