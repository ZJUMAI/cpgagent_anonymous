"""Build a lightweight benchmark case package from a TCGA report."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from medclaw.llm.factory import create_llm_client, load_config_from_env, resolve_provider
from medclaw.trajectory import build_guideline_trajectory
from medclaw.utils import read_json, read_yaml, write_json

from medclaw_benchmark.case_paths import (
    find_gold_trajectory,
    find_report_path,
    find_rubric_path,
    gold_trajectory_write_path,
    should_generate_gold_trajectory,
)
from medclaw_benchmark.io_utils import write_text
from medclaw_benchmark.report_extractor import LLMReportExtractor


MISSING = "not_available"
BIOMARKERS = (
    "EGFR",
    "ALK",
    "KRAS",
    "BRAF",
    "ROS1",
    "RET",
    "MET",
    "ERBB2",
    "NTRK",
    "NTRK1",
    "NTRK2",
    "NTRK3",
    "TP53",
    "STK11",
    "KEAP1",
    "SMARCA4",
    "RB1",
    "CDKN2A",
    "PIK3CA",
    "NF1",
    "ATM",
    "PD-L1",
    "TMB",
)


@dataclass(frozen=True)
class CaseBuilder:
    """Convert a Markdown case report into benchmark simulator files."""

    case_dir: str | Path
    report_path: str | Path | None = None
    rubric_path: str | Path | None = None
    llm_client: Any | None = None
    llm_config: Any | None = None
    provider: str | None = None
    qwen_config: Any | None = None

    def build(self) -> None:
        """Generate hidden state, release policy, and modality files."""

        case_dir = self._case_dir()
        case_dir.mkdir(parents=True, exist_ok=True)
        report = self.load_report()
        rubric = self.load_rubric()
        case_yaml = self._load_case_yaml()
        extraction = self.extract_report_structures(report)

        write_json(case_dir / "report_extraction.json", _extraction_audit_payload(extraction))
        write_json(
            case_dir / "hidden_state.json",
            self.build_hidden_state(report, rubric, extraction),
        )
        write_json(case_dir / "release_policy.json", self.build_release_policy())
        self.build_clinical_files(report, extraction)
        self.build_pathology_files(report, case_yaml)
        self.build_radiology_files(case_yaml)
        self.build_molecular_files(report, extraction)
        self.build_guideline_files()
        self.build_trajectory_file()

    def load_report(self) -> str:
        path = self._report_path()
        return path.read_text(encoding="utf-8")

    def load_rubric(self) -> dict[str, Any] | None:
        path = self._rubric_path()
        if path is None or not path.is_file():
            return None
        data = read_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"Rubric must contain a JSON object: {path}")
        return data

    def extract_report_structures(self, report: str | None = None) -> dict[str, Any]:
        """Extract report facts with Qwen when configured, otherwise use fallback."""

        mode = os.environ.get("MEDCLAW_CASE_BUILDER_LLM_MODE", "auto").strip().lower()
        if mode not in {"auto", "required", "disabled"}:
            raise ValueError(
                "MEDCLAW_CASE_BUILDER_LLM_MODE must be auto, required, or disabled."
            )
        force = _env_flag("MEDCLAW_CASE_BUILDER_FORCE_LLM_EXTRACTION")
        if not force:
            cached = self._read_cached_report_extraction()
            if cached and _can_reuse_cached_extraction(cached, mode):
                return cached
        if mode == "disabled":
            return {
                "status": "disabled",
                "method": "fallback",
                "warnings": ["LLM report extraction disabled by environment."],
            }

        report_text = report if report is not None else self.load_report()
        try:
            client = self.llm_client
            if client is None:
                provider = resolve_provider(
                    self.provider,
                    env_key="MEDCLAW_CASE_BUILDER_PROVIDER",
                )
                config = self.llm_config or self.qwen_config or load_config_from_env(provider)
                client = create_llm_client(provider, config=config)
            return LLMReportExtractor(client).extract(
                case_id=self.case_id,
                report_text=report_text,
            )
        except Exception as exc:
            if mode == "required":
                raise
            return {
                "status": "fallback",
                "method": "fallback",
                "warnings": [
                    "LLM report extraction was not used; falling back to local rules.",
                    f"{type(exc).__name__}: {exc}",
                ],
            }

    def _read_cached_report_extraction(self) -> dict[str, Any] | None:
        case_dir = self._case_dir()
        cached = _read_optional_json_object(case_dir / "report_extraction.json")
        if cached:
            result = dict(cached)
            result.setdefault("status", "success")
            result.setdefault("method", result.get("extraction_method", "llm"))
            result["cache_status"] = "hit_report_extraction"
            result.setdefault("warnings", [])
            result["warnings"] = [
                *_string_list(result.get("warnings")),
                "Reused cached report_extraction.json; set "
                "MEDCLAW_CASE_BUILDER_FORCE_LLM_EXTRACTION=1 to refresh.",
            ]
            return result

        generated = _existing_generated_extraction(case_dir)
        if generated:
            generated["cache_status"] = "hit_generated_case_json"
            generated["warnings"] = [
                *_string_list(generated.get("warnings")),
                "Reused existing generated clinical/molecular JSON; set "
                "MEDCLAW_CASE_BUILDER_FORCE_LLM_EXTRACTION=1 to refresh.",
            ]
            return generated
        return None

    def build_hidden_state(
        self,
        report: str | None = None,
        rubric: dict[str, Any] | None = None,
        extraction: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build hidden case metadata without leaking late-stage outcomes."""

        report_text = report if report is not None else self.load_report()
        clinical = _extracted_section(extraction, "clinical")
        case_yaml = self._load_case_yaml()
        yaml_clinical = (
            case_yaml.get("clinical", {}) if isinstance(case_yaml, Mapping) else {}
        )
        identity = _case_cancer_identity(
            clinical.get("project_id"),
            clinical.get("primary_diagnosis"),
            yaml_clinical.get("cancer") if isinstance(yaml_clinical, Mapping) else None,
            yaml_clinical.get("guideline") if isinstance(yaml_clinical, Mapping) else None,
            report_text,
        )
        return {
            "case_id": self.case_id,
            "project_id": _first_present(
                clinical.get("project_id"),
                _first_match(report_text, r"\*\*项目\*\*:\s*([^\n]+)"),
                identity["project_id"],
            ),
            "cancer_type": identity["cancer_type"],
            "report_language": "zh",
            "source_report_path": self._portable_path(self._report_path()),
            "source_rubric_path": self._portable_path(self._rubric_path())
            if self._rubric_path()
            else MISSING,
            "initial_prompt": {
                "age": _first_present(clinical.get("age"), _extract_age(report_text)),
                "sex": _first_present(clinical.get("sex"), _extract_sex(report_text)),
                "race": _first_present(clinical.get("race"), _field_value(report_text, "种族")),
                "ethnicity": _first_present(
                    clinical.get("ethnicity"),
                    _field_value(report_text, "民族"),
                ),
                "chief_problem": _first_present(
                    clinical.get("primary_diagnosis"),
                    yaml_clinical.get("cancer")
                    if isinstance(yaml_clinical, Mapping)
                    else None,
                    identity["chief_problem"],
                ),
                "known_status": "initial TCGA case report available but hidden from evaluated agent",
            },
            "modalities": {
                "clinical_table": True,
                "pathology_report": True,
                "pathology_slide_metadata": True,
                "wsi": "available_if_manifest_exists",
                "pathology_roi_skill": identity["pathology_roi_skill"],
                "radiology": "available_if_manifest_exists",
                "ct_roi_skill": identity["radiology_roi_skill"],
                "molecular": True,
                "follow_up": True,
                "guideline": True,
            },
        }

    def build_release_policy(self) -> dict[str, list[str]]:
        """Return the fixed MVP release policy."""

        return {
            "diagnosis_phase": [
                "demographics",
                "initial_diagnosis",
                "primary_site",
            ],
            "pathology_text_phase": [
                "pathology_report",
                "slide_metadata",
            ],
            "pathology_image_phase": [
                "wsi_manifest",
                "wsi_roi_artifacts",
                "conch_patch_roi_results",
            ],
            "radiology_phase": [
                "ct_manifest",
                "ct_roi_artifacts",
                "radiology_roi_results",
            ],
            "staging_phase": [
                "tnm_stage",
                "ajcc_stage",
                "tumor_features",
                "lymph_node_status",
                "metastasis_status",
                "radiology_findings_if_available",
            ],
            "treatment_phase": [
                "treatment_records",
                "molecular_results",
                "guideline_snippets",
            ],
            "progression_phase": [
                "progression_status",
                "recurrence_status",
                "follow_up_disease_status",
            ],
            "outcome_phase": [
                "vital_status",
                "last_follow_up_time",
                "survival_status",
            ],
        }

    def build_trajectory_file(self) -> Path | None:
        """Write gold trajectory under evaluation/{case_id}_trajectory.json.

        Existing gold files are left untouched. UCEC / NPC packs use the same
        path but may still be empty; nothing is generated for those cases.
        """

        case_dir = self._case_dir()
        existing = find_gold_trajectory(case_dir)
        if existing is not None:
            return existing
        if not should_generate_gold_trajectory(case_dir):
            return None
        output_path = gold_trajectory_write_path(case_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        case_yaml = self._load_case_yaml()
        clinical = case_yaml.get("clinical", {}) if isinstance(case_yaml, dict) else {}
        guideline_version = (
            clinical.get("guideline_version")
            if isinstance(clinical, Mapping)
            else None
        ) or "2010"
        guideline_id = (
            clinical.get("guideline_id")
            if isinstance(clinical, Mapping)
            else None
        ) or "NSCLC_2010"
        decision_date = (
            clinical.get("decision_date")
            if isinstance(clinical, Mapping)
            else None
        ) or "2010-12-31"
        diagnosis_year = (
            clinical.get("diagnosis_year")
            if isinstance(clinical, Mapping)
            else None
        )
        build_guideline_trajectory(
            self.case_id,
            self._report_path(),
            guideline_id=str(guideline_id),
            guideline_version=str(guideline_version),
            decision_date=str(decision_date),
            diagnosis_year=diagnosis_year if isinstance(diagnosis_year, int) else None,
            clinical_path=case_dir / "clinical" / "clinical.json",
            output_path=output_path,
        )
        return output_path

    def build_clinical_files(
        self,
        report: str | None = None,
        extraction: Mapping[str, Any] | None = None,
    ) -> None:
        report_text = report if report is not None else self.load_report()
        case_yaml = self._load_case_yaml()
        yaml_clinical = (
            case_yaml.get("clinical", {}) if isinstance(case_yaml, Mapping) else {}
        )
        extracted_clinical = _extracted_section(extraction, "clinical")
        identity = _case_cancer_identity(
            extracted_clinical.get("project_id"),
            extracted_clinical.get("primary_diagnosis"),
            yaml_clinical.get("cancer") if isinstance(yaml_clinical, Mapping) else None,
            report_text,
        )
        is_ucec = identity["cancer_type"] == "UCEC"
        clinical_dir = self._case_dir() / "clinical"
        clinical = {
            "case_id": self.case_id,
            "project_id": _first_match(report_text, r"\*\*项目\*\*:\s*([^\n]+)")
            or identity["project_id"],
            "age": _extract_age(report_text) or MISSING,
            "sex": _extract_sex(report_text),
            "race": _field_value(report_text, "种族"),
            "ethnicity": _field_value(report_text, "民族"),
            "country": _field_value(report_text, "居住国"),
            "primary_site": _first_match(report_text, r"原发部位为([^，。]+)")
            or ("uterine corpus" if is_ucec else "bronchus and lung"),
            "disease_type": _first_match(report_text, r"疾病类型为([^，。]+)")
            or ("endometrial carcinoma" if is_ucec else "adenomas and adenocarcinomas"),
            "primary_diagnosis": identity["chief_problem"],
            "diagnosis_age_days": _first_match(report_text, r"\*\*年龄 at 诊断\*\*:\s*(\d+)天") or MISSING,
            "tumor_classification": _field_value(report_text, "肿瘤分类"),
            "diagnosis_is_primary_disease": _field_value(report_text, "是否为原发疾病"),
            "prior_treatment": _field_value(report_text, "诊断前曾接受治疗"),
            "smoking_status": _field_value(report_text, "吸烟状态"),
            "pack_years": _field_value(report_text, "吸烟包年数"),
            "family_history": "missing",
            "pathologic_t_stage": MISSING,
            "pathologic_n_stage": MISSING,
            "pathologic_m_stage": MISSING,
            "pathologic_stage": MISSING,
            "vital_status_at_last_follow_up": "alive"
            if "末次随访时生存状态为存活" in report_text
            else MISSING,
            "lost_to_follow_up": False if "未失访" in report_text else MISSING,
            "source": self._report_path().name,
        }
        clinical.update(_clean_mapping(extracted_clinical))
        clinical.update(
            {
                "case_id": self.case_id,
                "source": self._report_path().name,
                "extraction_method": _extraction_method(extraction),
                "extraction_warnings": _extraction_warnings(extraction),
            }
        )
        write_json(clinical_dir / "clinical.json", clinical)
        write_json(
            clinical_dir / "treatment.json",
            _with_extraction_metadata(
                _merge_if_present(
                    self._treatment(report_text),
                    _extracted_section(extraction, "treatment"),
                ),
                extraction,
            ),
        )
        write_json(
            clinical_dir / "follow_up.json",
            _with_extraction_metadata(
                _merge_if_present(
                    self._follow_up(report_text),
                    _extracted_section(extraction, "follow_up"),
                ),
                extraction,
            ),
        )

    def build_pathology_files(
        self,
        report: str | None = None,
        case_yaml: dict[str, Any] | None = None,
    ) -> None:
        report_text = report if report is not None else self.load_report()
        yaml_data = case_yaml if case_yaml is not None else self._load_case_yaml()
        pathology_dir = self._case_dir() / "pathology"
        pathology_text = _section(report_text, "病理报告原文")
        if not pathology_text:
            pathology_text = (
                "Pathology report text is not available in the source case report. "
                "Structured pathology fields may still be available from TCGA tables."
            )
            report_available = False
        else:
            report_available = True
        write_text(pathology_dir / "pathology_report.txt", pathology_text.strip() + "\n")
        write_json(
            pathology_dir / "pathology_report_metadata.json",
            {
                "case_id": self.case_id,
                "report_text_available": report_available,
                "source": self._report_path().name,
            },
        )
        write_json(pathology_dir / "slide_metadata.json", self._slide_metadata(report_text))
        write_json(pathology_dir / "wsi_manifest.json", self._wsi_manifest(yaml_data))
        write_json(pathology_dir / "wsi_roi_manifest.json", self._wsi_roi_manifest(yaml_data))

    def build_radiology_files(self, case_yaml: dict[str, Any] | None = None) -> None:
        yaml_data = case_yaml if case_yaml is not None else self._load_case_yaml()
        radiology_dir = self._case_dir() / "radiology"
        manifest = self._ct_manifest(yaml_data)
        write_json(radiology_dir / "ct_manifest.json", manifest)
        write_json(
            radiology_dir / "ct_roi_manifest.json",
            {
                "case_id": self.case_id,
                "available": manifest["available"],
                "skill_name": "radiology.lung_tumor_roi",
                "source_ct_path": manifest["series"][0]["ct_path"],
                "roi_artifacts": [],
            },
        )

    def build_molecular_files(
        self,
        report: str | None = None,
        extraction: Mapping[str, Any] | None = None,
    ) -> None:
        molecular = _extracted_section(extraction, "molecular")
        biomarker_details = _biomarker_details(molecular.get("biomarkers"))
        data = {
            name: _biomarker_summary(biomarker_details.get(name))
            for name in BIOMARKERS
        }
        data.update(
            {
                "case_id": self.case_id,
                "source": self._report_path().name,
                "extraction_method": _extraction_method(extraction),
                "extraction_warnings": _extraction_warnings(extraction),
                "summary": _first_present(molecular.get("summary")),
                "molecular_subtype": _first_present(molecular.get("molecular_subtype")),
                "copy_number_summary": _first_present(molecular.get("copy_number_summary")),
                "high_amplification_genes": _string_list(
                    molecular.get("high_amplification_genes")
                ),
                "biomarker_details": biomarker_details,
            }
        )
        write_json(self._case_dir() / "molecular" / "biomarkers.json", data)

    def build_guideline_files(self) -> None:
        write_json(
            self._case_dir() / "guideline" / "relevant_nodes.json",
            {
                "case_id": self.case_id,
                "nodes": [
                    {
                        "node_id": "NSCLC_INITIAL_WORKUP",
                        "setting": "initial_workup",
                        "recommendation": (
                            "Confirm histology, complete staging imaging, TNM stage, "
                            "performance status, pulmonary function, comorbidities, smoking "
                            "history, molecular testing, and PD-L1 before final treatment planning."
                        ),
                    },
                    {
                        "node_id": "NSCLC_STAGING_REVIEW",
                        "setting": "staging",
                        "recommendation": (
                            "Review tumor size, nodal status, metastasis status, pleural "
                            "invasion, satellite nodules, margin status, and AJCC version."
                        ),
                    },
                    {
                        "node_id": "NSCLC_BIOMARKER_WARNING",
                        "setting": "biomarker_dependent_treatment",
                        "recommendation": (
                            "Do not recommend targeted therapy or immunotherapy as definite "
                            "treatment without confirmed biomarker or PD-L1 results."
                        ),
                    },
                    {
                        "node_id": "NSCLC_TREATMENT_PLANNING",
                        "setting": "treatment_planning",
                        "recommendation": (
                            "For lung adenocarcinoma, treatment planning should be conditioned "
                            "on stage, resectability, performance status, molecular biomarkers, "
                            "PD-L1, prior treatment, and patient-specific constraints."
                        ),
                    },
                    {
                        "node_id": "NSCLC_MULTIMODAL_EVIDENCE",
                        "setting": "multimodal_evidence_review",
                        "recommendation": (
                            "Use pathology text, WSI patch evidence, CT ROI evidence, molecular "
                            "testing, and clinical context as complementary evidence."
                        ),
                    },
                ],
                "warnings": [
                    "Do not recommend targeted therapy without confirmed actionable alteration.",
                    "Do not use follow-up or outcome information during initial treatment planning.",
                    "Do not claim CT or WSI review without corresponding tool evidence.",
                ],
            },
        )

    @property
    def case_id(self) -> str:
        return self._case_dir().name

    def _case_dir(self) -> Path:
        return Path(self.case_dir).resolve()

    def _report_path(self) -> Path:
        return find_report_path(self._case_dir(), self.report_path)

    def _rubric_path(self) -> Path | None:
        if self.rubric_path is not None:
            return Path(self.rubric_path).resolve()
        try:
            return find_rubric_path(self._case_dir())
        except FileNotFoundError:
            return None

    def _load_case_yaml(self) -> dict[str, Any]:
        path = self._case_dir() / "case.yaml"
        if not path.is_file():
            return {}
        data = read_yaml(path)
        return data if isinstance(data, dict) else {}

    def _portable_path(self, path: Path | None) -> str:
        if path is None:
            return MISSING
        resolved = Path(path).resolve()
        project_root = self._project_root()
        try:
            return resolved.relative_to(project_root).as_posix()
        except ValueError:
            return str(resolved)

    def _project_root(self) -> Path:
        case_dir = self._case_dir()
        return case_dir.parents[2] if len(case_dir.parents) >= 3 else Path.cwd().resolve()

    def _treatment(self, report: str) -> dict[str, Any]:
        section = _section(report, "治疗信息")
        treatment_names = re.findall(r"-\s*([^（\n]+)(?:（([^）]+)）)?", section)
        treatments = []
        for index, (name, english) in enumerate(treatment_names, 1):
            treatments.append(
                {
                    "treatment_id": f"T{index:03d}",
                    "treatment_type": name.strip() or MISSING,
                    "treatment_type_english": english.strip() or MISSING,
                    "intent": MISSING,
                    "received": "no" if "未接受该疗法" in section else MISSING,
                    "start_day": MISSING,
                    "end_day": MISSING,
                    "disease_status": MISSING,
                    "source": self._report_path().name,
                }
            )
        if not treatments:
            treatments.append(
                {
                    "treatment_id": "T001",
                    "treatment_type": MISSING,
                    "intent": MISSING,
                    "received": MISSING,
                    "start_day": MISSING,
                    "end_day": MISSING,
                    "disease_status": MISSING,
                    "source": self._report_path().name,
                }
            )
        return {"case_id": self.case_id, "treatments": treatments}

    def _follow_up(self, report: str) -> dict[str, Any]:
        section = _section(report, "随访与结局")
        days = re.findall(r"第(\d+)天", section)
        return {
            "case_id": self.case_id,
            "vital_status": "alive" if "生存状态为存活" in report else MISSING,
            "last_follow_up_day": max([int(day) for day in days]) if days else MISSING,
            "disease_status_at_last_follow_up": "tumor_free" if "无肿瘤" in section else MISSING,
            "progression": {
                "occurred": MISSING,
                "type": MISSING,
                "evidence": MISSING,
                "day": MISSING,
            },
            "source": self._report_path().name,
        }

    def _slide_metadata(self, report: str) -> dict[str, Any]:
        section = _section(report, "病理切片信息")
        slides: list[dict[str, Any]] = []
        blocks = re.findall(
            r"\*\*切片\d+\*\*.*?(?=\n\*\*切片\d+\*\*|\Z)",
            section,
            flags=re.DOTALL,
        )
        for block in blocks:
            slide_id = _first_match(block, r"切片(TCGA-[A-Za-z0-9-]+)") or MISSING
            slides.append(
                {
                    "slide_id": slide_id,
                    "sample_id": _first_match(block, r"\(([^，）]+)") or MISSING,
                    "section_location": _field_value(block, "切面位置"),
                    "tumor_cell_percentage": _field_value(block, "肿瘤细胞占比"),
                    "tumor_nuclei_percentage": _field_value(block, "肿瘤细胞核占比"),
                    "normal_cell_percentage": _field_value(block, "正常细胞占比"),
                    "stromal_cell_percentage": _field_value(block, "间质细胞占比"),
                    "necrosis_percentage": _field_value(block, "坏死细胞占比"),
                    "source": self._report_path().name,
                }
            )
        if not slides:
            slides.append(
                {
                    "slide_id": MISSING,
                    "sample_id": MISSING,
                    "section_location": MISSING,
                    "tumor_cell_percentage": MISSING,
                    "tumor_nuclei_percentage": MISSING,
                    "normal_cell_percentage": MISSING,
                    "stromal_cell_percentage": MISSING,
                    "necrosis_percentage": MISSING,
                    "source": self._report_path().name,
                }
            )
        return {"case_id": self.case_id, "slides": slides}

    def _wsi_manifest(self, case_yaml: dict[str, Any]) -> dict[str, Any]:
        pathology = _nested(case_yaml, "data", "pathology")
        wsi_dir = _path_from_case(self._case_dir(), pathology.get("wsi_dir"))
        if wsi_dir is None:
            wsi_dir = self._case_dir() / "pathology" / "wsi"
        files = sorted(path for path in wsi_dir.glob("*") if path.is_file()) if wsi_dir.is_dir() else []
        return {
            "case_id": self.case_id,
            "available": bool(files),
            "slides": [
                {
                    "slide_id": pathology.get("slide_stem") or file.stem,
                    "wsi_path": self._portable_path(file),
                    "sample_id": "TCGA-38-4626-01Z" if self.case_id == "TCGA-38-4626" else MISSING,
                    "magnification": MISSING,
                    "mpp": MISSING,
                    "source": "pathology/wsi",
                }
                for file in files
            ]
            or [
                {
                    "slide_id": pathology.get("slide_stem") or MISSING,
                    "wsi_path": MISSING,
                    "sample_id": MISSING,
                    "magnification": MISSING,
                    "mpp": MISSING,
                    "source": self._report_path().name,
                }
            ],
        }

    def _wsi_roi_manifest(self, case_yaml: dict[str, Any]) -> dict[str, Any]:
        pathology = _nested(case_yaml, "data", "pathology")
        slide_dir = _find_conch_roi_slide_dir(self._case_dir(), case_yaml)
        slide_stem = slide_dir.name if slide_dir else pathology.get("slide_stem") or MISSING
        available = bool(slide_dir and (slide_dir / "prompt_scores.json").is_file())
        return {
            "case_id": self.case_id,
            "available": available,
            "skill_name": "pathology.conch_patch_roi",
            "slide_stem": slide_stem,
            "conch_roi_path": self._portable_path(slide_dir) if available and slide_dir else MISSING,
            "roi_artifacts": [],
        }

    def _ct_manifest(self, case_yaml: dict[str, Any]) -> dict[str, Any]:
        ct_uri = _nested(case_yaml, "data").get("ct_preprocessed_uri")
        ct_path = _path_from_case(self._case_dir(), ct_uri)
        source = "case.yaml"
        if ct_path is None:
            ct_path = _find_preprocessed_ct(self._case_dir())
            source = "radiology/nifti"
        available = bool(ct_path and ct_path.is_file())
        return {
            "case_id": self.case_id,
            "available": available,
            "series": [
                {
                    "series_id": "T0",
                    "ct_path": self._portable_path(ct_path) if ct_path else MISSING,
                    "modality": "CT",
                    "body_part": "chest",
                    "source": source if available else MISSING,
                    "preprocessed": True,
                }
            ],
        }


def _extract_age(text: str) -> int | None:
    match = re.search(r"(\d+)岁女性", text) or re.search(r"\*\*入组时年龄\*\*:\s*(\d+)岁", text)
    return int(match.group(1)) if match else None


def _extraction_audit_payload(extraction: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": extraction.get("status"),
        "method": extraction.get("method"),
        "cache_status": extraction.get("cache_status"),
        "model": extraction.get("model"),
        "usage": extraction.get("usage"),
        "warnings": _extraction_warnings(extraction),
        "clinical": _extracted_section(extraction, "clinical"),
        "treatment": _extracted_section(extraction, "treatment"),
        "follow_up": _extracted_section(extraction, "follow_up"),
        "molecular": _extracted_section(extraction, "molecular"),
    }


def _can_reuse_cached_extraction(extraction: Mapping[str, Any], mode: str) -> bool:
    method = str(extraction.get("method", "")).lower()
    if method == "llm":
        return True
    if mode == "disabled":
        return True
    if mode == "auto" and not _case_builder_llm_is_configured():
        return True
    return False


def _case_builder_llm_is_configured() -> bool:
    try:
        provider = resolve_provider(
            None,
            env_key="MEDCLAW_CASE_BUILDER_PROVIDER",
        )
        load_config_from_env(provider)
    except Exception:
        return False
    return True


def _read_optional_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = read_json(path)
    except Exception:
        return None
    return dict(data) if isinstance(data, Mapping) else None


def _existing_generated_extraction(case_dir: Path) -> dict[str, Any] | None:
    clinical = _read_optional_json_object(case_dir / "clinical" / "clinical.json")
    treatment = _read_optional_json_object(case_dir / "clinical" / "treatment.json")
    follow_up = _read_optional_json_object(case_dir / "clinical" / "follow_up.json")
    molecular = _read_optional_json_object(case_dir / "molecular" / "biomarkers.json")
    if not any((clinical, treatment, follow_up, molecular)):
        return None

    methods = [
        value.get("extraction_method")
        for value in (clinical, treatment, follow_up, molecular)
        if isinstance(value, Mapping)
    ]
    method = next(
        (str(item) for item in methods if isinstance(item, str) and item),
        "existing_files",
    )
    warnings: list[str] = []
    for value in (clinical, treatment, follow_up, molecular):
        if isinstance(value, Mapping):
            warnings.extend(_string_list(value.get("extraction_warnings")))

    return {
        "status": "cached",
        "method": method,
        "warnings": warnings,
        "clinical": _strip_generated_metadata(clinical or {}),
        "treatment": _strip_generated_metadata(treatment or {}),
        "follow_up": _strip_generated_metadata(follow_up or {}),
        "molecular": _molecular_output_to_extraction(molecular or {}),
    }


def _strip_generated_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    skipped = {
        "case_id",
        "source",
        "extraction_method",
        "extraction_warnings",
    }
    return {str(key): item for key, item in value.items() if key not in skipped}


def _molecular_output_to_extraction(value: Mapping[str, Any]) -> dict[str, Any]:
    molecular = _strip_generated_metadata(value)
    details = molecular.get("biomarker_details")
    if isinstance(details, Mapping):
        molecular["biomarkers"] = dict(details)
    return molecular


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def _extracted_section(
    extraction: Mapping[str, Any] | None,
    key: str,
) -> dict[str, Any]:
    if not isinstance(extraction, Mapping):
        return {}
    value = extraction.get(key)
    return dict(value) if isinstance(value, Mapping) else {}


def _extraction_method(extraction: Mapping[str, Any] | None) -> str:
    if not isinstance(extraction, Mapping):
        return "fallback"
    method = extraction.get("method")
    return str(method) if isinstance(method, str) and method else "fallback"


def _extraction_warnings(extraction: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(extraction, Mapping):
        return []
    return _string_list(extraction.get("warnings"))


def _with_extraction_metadata(
    data: Mapping[str, Any],
    extraction: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = dict(data)
    result["extraction_method"] = _extraction_method(extraction)
    result["extraction_warnings"] = _extraction_warnings(extraction)
    return result


def _merge_if_present(
    base: Mapping[str, Any],
    extracted: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(base)
    result.update(_clean_mapping(extracted))
    return result


def _clean_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        if _is_missing_like(item):
            continue
        cleaned[str(key)] = _clean_value(item)
    return cleaned


def _clean_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _clean_mapping(value)
    if isinstance(value, list):
        return [_clean_value(item) for item in value if not _is_missing_like(item)]
    return value


def _first_present(*values: Any) -> Any:
    for value in values:
        if not _is_missing_like(value):
            return value
    return MISSING


def _is_missing_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in {"", MISSING, "missing", "null", "None", "not reported"}
    return False


def _biomarker_details(value: Any) -> dict[str, Any]:
    raw = value if isinstance(value, Mapping) else {}
    details: dict[str, Any] = {}
    aliases = {
        "PDL1": "PD-L1",
        "PD_L1": "PD-L1",
        "HER2": "ERBB2",
    }
    for key, item in raw.items():
        canonical = aliases.get(str(key).upper(), str(key).upper())
        if canonical not in BIOMARKERS:
            continue
        if isinstance(item, Mapping):
            details[canonical] = _clean_mapping(item)
        elif not _is_missing_like(item):
            details[canonical] = {"summary": str(item)}
    return {
        name: details.get(name, {})
        for name in BIOMARKERS
    }


def _biomarker_summary(value: Any) -> str:
    if not isinstance(value, Mapping) or not value:
        return MISSING
    if summary := value.get("summary"):
        return str(summary)
    parts: list[str] = []
    for key in (
        "alteration_status",
        "status",
        "mutation",
        "cnv",
        "rna_expression",
        "interpretation",
        "evidence",
    ):
        item = value.get(key)
        if not _is_missing_like(item):
            parts.append(f"{key}={item}")
    return "; ".join(parts) if parts else MISSING


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if not _is_missing_like(item)]
    if isinstance(value, str) and value:
        return [value]
    return []


def _case_cancer_identity(*values: Any) -> dict[str, str]:
    """Infer the case family without retaining the old lung-only defaults."""

    text = " ".join(str(value) for value in values if value not in (None, ""))
    lowered = text.lower()
    if any(token in lowered for token in ("ucec", "endometr", "子宫内膜")):
        return {
            "project_id": "TCGA-UCEC",
            "cancer_type": "UCEC",
            "chief_problem": "endometrial carcinoma",
            "pathology_roi_skill": "pathology.ucec_conch_patch_roi",
            "radiology_roi_skill": "radiology.ucec_mri_roi",
        }
    if any(token in lowered for token in ("nasopharyn", " npc", "鼻咽")):
        return {
            "project_id": "NPC",
            "cancer_type": "NPC",
            "chief_problem": "nasopharyngeal carcinoma",
            "pathology_roi_skill": "pathology.conch_patch_roi",
            "radiology_roi_skill": "radiology.read_ct_manifest",
        }
    if (
        (re.search(r"\bsclc\b", lowered) and "nsclc" not in lowered)
        or "small cell lung" in lowered
        or "小细胞肺" in text
    ):
        return {
            "project_id": "SCLC",
            "cancer_type": "SCLC",
            "chief_problem": "small cell lung cancer",
            "pathology_roi_skill": "pathology.conch_patch_roi",
            "radiology_roi_skill": "radiology.lung_tumor_roi",
        }
    return {
        "project_id": "TCGA-LUAD",
        "cancer_type": "NSCLC-LUAD",
        "chief_problem": "primary lung adenocarcinoma",
        "pathology_roi_skill": "pathology.conch_patch_roi",
        "radiology_roi_skill": "radiology.lung_tumor_roi",
    }


def _extract_sex(text: str) -> str:
    value = _field_value(text, "性别")
    if value == "女性" or "女性患者" in text:
        return "female"
    if value == "男性" or "男性患者" in text:
        return "male"
    return MISSING


def _field_value(text: str, label: str) -> str:
    patterns = [
        rf"-\s*\*\*{re.escape(label)}\*\*:\s*([^\n]+)",
        rf"\*\*{re.escape(label)}\*\*:\s*([^\n]+)",
        rf"-\s*{re.escape(label)}[：:]\s*([^\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return MISSING


def _first_match(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1).strip() if match else None


def _section(text: str, heading: str) -> str:
    pattern = rf"^##\s+{re.escape(heading)}\s*$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return ""
    start = match.end()
    next_match = re.search(r"^##\s+", text[start:], flags=re.MULTILINE)
    end = start + next_match.start() if next_match else len(text)
    return text[start:end].strip()


def _nested(data: dict[str, Any], *keys: str) -> dict[str, Any]:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key, {})
    return current if isinstance(current, dict) else {}


def _path_from_case(case_dir: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    project_root = case_dir.parents[2] if len(case_dir.parents) >= 3 else Path.cwd()
    return (project_root / path).resolve()


def _find_preprocessed_ct(case_dir: Path) -> Path | None:
    preferred = sorted(
        path
        for path in (case_dir / "radiology" / "nifti").glob("*_ct_preprocessed.nii.gz")
        if path.is_file()
    )
    if preferred:
        return preferred[0].resolve()
    matches = sorted(
        path for path in case_dir.rglob("*_ct_preprocessed.nii.gz") if path.is_file()
    )
    return matches[0].resolve() if matches else None


def _find_conch_roi_slide_dir(
    case_dir: Path,
    case_yaml: dict[str, Any],
) -> Path | None:
    pathology = _nested(case_yaml, "data", "pathology")
    roots: list[Path] = []
    configured = _path_from_case(case_dir, pathology.get("conch_roi_dir"))
    if configured is not None:
        roots.append(configured)
    roots.extend(
        [
            case_dir / "pathology" / "roi_256",
            case_dir / "pathology" / "roi_512",
            case_dir / "pathology" / "conch_roi",
        ]
    )
    slide_stem = pathology.get("slide_stem")
    for root in roots:
        if (root / "prompt_scores.json").is_file():
            return root.resolve()
        if not root.is_dir():
            continue
        if isinstance(slide_stem, str) and slide_stem:
            exact = root / slide_stem
            if (exact / "prompt_scores.json").is_file():
                return exact.resolve()
        matches = sorted(
            path.parent for path in root.rglob("prompt_scores.json") if path.is_file()
        )
        if matches:
            return matches[0].resolve()
    return None
