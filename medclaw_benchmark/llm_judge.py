"""LLM-as-judge rubric scoring for benchmark runs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from medclaw.llm.factory import create_llm_client, load_config_from_env, resolve_provider
from medclaw.llm.protocols import ChatCompletion
from medclaw.utils import read_json, write_json

from medclaw_benchmark.io_utils import read_jsonl, write_text
from medclaw_benchmark.trajectory_scorer import (
    combine_micro_macro_scores,
    score_trajectory_run,
)


DIMENSION_WEIGHTS = {
    "CI": 20,
    "GM": 15,
    "RA": 25,
    "CR": 15,
    "EG": 15,
    "COMM": 10,
}
class JudgeResponseError(ValueError):
    """Raised when the LLM judge response cannot be normalized."""


@dataclass
class LLMRubricJudge:
    """Evaluate benchmark answers with an LLM rubric judge."""

    rubric_path: str | Path
    run_dir: str | Path
    llm_client: Any | None = None
    weights: Mapping[str, int] | None = None
    provider: str | None = None
    config: Any | None = None
    strict_static: bool = False
    ultra_strict_static: bool = False
    dimensions: Mapping[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.rubric_path = Path(self.rubric_path).resolve()
        self.run_dir = Path(self.run_dir).resolve()
        self.dimensions = dict(self.weights or DIMENSION_WEIGHTS)
        if self.llm_client is None:
            judge_provider = resolve_provider(
                self.provider,
                env_key="MEDCLAW_JUDGE_PROVIDER",
            )
            judge_config = self.config or load_config_from_env(judge_provider)
            self.config = judge_config
            self.llm_client = create_llm_client(judge_provider, config=judge_config)
        elif self.config is None:
            self.config = getattr(self.llm_client, "config", None)

    def evaluate(self) -> dict[str, Any]:
        """Run the LLM judge, write outputs, and return judge_scores."""

        prompt_payload = self.build_prompt_payload()
        messages = self._messages(prompt_payload)
        write_json(
            self.run_dir / "judge_prompt.json",
            {
                "messages": messages,
                "payload": prompt_payload,
                "note": "This file is for judge audit only. It is never shown to the evaluated agent.",
            },
        )

        raw_response: dict[str, Any] | None = None
        try:
            completion: ChatCompletion = self.llm_client.complete(
                messages=messages,
                tools=[],
                tool_choice="none",
            )
            raw_response = {
                "message": completion.message,
                "model": completion.model,
                "usage": completion.usage,
            }
            write_json(self.run_dir / "judge_raw_response.json", raw_response)
            llm_scores = self._normalize_response(completion.message)
            result = self._finalize_scores(llm_scores, prompt_payload, raw_response)
        except Exception as exc:
            if raw_response is None:
                write_json(
                    self.run_dir / "judge_raw_response.json",
                    {
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        }
                    },
                )
            result = self._failed_result(str(exc), prompt_payload)

        trajectory_scores = score_trajectory_run(self.run_dir)
        result["trajectory_scores_path"] = str(self.run_dir / "trajectory_scores.json")
        result["trajectory_evaluation"] = {
            "status": trajectory_scores.get("status"),
            "macro_trajectory_total": trajectory_scores.get("macro_trajectory_total"),
            "coverage": trajectory_scores.get("legacy_macro", {}).get("coverage"),
            "expected_action_count": trajectory_scores.get("legacy_macro", {}).get(
                "expected_action_count"
            ),
            "matched_action_count": trajectory_scores.get("legacy_macro", {}).get(
                "matched_action_count"
            ),
            "alignment_size": trajectory_scores.get("alignment_size"),
            "avg_match_score": trajectory_scores.get("avg_match_score"),
            "avg_state_similarity": trajectory_scores.get("avg_state_similarity"),
            "avg_action_match": trajectory_scores.get("avg_action_match"),
            "avg_act_score": trajectory_scores.get("avg_act_score"),
            "avg_step_score": trajectory_scores.get("avg_step_score"),
            "miss_step_count": trajectory_scores.get("miss_step_count"),
            "redundant_step_count": trajectory_scores.get("redundant_step_count"),
            "coherence": trajectory_scores.get("coherence"),
            "dyn_traj_score": trajectory_scores.get("dyn_traj_score"),
            "score_summary": trajectory_scores.get("score_summary"),
            "trajectory_score_components": trajectory_scores.get(
                "trajectory_score_components"
            ),
        }
        result["dual_layer_scores"] = combine_micro_macro_scores(
            micro_total=result.get("final_total"),
            macro_total=trajectory_scores.get("macro_trajectory_total"),
        )
        write_json(self.run_dir / "judge_scores.json", result)
        self._write_error_analysis(result)
        return result

    def build_prompt_payload(self) -> dict[str, Any]:
        rubric = self._read_object(self.rubric_path)
        final_answers = self._read_object(self.run_dir / "final_answers.json")
        evidence_board = self._read_object(self.run_dir / "evidence_board.json")
        trajectory = _compact_records(read_jsonl(self.run_dir / "trajectory.jsonl"))
        tool_calls = _compact_records(read_jsonl(self.run_dir / "tool_calls.jsonl"))
        return {
            "case_id": rubric.get("case_id") or final_answers.get("case_id"),
            "run_id": final_answers.get("run_id"),
            "rubric_version": rubric.get("rubric_version"),
            "case_summary": rubric.get("case_summary", []),
            "rubric": rubric.get("rubric", {}),
            "manual_review_flags": rubric.get("manual_review_flags", []),
            "weights": dict(self.dimensions),
            "final_answer": final_answers.get("answer_text", ""),
            "trajectory_summary": trajectory,
            "tool_calls": tool_calls,
            "evidence_board": _artifact_metadata_only(evidence_board),
        }

    def _messages(self, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        system = (
            "你是 MedClaw benchmark 的严格医学评测 Judge。"
            "你只能根据 rubric、final answer、trajectory、tool_calls 和 evidence_board 评分。"
            "rubric 只供 Judge 使用，不能假设被评测 Agent 看过 rubric。"
            "不要因为 CT/WSI 工具环境缺失而惩罚 Agent；如果 final answer 编造已经查看或编造视觉结论，"
            "请在相关维度的 missed、rationale 或 needs_manual_review 中说明。"
            "必须只输出 JSON，不要输出 Markdown。"
            "JSON 字符串内如需引号请使用中文弯引号（“”）或单引号，"
            "不要使用未转义的 ASCII 双引号。"
        )
        if self.strict_static or self.ultra_strict_static:
            system += (
                "本次使用严格静态评分协议：逐条 rubric 原子标准核对，只有 final answer、"
                "trajectory、tool_calls 或 evidence_board 中可定位的明确内容才能计分；"
                "仅有笼统方向、相近措辞或 Judge 自行推断不得视为完全命中。"
                "partially_matched 必须按缺失比例扣分，不能因整体印象补分。"
                "诊断和分期必须区分已证实、推断与待确认；缺失检查必须说明其决策影响。"
                "指南映射必须准确到适用亚组、推荐层级和证据类别；未明确写出的等级不得计分。"
                "治疗建议必须包含适用条件、重要替代路径、禁忌或安全边界；"
                "把条件性方案写成无条件方案应在相关维度显著扣分。"
                "引用必须真实支持对应主张，不能以引用了指南名称替代具体证据。"
                "每个维度的 matched、missed、criteria 和 score 必须相互一致。"
            )
        if self.ultra_strict_static:
            system += (
                "使用究极严格、可复现的原子计分：先把每个维度的适用 rubric 拆成最小可验证条目，"
                "原则上等权计分（rubric 明示权重时服从明示权重）。"
                "完全且直接满足记1.0，部分满足最多记0.25，隐含、模糊、靠常识补全或完全缺失记0。"
                "内容只在 trajectory 中出现、但应当进入最终诊疗回答而未收敛时，最多按部分满足。"
                "一个句子笼统覆盖多个检查、条件或风险时，不得替代逐项要求。"
                "诊断、TNM/分期、病理亚型、关键生物标志物、M分期与证据缺口必须逐项正确；"
                "任何把未知写成阴性/正常/已排除的表述均视为严重错误。"
                "指南映射必须同时给出适用亚组、推荐动作、推荐等级、证据类别及其条件；"
                "任一缺失时该原子条目不得算完全命中。"
                "EG 中只写指南名称、年份或章节而没有逐主张对应的具体支持，不得获得引用准确性分；"
                "引用错误、版本错误或证据不支持主张时该条记0并在 missed 中指出。"
                "RA 必须逐项覆盖必要检查、首选路径、合理替代、实施前提、重要毒性/禁忌及随访；"
                "缺一项即按原子比例扣分。"
                "CR 对每条负向约束分别核对；任何无条件越级治疗、条件错配或危险建议均应显著扣分。"
                "COMM 不因语言流畅自动高分，必须清楚区分事实、建议、条件、不确定性和下一步。"
                "禁止使用整体印象、文风、篇幅或模型身份补分；宁可保守低分，也不得推定未写出的内容。"
            )
        schema = {
            "dimensions": {
                dim: {
                    "score": f"0 到 {max_score} 的数字",
                    "max_score": max_score,
                    "matched": ["满足的 rubric 条目"],
                    "missed": ["未满足的 rubric 条目"],
                    "rationale": "中文评分理由",
                    "needs_manual_review": ["需要人工复核的点"],
                    "criteria": [
                        {
                            "criterion": "rubric 原文",
                            "status": "matched|partially_matched|missed|not_applicable",
                            "reason": "简短理由",
                        }
                    ],
                }
                for dim, max_score in self.dimensions.items()
            },
            "overall_rationale": "中文总体评价",
        }
        user = {
            "instruction": (
                "请按给定权重为 CI、GM、RA、CR、EG、COMM 六个维度打分。"
                "分数可以是小数，但不得超过 max_score。"
                "如果证据不足，请降低相关维度并在 missed 或 needs_manual_review 中说明。"
                + (
                    "严格逐项计分：不得把未明确陈述的内容视为已满足；每个满分维度必须证明"
                    "该维度所有适用 rubric 原子标准均已明确满足。"
                    if (self.strict_static or self.ultra_strict_static)
                    else ""
                )
                + (
                    "采用究极严格原子计分：完全明确=1，部分满足最多=0.25，隐含或缺失=0；"
                    "逐条列出所有扣分依据，禁止四舍五入式宽松给分。"
                    if self.ultra_strict_static
                    else ""
                )
            ),
            "required_output_schema": schema,
            "evaluation_payload": payload,
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False, sort_keys=True)},
        ]

    def _normalize_response(self, message: Mapping[str, Any]) -> dict[str, Any]:
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise JudgeResponseError("LLM judge response did not contain text content.")
        data = _extract_json_object(content)
        dimensions = data.get("dimensions", data.get("scores"))
        if not isinstance(dimensions, Mapping):
            raise JudgeResponseError("LLM judge JSON must contain a dimensions object.")

        normalized: dict[str, Any] = {"dimensions": {}, "overall_rationale": data.get("overall_rationale", "")}
        for dim, max_score in self.dimensions.items():
            raw = dimensions.get(dim)
            if not isinstance(raw, Mapping):
                raise JudgeResponseError(f"LLM judge response is missing dimension {dim}.")
            score = _clamp_float(raw.get("score"), 0.0, float(max_score))
            normalized["dimensions"][dim] = {
                "score": score,
                "max_score": max_score,
                "matched": _string_list(raw.get("matched")),
                "missed": _string_list(raw.get("missed")),
                "rationale": str(raw.get("rationale", "")),
                "needs_manual_review": _string_list(raw.get("needs_manual_review")),
                "criteria": _criteria_list(raw.get("criteria")),
            }
        return normalized

    def _finalize_scores(
        self,
        llm_scores: dict[str, Any],
        payload: Mapping[str, Any],
        raw_response: Mapping[str, Any],
    ) -> dict[str, Any]:
        raw_total = round(
            sum(float(item["score"]) for item in llm_scores["dimensions"].values()),
            3,
        )
        return {
            "status": "success",
            "judge_failed": False,
            "case_id": payload.get("case_id"),
            "run_id": payload.get("run_id"),
            "judge_type": "llm",
            "rubric_version": payload.get("rubric_version"),
            "weights": dict(self.dimensions),
            "llm_scores": llm_scores["dimensions"],
            "overall_rationale": llm_scores.get("overall_rationale", ""),
            "raw_total": raw_total,
            "final_total": raw_total,
            "score_interpretation": {
                "final_total": "micro final-answer score from the case-specific rubric",
                "trajectory_scores": "macro process/tree score from guideline_trajectory.json",
            },
            "manual_review_flags": payload.get("manual_review_flags", []),
            "judge_model": _public_config(self.config),
            "judge_response_model": raw_response.get("model"),
            "judge_usage": raw_response.get("usage"),
        }

    def _failed_result(self, error: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "status": "judge_failed",
            "judge_failed": True,
            "case_id": payload.get("case_id"),
            "run_id": payload.get("run_id"),
            "judge_type": "llm",
            "error": error,
            "weights": dict(self.dimensions),
            "llm_scores": {},
            "raw_total": None,
            "final_total": None,
            "manual_review_flags": payload.get("manual_review_flags", []),
            "judge_model": _public_config(self.config),
        }

    def _write_error_analysis(self, result: Mapping[str, Any]) -> None:
        lines = [
            "# Rubric Evaluation Report",
            "",
            "## Case",
            f"- Case ID: {result.get('case_id')}",
            f"- Run ID: {result.get('run_id')}",
            f"- Rubric version: {result.get('rubric_version', 'not_available')}",
            f"- Judge status: {result.get('status')}",
            "",
            "## Total Score",
            f"- Raw total: {result.get('raw_total')}",
            f"- Final total: {result.get('final_total')}",
        ]
        lines.extend(_trajectory_score_markdown(result, self.run_dir))
        lines.extend(["", "## Dimension Scores"])
        scores = result.get("llm_scores", {})
        if isinstance(scores, Mapping):
            for dim in self.dimensions:
                score = scores.get(dim, {})
                if not isinstance(score, Mapping):
                    continue
                lines.extend(
                    [
                        "",
                        f"### {dim}: {score.get('score')} / {score.get('max_score')}",
                        f"Rationale: {score.get('rationale', '')}",
                        "Matched:",
                        *[f"- {item}" for item in score.get("matched", [])],
                        "Missed:",
                        *[f"- {item}" for item in score.get("missed", [])],
                    ]
                )
        lines.extend(["", "## Manual Review Flags"])
        manual_flags = [f"- {item}" for item in result.get("manual_review_flags", [])]
        lines.extend(manual_flags or ["- None"])
        write_text(self.run_dir / "error_analysis.md", "\n".join(lines) + "\n")

    @staticmethod
    def _read_object(path: Path) -> dict[str, Any]:
        data = read_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a JSON object: {path}")
        return data


def _trajectory_score_markdown(
    result: Mapping[str, Any], run_dir: Path
) -> list[str]:
    path = run_dir / "trajectory_scores.json"
    trajectory = result.get("trajectory_evaluation", {})
    if not path.is_file():
        return [
            "",
            "## Trajectory Process Score",
            "- Status: missing",
            "- trajectory_scores.json was not found.",
        ]
    try:
        data = read_json(path)
    except Exception as exc:
        return [
            "",
            "## Trajectory Process Score",
            f"- Status: unreadable ({type(exc).__name__}: {exc})",
        ]
    if not isinstance(data, Mapping):
        return [
            "",
            "## Trajectory Process Score",
            "- Status: invalid",
        ]

    dual = result.get("dual_layer_scores", {})
    legacy = data.get("legacy_macro", {})
    lines = [
        "",
        "## Trajectory Process Score",
        f"- Status: {data.get('status')}",
        f"- Score type: {data.get('score_type')}",
        f"- Macro trajectory total: {data.get('macro_trajectory_total')}",
        f"- DynTraj score: {data.get('dyn_traj_score')}",
        f"- Alignment size: {data.get('alignment_size')}",
        f"- Average match score: {data.get('avg_match_score')}",
        f"- Average state similarity: {data.get('avg_state_similarity')}",
        f"- Average action match: {data.get('avg_action_match')}",
        f"- Average ActScore: {data.get('avg_act_score')}",
        f"- Average step score: {data.get('avg_step_score')}",
        f"- Miss steps: {data.get('miss_step_count')}",
        f"- Redundant steps: {data.get('redundant_step_count')}",
        f"- Coherence: {data.get('coherence')}",
        f"- Legacy action coverage: {legacy.get('coverage')}",
        f"- Combined dual-layer total: {dual.get('combined_total') if isinstance(dual, Mapping) else None}",
    ]
    if isinstance(trajectory, Mapping) and trajectory:
        lines.append(f"- Trajectory scores path: {result.get('trajectory_scores_path')}")

    weak_steps = []
    for step in data.get("miss_steps", []):
        if isinstance(step, Mapping):
            weak_steps.append(step)
    for step in data.get("step_scores", []):
        if not isinstance(step, Mapping):
            continue
        coverage = step.get("coverage")
        if coverage is None or float(coverage) >= 0.8:
            continue
        weak_steps.append(step)
    if weak_steps:
        lines.extend(["", "### Low-Coverage Steps"])
        for step in weak_steps:
            if "phase" in step and "gold_index" in step:
                lines.append(
                    f"- Missed gold step {step.get('step')} ({step.get('phase')})"
                )
                continue
            missed = step.get("missed_actions", [])
            missed_text = []
            if isinstance(missed, list):
                missed_text = [
                    str(item.get("action"))
                    for item in missed
                    if isinstance(item, Mapping) and item.get("action")
                ]
            lines.append(
                f"- Step {step.get('step')} ({step.get('phase')}): "
                f"{step.get('matched_count')} / {step.get('expected_count')} matched; "
                f"missed: {', '.join(missed_text) if missed_text else 'none listed'}"
            )
    return lines


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise JudgeResponseError("LLM judge response was not valid JSON.")
    candidate = stripped[start : end + 1]

    last_error: json.JSONDecodeError | None = None
    for attempt in (
        candidate,
        _repair_judge_json(candidate),
    ):
        try:
            data = json.loads(attempt)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if not isinstance(data, dict):
            raise JudgeResponseError("LLM judge JSON must be an object.")
        return data

    message = last_error.msg if last_error is not None else "unknown parse error"
    raise JudgeResponseError(f"LLM judge response was not valid JSON: {message}") from last_error


def _repair_judge_json(text: str) -> str:
    """Best-effort repair for common LLM judge JSON formatting mistakes."""

    repaired = _fix_unescaped_quotes_in_json_strings(text)
    repaired = _fix_missing_criteria_object_braces(repaired)
    return repaired


def _fix_unescaped_quotes_in_json_strings(text: str) -> str:
    """Escape ASCII double quotes that appear inside JSON string values."""

    result: list[str] = []
    in_string = False
    escape = False
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if not in_string:
            result.append(char)
            if char == '"':
                in_string = True
                escape = False
            index += 1
            continue

        if escape:
            result.append(char)
            escape = False
            index += 1
            continue

        if char == "\\":
            result.append(char)
            escape = True
            index += 1
            continue

        if char == '"':
            lookahead = index + 1
            while lookahead < length and text[lookahead] in " \t\r\n":
                lookahead += 1
            if lookahead >= length or text[lookahead] in ":,}]":
                result.append(char)
                in_string = False
            else:
                result.append('\\"')
            index += 1
            continue

        result.append(char)
        index += 1

    return "".join(result)


def _fix_missing_criteria_object_braces(text: str) -> str:
    """Close criteria objects when the judge omits `}` before `],`."""

    return re.sub(
        r'("status"\s*:\s*"(?:matched|partially_matched|missed|not_applicable)")'
        r'(\s*\n\s*)(?!})(\],)',
        r"\1\2}\3",
        text,
    )


def _clamp_float(value: Any, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = minimum
    return round(min(max(number, minimum), maximum), 3)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value in (None, ""):
        return []
    return [str(value)]


def _criteria_list(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, Mapping):
            result.append(
                {
                    "criterion": str(item.get("criterion", "")),
                    "status": str(item.get("status", "missed")),
                    "reason": str(item.get("reason", "")),
                }
            )
    return result


def _compact_records(records: list[dict[str, Any]], limit: int = 50) -> list[dict[str, Any]]:
    compact = []
    for record in records[:limit]:
        compact.append(
            {
                key: value
                for key, value in record.items()
                if key
                in {
                    "event_type",
                    "phase",
                    "skill_name",
                    "status",
                    "summary",
                    "artifact_paths",
                    "final_answer",
                    "answer_text",
                    "timestamp",
                }
            }
        )
    return compact


def _artifact_metadata_only(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    items = result.get("evidence_items", result.get("evidence", []))
    if not isinstance(items, list):
        return result
    cleaned = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        copy = dict(item)
        artifacts = copy.get("artifacts")
        if isinstance(artifacts, list):
            copy["artifacts"] = [
                {
                    key: artifact.get(key)
                    for key in ("type", "role", "uri", "size_bytes", "sha256")
                    if isinstance(artifact, Mapping) and key in artifact
                }
                for artifact in artifacts
                if isinstance(artifact, Mapping)
            ]
        cleaned.append(copy)
    if "evidence_items" in result:
        result["evidence_items"] = cleaned
    else:
        result["evidence"] = cleaned
    return result


def _public_config(config: Any) -> dict[str, Any]:
    summary = getattr(config, "public_summary", None)
    if callable(summary):
        return dict(summary())
    return {"client_type": type(config).__name__ if config is not None else None}
