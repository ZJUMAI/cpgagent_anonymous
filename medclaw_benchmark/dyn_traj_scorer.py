"""Dynamic rubric-guided trajectory alignment scoring (§4.1–4.4)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from medclaw.trajectory.action_set import flatten_visible_state, resolve_action_set
from medclaw.utils import read_json, write_json

from medclaw_benchmark.io_utils import read_jsonl
from medclaw_benchmark.trajectory_scorer import (
    _matches_action,
    _matches_unsafe_action,
    _normalized,
    score_trajectory_run_legacy,
)

# Alignment hyperparameters
ALPHA_STATE = 0.4
BETA_ACTION = 0.6
LAMBDA_GAP = 0.3
GAMMA_STATE_IN_STEP = 0.3

# ActScore weights: Cover, Accept, CondAware, UnsafeHit
W_COVER = 0.5
W_ACCEPT = 0.15
W_COND = 0.15
W_UNSAFE = 0.2

# Trajectory-level penalties / bonus
LAMBDA_MISS = 0.15
LAMBDA_RED = 0.1
LAMBDA_COH = 0.05

CONDITION_MARKERS = (
    "if ",
    " if ",
    "when ",
    " when ",
    "provided that",
    "conditional",
    "pending",
    "若",
    "如果",
    "当",
    "待",
    "条件",
)

STATE_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "tnm": ("tnm", "known_stage", "stage", "pathologic_stage"),
    "ecog": ("ECOG", "ecog", "performance_status"),
    "histology": ("histology", "known_diagnosis", "diagnosis"),
    "margin": ("margin_or_residual_status", "margin_status", "margin"),
    "surgery": ("surgery", "surgical_history"),
    "biomarkers": ("biomarkers_mentioned", "known_biomarkers", "molecular_summary"),
    "postoperative": ("postoperative_case",),
}


@dataclass
class GoldStep:
    index: int
    step_id: int | None
    phase: str | None
    visible_state: dict[str, Any]
    action_set: dict[str, list[dict[str, Any]]]
    raw: dict[str, Any] = field(repr=False)


@dataclass
class AgentStep:
    index: int
    round_index: int | None
    phase: str | None
    action_text: str
    patient_state: dict[str, Any]
    source: str
    raw: dict[str, Any] = field(repr=False)


@dataclass
class AlignmentPair:
    gold_index: int
    agent_index: int
    match_score: float
    state_sim: float
    act_match: float
    act_score: float
    step_score: float
    gold_indices: tuple[int, ...] = ()
    agent_indices: tuple[int, ...] = ()
    match_score_detail: dict[str, Any] = field(default_factory=dict)
    act_score_detail: dict[str, Any] = field(default_factory=dict)
    step_score_detail: dict[str, Any] = field(default_factory=dict)


def score_dyn_trajectory_run(run_dir: str | Path) -> dict[str, Any]:
    """Score one benchmark run via dynamic rubric step alignment."""

    run_path = Path(run_dir).resolve()
    trajectory_path = run_path / "guideline_trajectory.json"
    final_answer_path = run_path / "final_answers.json"
    if not trajectory_path.is_file():
        return _missing_result(run_path, "guideline_trajectory.json is missing")
    if not final_answer_path.is_file():
        return _missing_result(run_path, "final_answers.json is missing")

    trajectory = read_json(trajectory_path)
    final_answers = read_json(final_answer_path)
    gold_steps = extract_gold_steps(trajectory)
    agent_steps = extract_agent_steps(run_path, final_answers)

    if not gold_steps:
        legacy = score_trajectory_run_legacy(run_path)
        legacy["score_type"] = "dyn_traj_alignment_v1"
        legacy["status"] = "missing"
        legacy["reason"] = "guideline_trajectory has no steps"
        write_json(run_path / "trajectory_scores.json", legacy)
        return legacy

    if not agent_steps:
        legacy = score_trajectory_run_legacy(run_path)
        legacy["score_type"] = "dyn_traj_alignment_v1"
        legacy["status"] = "missing"
        legacy["reason"] = "no agent steps found in dual_agent_rounds or tool_calls"
        write_json(run_path / "trajectory_scores.json", legacy)
        return legacy

    alignment, miss_gold, miss_agent = align_steps(gold_steps, agent_steps)
    marginal_redundant = _marginal_redundant_agent_steps(
        gold_steps, agent_steps, alignment
    )
    miss_agent = sorted(set(miss_agent) | marginal_redundant)
    step_pair_scores = score_alignment_pairs(gold_steps, agent_steps, alignment)
    coherence_detail = compute_coherence(gold_steps, agent_steps, step_pair_scores)
    coherence = coherence_detail["score"]

    if step_pair_scores:
        avg_step = sum(pair.step_score for pair in step_pair_scores) / len(step_pair_scores)
    else:
        avg_step = 0.0

    score_summary = _build_score_summary(step_pair_scores)

    n_gold = len(gold_steps)
    n_agent = len(agent_steps)
    miss_rate = len(miss_gold) / n_gold if n_gold else 0.0
    red_rate = len(miss_agent) / n_agent if n_agent else 0.0

    miss_penalty = LAMBDA_MISS * miss_rate
    redundant_penalty = LAMBDA_RED * red_rate
    coherence_bonus = LAMBDA_COH * coherence
    dyn_score_before_clamp = (
        avg_step
        - miss_penalty
        - redundant_penalty
        + coherence_bonus
    )
    dyn_score = max(0.0, min(1.0, dyn_score_before_clamp))
    macro_total = round(dyn_score * 100, 3)

    legacy = score_trajectory_run_legacy(run_path)

    result = {
        "status": "success",
        "case_id": trajectory.get("case_id") or final_answers.get("case_id"),
        "run_id": final_answers.get("run_id"),
        "score_type": "dyn_traj_alignment_v1",
        "macro_trajectory_total": macro_total,
        "dyn_traj_score": round(dyn_score, 4),
        "avg_step_score": round(avg_step, 4),
        "avg_match_score": score_summary["alignment"]["avg_match_score"],
        "avg_state_similarity": score_summary["alignment"]["avg_state_similarity"],
        "avg_action_match": score_summary["alignment"]["avg_action_match"],
        "avg_act_score": score_summary["action"]["avg_act_score"],
        "score_summary": score_summary,
        "trajectory_score_components": {
            "avg_step_score": round(avg_step, 4),
            "miss_rate": round(miss_rate, 4),
            "miss_penalty": round(miss_penalty, 4),
            "redundant_step_rate": round(red_rate, 4),
            "redundant_step_penalty": round(redundant_penalty, 4),
            "coherence": round(coherence, 4),
            "coherence_bonus": round(coherence_bonus, 4),
            "score_before_clamp": round(dyn_score_before_clamp, 4),
            "score_after_clamp": round(dyn_score, 4),
        },
        "alignment_size": len(step_pair_scores),
        "gold_step_count": n_gold,
        "agent_step_count": n_agent,
        "miss_step_count": len(miss_gold),
        "redundant_step_count": len(miss_agent),
        "miss_step_rate": round(miss_rate, 4),
        "redundant_step_rate": round(red_rate, 4),
        "coherence": round(coherence, 4),
        "coherence_detail": coherence_detail,
        "alignment": [
            {
                "gold_index": pair.gold_index,
                "agent_index": pair.agent_index,
                "gold_step": gold_steps[pair.gold_index].step_id,
                "gold_phase": gold_steps[pair.gold_index].phase,
                "agent_round": agent_steps[pair.agent_index].round_index,
                "match_score": round(pair.match_score, 4),
                "state_sim": round(pair.state_sim, 4),
                "act_match": round(pair.act_match, 4),
                "act_score": round(pair.act_score, 4),
                "step_score": round(pair.step_score, 4),
                "gold_indices": list(pair.gold_indices or (pair.gold_index,)),
                "agent_indices": list(pair.agent_indices or (pair.agent_index,)),
                "match_score_detail": _rounded_mapping(pair.match_score_detail),
                "act_score_detail": _rounded_mapping(pair.act_score_detail),
                "step_score_detail": _rounded_mapping(pair.step_score_detail),
            }
            for pair in step_pair_scores
        ],
        "step_pair_scores": [
            {
                "gold_index": pair.gold_index,
                "agent_index": pair.agent_index,
                "step_score": round(pair.step_score, 4),
                "act_score": round(pair.act_score, 4),
                "state_sim": round(pair.state_sim, 4),
                "act_match": round(pair.act_match, 4),
                "match_score": round(pair.match_score, 4),
                "match_score_detail": _rounded_mapping(pair.match_score_detail),
                "act_score_detail": _rounded_mapping(pair.act_score_detail),
                "step_score_detail": _rounded_mapping(pair.step_score_detail),
            }
            for pair in step_pair_scores
        ],
        "miss_steps": [
            {
                "gold_index": idx,
                "step": gold_steps[idx].step_id,
                "phase": gold_steps[idx].phase,
            }
            for idx in miss_gold
        ],
        "redundant_steps": [
            {
                "agent_index": idx,
                "round_index": agent_steps[idx].round_index,
                "phase": agent_steps[idx].phase,
                "source": agent_steps[idx].source,
            }
            for idx in miss_agent
        ],
        "agent_step_source": agent_steps[0].source if agent_steps else None,
        "legacy_macro": {
            "macro_trajectory_total": legacy.get("macro_trajectory_total"),
            "coverage": legacy.get("coverage"),
            "score_type": legacy.get("score_type"),
        },
        "hyperparameters": {
            "alpha_state": ALPHA_STATE,
            "beta_action": BETA_ACTION,
            "lambda_gap": LAMBDA_GAP,
            "gamma_state_in_step": GAMMA_STATE_IN_STEP,
            "w_cover": W_COVER,
            "w_accept": W_ACCEPT,
            "w_cond": W_COND,
            "w_unsafe": W_UNSAFE,
            "lambda_miss": LAMBDA_MISS,
            "lambda_red": LAMBDA_RED,
            "lambda_coh": LAMBDA_COH,
        },
        "limitations": [
            "Step alignment uses monotonic DP with keyword-based Match (no embedding/LLM).",
            "Agent steps prefer dual_agent_rounds.jsonl; single-agent runs fall back to phase-grouped tool_calls.",
            "legacy_macro retains the previous final-answer keyword coverage score for comparison.",
        ],
    }
    write_json(run_path / "trajectory_scores.json", result)
    return result


def extract_gold_steps(trajectory: Mapping[str, Any]) -> list[GoldStep]:
    steps = trajectory.get("trajectory", [])
    if not isinstance(steps, list):
        return []
    gold: list[GoldStep] = []
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            continue
        visible = flatten_visible_state(step.get("visible_state"))
        action_set = resolve_action_set(step)
        gold.append(
            GoldStep(
                index=len(gold),
                step_id=step.get("step"),
                phase=str(step.get("phase") or "") or None,
                visible_state=visible,
                action_set=action_set,
                raw=dict(step),
            )
        )
    return gold


def extract_agent_steps(
    run_path: Path,
    final_answers: Mapping[str, Any],
) -> list[AgentStep]:
    dual_rounds_path = run_path / "dual_agent_rounds.jsonl"
    if dual_rounds_path.is_file():
        rounds = read_jsonl(dual_rounds_path)
        if rounds:
            state_by_round = _patient_state_by_round(run_path)
            steps: list[AgentStep] = []
            for index, record in enumerate(rounds):
                if not isinstance(record, Mapping):
                    continue
                round_index = record.get("round_index")
                planner = record.get("planner_output") or {}
                if not isinstance(planner, Mapping):
                    planner = {}
                action_parts = [
                    str(planner.get("next_step") or ""),
                    str(record.get("agent_round_answer") or ""),
                ]
                tool_skills = record.get("tool_skills") or []
                if isinstance(tool_skills, list):
                    action_parts.extend(str(skill) for skill in tool_skills if skill)
                action_text = "\n".join(part for part in action_parts if part.strip())
                patient_state = state_by_round.get(round_index, {})
                if not patient_state:
                    patient_state = _state_from_delta(record)
                steps.append(
                    AgentStep(
                        index=len(steps),
                        round_index=round_index if isinstance(round_index, int) else None,
                        phase=str(planner.get("current_phase") or "") or None,
                        action_text=action_text,
                        patient_state=patient_state,
                        source="dual_agent_rounds",
                        raw=dict(record),
                    )
                )
            if steps:
                return steps

    tool_calls = read_jsonl(run_path / "tool_calls.jsonl")
    if tool_calls:
        return _agent_steps_from_tool_calls(tool_calls, final_answers)

    answer_text = str(final_answers.get("answer_text", ""))
    if answer_text.strip():
        return [
            AgentStep(
                index=0,
                round_index=None,
                phase="final_answer",
                action_text=answer_text,
                patient_state={},
                source="final_answer_only",
                raw={"answer_text": answer_text},
            )
        ]
    return []


def _patient_state_by_round(run_path: Path) -> dict[int, dict[str, Any]]:
    history = read_jsonl(run_path / "patient_state_history.jsonl")
    by_round: dict[int, dict[str, Any]] = {}
    for record in history:
        if not isinstance(record, Mapping):
            continue
        if record.get("event_type") != "patient_state_updated":
            continue
        round_index = record.get("round_index")
        after = record.get("patient_state_after")
        if isinstance(round_index, int) and isinstance(after, dict):
            by_round[round_index] = dict(after)
    return by_round


def _state_from_delta(record: Mapping[str, Any]) -> dict[str, Any]:
    delta = record.get("patient_state_delta")
    if isinstance(delta, list):
        return {str(key): "updated" for key in delta}
    return {}


def _agent_steps_from_tool_calls(
    tool_calls: list[dict[str, Any]],
    final_answers: Mapping[str, Any],
) -> list[AgentStep]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for record in tool_calls:
        if not isinstance(record, Mapping):
            continue
        phase = str(record.get("phase") or "unknown")
        if phase not in grouped:
            grouped[phase] = []
            order.append(phase)
        grouped[phase].append(dict(record))

    steps: list[AgentStep] = []
    for phase in order:
        records = grouped[phase]
        action_parts: list[str] = []
        for record in records:
            skill = str(record.get("skill_name") or "")
            result = record.get("result") or {}
            summary = ""
            if isinstance(result, Mapping):
                summary = str(result.get("summary") or result.get("text") or "")
            action_parts.append(" ".join(part for part in [skill, summary] if part))
        steps.append(
            AgentStep(
                index=len(steps),
                round_index=len(steps),
                phase=phase,
                action_text="\n".join(action_parts),
                patient_state={"current_phase": phase},
                source="tool_calls_by_phase",
                raw={"phase": phase, "tool_count": len(records)},
            )
        )

    answer_text = str(final_answers.get("answer_text", ""))
    if answer_text.strip():
        steps.append(
            AgentStep(
                index=len(steps),
                round_index=len(steps),
                phase="final_answer",
                action_text=answer_text,
                patient_state={},
                source="tool_calls_by_phase",
                raw={"final_answer": True},
            )
        )
    return steps


def align_steps(
    gold_steps: list[GoldStep],
    agent_steps: list[AgentStep],
) -> tuple[list[AlignmentPair], list[int], list[int]]:
    n = len(gold_steps)
    m = len(agent_steps)
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[str, int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] - LAMBDA_GAP
        back[i][0] = ("miss_gold", i - 1, 0)
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] - LAMBDA_GAP
        back[0][j] = ("redundant_agent", 0, j - 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = dp[i - 1][j] - LAMBDA_GAP
            direction: tuple[str, int, int] = ("miss_gold", i - 1, j)

            redundant = dp[i][j - 1] - LAMBDA_GAP
            if redundant > best:
                best = redundant
                direction = ("redundant_agent", i, j - 1)

            for k in range(1, j + 1):
                match_score = _match_gold_to_agent_span(
                    gold_steps[i - 1],
                    agent_steps[j - k : j],
                )
                candidate = dp[i - 1][j - k] + match_score
                if candidate > best:
                    best = candidate
                    direction = ("match_span", i - 1, j - k)

            for span in range(1, i + 1):
                match_score = _match_gold_span_to_agent(
                    gold_steps[i - span : i],
                    agent_steps[j - 1],
                )
                candidate = dp[i - span][j - 1] + match_score
                if candidate > best:
                    best = candidate
                    direction = ("match_gold_span", i - span, j - 1)

            dp[i][j] = best
            back[i][j] = direction

    aligned_gold: set[int] = set()
    aligned_agent: set[int] = set()
    raw_pairs: list[tuple[tuple[int, ...], tuple[int, ...], float]] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if move is None:
            break
        kind, a, b = move
        if kind == "match_span":
            gold_indices = (a,)
            agent_indices = tuple(range(b, j))
            score = _match_gold_to_agent_span(gold_steps[a], agent_steps[b:j])
            raw_pairs.append((gold_indices, agent_indices, score))
            aligned_gold.update(gold_indices)
            aligned_agent.update(agent_indices)
            i = a
            j = b
        elif kind == "match_gold_span":
            gold_indices = tuple(range(a, i))
            agent_indices = (b,)
            score = _match_gold_span_to_agent(gold_steps[a:i], agent_steps[b])
            raw_pairs.append((gold_indices, agent_indices, score))
            aligned_gold.update(gold_indices)
            aligned_agent.update(agent_indices)
            i = a
            j = b
        elif kind == "miss_gold":
            i = a
            j = b
        elif kind == "redundant_agent":
            i = a
            j = b
        else:
            break
    raw_pairs.reverse()

    alignment: list[AlignmentPair] = []
    for gold_indices, agent_indices, match_score in raw_pairs:
        gi = gold_indices[0]
        aj = agent_indices[-1]
        if len(gold_indices) == 1 and len(agent_indices) == 1:
            action_set = gold_steps[gi].action_set
            agent_text = agent_steps[aj].action_text
            visible_state = gold_steps[gi].visible_state
        elif len(gold_indices) == 1:
            action_set = gold_steps[gi].action_set
            agent_text = _aggregate_agent_text(agent_steps, agent_indices)
            visible_state = gold_steps[gi].visible_state
        else:
            action_set = _merge_gold_action_sets(gold_steps, gold_indices)
            agent_text = agent_steps[aj].action_text
            visible_state = gold_steps[gold_indices[-1]].visible_state
        state_sim = state_similarity(visible_state, agent_steps[aj].patient_state)
        act_match = action_match_score(action_set, agent_text)
        alignment.append(
            AlignmentPair(
                gold_index=gi,
                agent_index=aj,
                match_score=match_score,
                state_sim=state_sim,
                act_match=act_match,
                act_score=0.0,
                step_score=0.0,
                gold_indices=gold_indices,
                agent_indices=agent_indices,
            )
        )

    miss_gold = [idx for idx in range(n) if idx not in aligned_gold]
    miss_agent = [idx for idx in range(m) if idx not in aligned_agent]
    return alignment, miss_gold, miss_agent


def _marginal_redundant_agent_steps(
    gold_steps: list[GoldStep],
    agent_steps: list[AgentStep],
    alignment: list[AlignmentPair],
) -> set[int]:
    redundant: set[int] = set()
    for pair in alignment:
        agent_indices = pair.agent_indices or (pair.agent_index,)
        if len(agent_indices) <= 1:
            continue
        gold_indices = pair.gold_indices or (pair.gold_index,)
        if len(gold_indices) == 1:
            action_set = gold_steps[gold_indices[0]].action_set
        else:
            action_set = _merge_gold_action_sets(gold_steps, gold_indices)
        covered_required: set[str] = set()
        for aj in agent_indices:
            agent_text = agent_steps[aj].action_text
            newly_covered = {
                str(action.get("action_id") or action.get("action"))
                for action in action_set.get("required") or []
                if isinstance(action, Mapping)
                and str(action.get("action_id") or action.get("action")) not in covered_required
                and _matches_action(
                    agent_text,
                    str(action.get("action", "")),
                    str(action.get("action_type", "")),
                )
            }
            if not newly_covered and not action_match_score(action_set, agent_text):
                redundant.add(aj)
            covered_required.update(newly_covered)
    return redundant


def _aggregate_agent_text(agent_steps: list[AgentStep], indices: tuple[int, ...]) -> str:
    return "\n".join(
        agent_steps[index].action_text
        for index in indices
        if agent_steps[index].action_text
    )


def _merge_gold_action_sets(
    gold_steps: list[GoldStep],
    indices: tuple[int, ...],
) -> dict[str, list[dict[str, Any]]]:
    merged = {key: [] for key in ("required", "acceptable", "conditional", "unsafe")}
    for index in indices:
        action_set = gold_steps[index].action_set
        for bucket in merged:
            merged[bucket].extend(action_set.get(bucket) or [])
    return merged


def _match_gold_to_agent_span(gold: GoldStep, agent_span: list[AgentStep]) -> float:
    if not agent_span:
        return -LAMBDA_GAP
    agent_text = "\n".join(step.action_text for step in agent_span if step.action_text)
    state_sim = state_similarity(gold.visible_state, agent_span[-1].patient_state)
    act_match = action_match_score(gold.action_set, agent_text)
    span_bonus = min(0.1, 0.03 * (len(agent_span) - 1))
    return ALPHA_STATE * state_sim + BETA_ACTION * act_match + span_bonus


def _match_gold_span_to_agent(gold_span: list[GoldStep], agent: AgentStep) -> float:
    if not gold_span:
        return -LAMBDA_GAP
    merged = _merge_gold_action_sets(gold_span, tuple(range(len(gold_span))))
    state_sim = state_similarity(gold_span[-1].visible_state, agent.patient_state)
    act_match = action_match_score(merged, agent.action_text)
    span_bonus = min(0.1, 0.03 * (len(gold_span) - 1))
    return ALPHA_STATE * state_sim + BETA_ACTION * act_match + span_bonus


def score_alignment_pairs(
    gold_steps: list[GoldStep],
    agent_steps: list[AgentStep],
    alignment: list[AlignmentPair],
) -> list[AlignmentPair]:
    scored: list[AlignmentPair] = []
    for pair in alignment:
        gold_indices = pair.gold_indices or (pair.gold_index,)
        agent_indices = pair.agent_indices or (pair.agent_index,)
        if len(gold_indices) == 1:
            action_set = gold_steps[gold_indices[0]].action_set
            visible_state = gold_steps[gold_indices[0]].visible_state
        else:
            action_set = _merge_gold_action_sets(gold_steps, gold_indices)
            visible_state = gold_steps[gold_indices[-1]].visible_state
        if len(agent_indices) == 1:
            agent_text = agent_steps[agent_indices[0]].action_text
        else:
            agent_text = _aggregate_agent_text(agent_steps, agent_indices)
        act_score_detail = compute_act_score_detail(
            action_set,
            agent_text,
            visible_state,
        )
        act_score = float(act_score_detail["score"])
        step_score = GAMMA_STATE_IN_STEP * pair.state_sim + (1 - GAMMA_STATE_IN_STEP) * act_score
        span_length = max(len(gold_indices), len(agent_indices))
        span_bonus = min(0.1, 0.03 * (span_length - 1))
        match_score_detail = {
            "state_similarity": pair.state_sim,
            "state_weight": ALPHA_STATE,
            "state_contribution": ALPHA_STATE * pair.state_sim,
            "action_match": pair.act_match,
            "action_match_weight": BETA_ACTION,
            "action_match_contribution": BETA_ACTION * pair.act_match,
            "span_length": span_length,
            "span_bonus": span_bonus,
            "score": pair.match_score,
        }
        step_score_detail = {
            "state_similarity": pair.state_sim,
            "state_weight": GAMMA_STATE_IN_STEP,
            "state_contribution": GAMMA_STATE_IN_STEP * pair.state_sim,
            "act_score": act_score,
            "act_score_weight": 1 - GAMMA_STATE_IN_STEP,
            "act_score_contribution": (1 - GAMMA_STATE_IN_STEP) * act_score,
            "score": step_score,
        }
        scored.append(
            AlignmentPair(
                gold_index=pair.gold_index,
                agent_index=pair.agent_index,
                match_score=pair.match_score,
                state_sim=pair.state_sim,
                act_match=pair.act_match,
                act_score=act_score,
                step_score=step_score,
                gold_indices=gold_indices,
                agent_indices=agent_indices,
                match_score_detail=match_score_detail,
                act_score_detail=act_score_detail,
                step_score_detail=step_score_detail,
            )
        )
    return scored


def _build_score_summary(pairs: list[AlignmentPair]) -> dict[str, Any]:
    """Aggregate every independently computed alignment score component."""

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    def extrema(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        return round(min(values), 4), round(max(values), 4)

    match_scores = [pair.match_score for pair in pairs]
    state_scores = [pair.state_sim for pair in pairs]
    action_matches = [pair.act_match for pair in pairs]
    act_scores = [pair.act_score for pair in pairs]
    step_scores = [pair.step_score for pair in pairs]
    match_min, match_max = extrema(match_scores)
    act_min, act_max = extrema(act_scores)
    step_min, step_max = extrema(step_scores)

    required_details = [pair.act_score_detail.get("required", {}) for pair in pairs]
    acceptable_details = [pair.act_score_detail.get("acceptable", {}) for pair in pairs]
    conditional_details = [pair.act_score_detail.get("conditional", {}) for pair in pairs]
    unsafe_details = [pair.act_score_detail.get("unsafe", {}) for pair in pairs]
    weighted_details = [
        pair.act_score_detail.get("weighted_components", {}) for pair in pairs
    ]

    def detail_mean(details: list[Mapping[str, Any]], key: str) -> float:
        return mean([float(detail.get(key, 0.0)) for detail in details])

    def detail_sum(details: list[Mapping[str, Any]], key: str) -> int:
        return sum(int(detail.get(key, 0)) for detail in details)

    return {
        "alignment": {
            "count": len(pairs),
            "avg_match_score": mean(match_scores),
            "min_match_score": match_min,
            "max_match_score": match_max,
            "avg_state_similarity": mean(state_scores),
            "avg_action_match": mean(action_matches),
            "avg_span_bonus": mean(
                [float(pair.match_score_detail.get("span_bonus", 0.0)) for pair in pairs]
            ),
        },
        "action": {
            "avg_act_score": mean(act_scores),
            "min_act_score": act_min,
            "max_act_score": act_max,
            "required_coverage": detail_mean(required_details, "score"),
            "required_matched_count": detail_sum(required_details, "matched_count"),
            "required_total_count": detail_sum(required_details, "total_count"),
            "acceptable_coverage": detail_mean(acceptable_details, "score"),
            "acceptable_matched_count": detail_sum(acceptable_details, "matched_count"),
            "acceptable_total_count": detail_sum(acceptable_details, "total_count"),
            "conditional_awareness": detail_mean(
                conditional_details, "awareness_score"
            ),
            "conditional_matched_count": detail_sum(
                conditional_details, "matched_count"
            ),
            "conditional_aware_count": detail_sum(
                conditional_details, "condition_aware_count"
            ),
            "conditional_total_count": detail_sum(conditional_details, "total_count"),
            "conditional_unsafe": detail_mean(conditional_details, "unsafe_score"),
            "conditional_unsafe_violation_count": detail_sum(
                conditional_details, "unsafe_violation_count"
            ),
            "unsafe_hit": detail_mean(unsafe_details, "score"),
            "unsafe_hit_count": detail_sum(unsafe_details, "hit_count"),
            "unsafe_total_count": detail_sum(unsafe_details, "total_count"),
            "avg_required_contribution": detail_mean(
                weighted_details, "required_coverage"
            ),
            "avg_acceptable_contribution": detail_mean(
                weighted_details, "acceptable_coverage"
            ),
            "avg_conditional_contribution": detail_mean(
                weighted_details, "conditional_awareness"
            ),
            "avg_unsafe_penalty": detail_mean(weighted_details, "unsafe_penalty"),
        },
        "step": {
            "avg_step_score": mean(step_scores),
            "min_step_score": step_min,
            "max_step_score": step_max,
            "avg_state_contribution": mean(
                [
                    float(pair.step_score_detail.get("state_contribution", 0.0))
                    for pair in pairs
                ]
            ),
            "avg_act_score_contribution": mean(
                [
                    float(pair.step_score_detail.get("act_score_contribution", 0.0))
                    for pair in pairs
                ]
            ),
        },
    }


def _rounded_mapping(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, Mapping):
        return {str(key): _rounded_mapping(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded_mapping(item) for item in value]
    return value


def state_similarity(
    gold_state: Mapping[str, Any],
    agent_state: Mapping[str, Any],
) -> float:
    comparable = 0
    matched = 0
    for _field, aliases in STATE_FIELD_ALIASES.items():
        gold_value = _first_present(gold_state, aliases)
        agent_value = _first_present(agent_state, aliases)
        if gold_value is None:
            continue
        comparable += 1
        if agent_value is None:
            continue
        if _values_compatible(gold_value, agent_value):
            matched += 1
    if comparable == 0:
        return 0.5
    return matched / comparable


def action_match_score(
    action_set: Mapping[str, list[dict[str, Any]]],
    agent_text: str,
) -> float:
    positive_actions: list[dict[str, Any]] = []
    for bucket in ("required", "acceptable", "conditional"):
        for action in action_set.get(bucket) or []:
            if isinstance(action, Mapping):
                positive_actions.append(dict(action))
    if not positive_actions:
        return 0.5 if agent_text.strip() else 0.0
    hits = sum(
        1
        for action in positive_actions
        if _matches_action(
            agent_text,
            str(action.get("action", "")),
            str(action.get("action_type", "")),
        )
    )
    return hits / len(positive_actions)


def compute_act_score(
    action_set: Mapping[str, list[dict[str, Any]]],
    agent_text: str,
    gold_state: Mapping[str, Any],
) -> float:
    return float(compute_act_score_detail(action_set, agent_text, gold_state)["score"])


def compute_act_score_detail(
    action_set: Mapping[str, list[dict[str, Any]]],
    agent_text: str,
    gold_state: Mapping[str, Any],
) -> dict[str, Any]:
    required = _bucket_coverage_detail(agent_text, action_set.get("required") or [])
    acceptable = _bucket_coverage_detail(
        agent_text,
        action_set.get("acceptable") or [],
    )
    conditional = _conditional_score_detail(
        agent_text,
        action_set.get("conditional") or [],
        gold_state,
    )
    unsafe = _unsafe_hit_detail(agent_text, action_set.get("unsafe") or [])
    unsafe_total = max(float(unsafe["score"]), float(conditional["unsafe_score"]))
    weighted_components = {
        "required_coverage": W_COVER * float(required["score"]),
        "acceptable_coverage": W_ACCEPT * float(acceptable["score"]),
        "conditional_awareness": W_COND * float(conditional["awareness_score"]),
        "unsafe_penalty": W_UNSAFE * unsafe_total,
    }
    score_before_clamp = (
        weighted_components["required_coverage"]
        + weighted_components["acceptable_coverage"]
        + weighted_components["conditional_awareness"]
        - weighted_components["unsafe_penalty"]
    )
    return {
        "required": required,
        "acceptable": acceptable,
        "conditional": conditional,
        "unsafe": unsafe,
        "combined_unsafe_score": unsafe_total,
        "weights": {
            "required_coverage": W_COVER,
            "acceptable_coverage": W_ACCEPT,
            "conditional_awareness": W_COND,
            "unsafe_penalty": W_UNSAFE,
        },
        "weighted_components": weighted_components,
        "score_before_clamp": score_before_clamp,
        "score": max(0.0, min(1.0, score_before_clamp)),
    }


def _bucket_coverage(agent_text: str, actions: list[Any]) -> float:
    return float(_bucket_coverage_detail(agent_text, actions)["score"])


def _bucket_coverage_detail(agent_text: str, actions: list[Any]) -> dict[str, Any]:
    if not actions:
        return {
            "score": 1.0,
            "matched_count": 0,
            "total_count": 0,
            "matched_action_ids": [],
        }
    matched_action_ids: list[str] = []
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        if _matches_action(
            agent_text,
            str(action.get("action", "")),
            str(action.get("action_type", "")),
        ):
            matched_action_ids.append(_action_identifier(action))
    return {
        "score": len(matched_action_ids) / len(actions),
        "matched_count": len(matched_action_ids),
        "total_count": len(actions),
        "matched_action_ids": matched_action_ids,
    }


def _conditional_scores(
    agent_text: str,
    actions: list[Any],
    gold_state: Mapping[str, Any],
) -> tuple[float, float]:
    detail = _conditional_score_detail(agent_text, actions, gold_state)
    return float(detail["awareness_score"]), float(detail["unsafe_score"])


def _conditional_score_detail(
    agent_text: str,
    actions: list[Any],
    gold_state: Mapping[str, Any],
) -> dict[str, Any]:
    if not actions:
        return {
            "awareness_score": 1.0,
            "unsafe_score": 0.0,
            "matched_count": 0,
            "condition_aware_count": 0,
            "unsafe_violation_count": 0,
            "total_count": 0,
            "matched_action_ids": [],
            "condition_aware_action_ids": [],
            "unsafe_violation_action_ids": [],
        }
    matched_action_ids: list[str] = []
    condition_aware_action_ids: list[str] = []
    unsafe_violation_action_ids: list[str] = []
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        action_text = str(action.get("action", ""))
        action_type = str(action.get("action_type", ""))
        if not _matches_action(agent_text, action_text, action_type):
            continue
        action_id = _action_identifier(action)
        matched_action_ids.append(action_id)
        if _has_condition_expression(agent_text, action):
            condition_aware_action_ids.append(action_id)
        elif _is_unconditional_conditional_violation(action, gold_state):
            unsafe_violation_action_ids.append(action_id)
    return {
        "awareness_score": len(condition_aware_action_ids) / len(actions),
        "unsafe_score": 1.0 if unsafe_violation_action_ids else 0.0,
        "matched_count": len(matched_action_ids),
        "condition_aware_count": len(condition_aware_action_ids),
        "unsafe_violation_count": len(unsafe_violation_action_ids),
        "total_count": len(actions),
        "matched_action_ids": matched_action_ids,
        "condition_aware_action_ids": condition_aware_action_ids,
        "unsafe_violation_action_ids": unsafe_violation_action_ids,
    }


def _has_condition_expression(agent_text: str, action: Mapping[str, Any]) -> bool:
    lowered = agent_text.lower()
    if any(marker in lowered for marker in CONDITION_MARKERS):
        return True
    condition = str(action.get("condition") or "")
    if condition:
        tokens = re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", condition.lower())
        if tokens and any(token in lowered for token in tokens[:3]):
            return True
    return False


def _is_unconditional_conditional_violation(
    action: Mapping[str, Any],
    gold_state: Mapping[str, Any],
) -> bool:
    action_type = str(action.get("action_type", "")).lower()
    action_text = str(action.get("action", "")).lower()
    if action_type in {"treatment_planning", "adjuvant_therapy", "targeted_therapy"}:
        return True
    treatment_tokens = ("osimertinib", "alectinib", "adjuvant", "chemotherapy", "immunotherapy", "靶向", "辅助", "免疫")
    if any(token in action_text for token in treatment_tokens):
        required_evidence = action.get("required_evidence") or []
        if isinstance(required_evidence, list):
            for key in required_evidence:
                value = _first_present(gold_state, (str(key),))
                if value is None or str(value).lower() in {"missing", "unknown", ""}:
                    return True
    return False


def _unsafe_hit(agent_text: str, actions: list[Any]) -> float:
    return float(_unsafe_hit_detail(agent_text, actions)["score"])


def _unsafe_hit_detail(agent_text: str, actions: list[Any]) -> dict[str, Any]:
    hit_action_ids: list[str] = []
    for action in actions:
        if not isinstance(action, Mapping):
            continue
        if _matches_unsafe_action(agent_text, str(action.get("action", ""))):
            hit_action_ids.append(_action_identifier(action))
    return {
        "score": 1.0 if hit_action_ids else 0.0,
        "hit_count": len(hit_action_ids),
        "total_count": len(actions),
        "hit_action_ids": hit_action_ids,
    }


def _action_identifier(action: Mapping[str, Any]) -> str:
    return str(action.get("action_id") or action.get("action") or "")


def compute_coherence(
    gold_steps: list[GoldStep],
    agent_steps: list[AgentStep],
    alignment: list[AlignmentPair],
) -> dict[str, Any]:
    issues: list[str] = []
    deductions: list[dict[str, Any]] = []
    score = 1.0

    # Repetition without progress
    seen_actions: dict[str, int] = {}
    prev_state_keys: set[str] = set()
    for agent in agent_steps:
        normalized = _normalized(agent.action_text)
        if normalized:
            seen_actions[normalized] = seen_actions.get(normalized, 0) + 1
        current_keys = set(str(key) for key in agent.patient_state.keys())
        if normalized and seen_actions[normalized] > 1 and current_keys == prev_state_keys:
            issues.append("repeated_action_without_state_progress")
            deductions.append(
                {
                    "issue": "repeated_action_without_state_progress",
                    "amount": 0.15,
                    "agent_index": agent.index,
                    "round_index": agent.round_index,
                }
            )
            score -= 0.15
        prev_state_keys = current_keys

    # Treatment jump without prior diagnostic alignment
    diagnostic_phases = {"diagnosis", "staging", "pathology", "molecular", "postoperative"}
    aligned_phases = [
        (gold_steps[pair.gold_index].phase or "").lower()
        for pair in alignment
    ]
    treatment_mentions = any(
        token in _normalized(agent.action_text)
        for agent in agent_steps
        for token in ("adjuvant", "chemotherapy", "osimertinib", "radiation", "辅助", "化疗", "放疗")
    )
    has_diagnostic_alignment = any(
        any(token in phase for token in diagnostic_phases) for phase in aligned_phases
    )
    if treatment_mentions and not has_diagnostic_alignment and len(gold_steps) > 1:
        issues.append("possible_treatment_without_diagnostic_alignment")
        deductions.append(
            {
                "issue": "possible_treatment_without_diagnostic_alignment",
                "amount": 0.2,
            }
        )
        score -= 0.2

    # Deferred action treated as confirmed
    for pair in alignment:
        gold = gold_steps[pair.gold_index]
        agent = agent_steps[pair.agent_index]
        for action in gold.action_set.get("conditional") or []:
            if not isinstance(action, Mapping):
                continue
            action_text = str(action.get("action", ""))
            action_type = str(action.get("action_type", ""))
            if _matches_action(agent.action_text, action_text, action_type):
                if not _has_condition_expression(agent.action_text, action):
                    if _is_unconditional_conditional_violation(action, gold.visible_state):
                        issues.append("deferred_action_treated_as_confirmed")
                        deductions.append(
                            {
                                "issue": "deferred_action_treated_as_confirmed",
                                "amount": 0.25,
                                "gold_index": pair.gold_index,
                                "agent_index": pair.agent_index,
                                "action_id": _action_identifier(action),
                            }
                        )
                        score -= 0.25
                        break

    # State contradiction across aligned steps
    last_tnm: str | None = None
    for pair in alignment:
        tnm = _first_present(
            agent_steps[pair.agent_index].patient_state,
            STATE_FIELD_ALIASES["tnm"],
        )
        if tnm is not None:
            tnm_text = str(tnm).lower()
            if last_tnm and last_tnm != tnm_text and "unknown" not in {last_tnm, tnm_text}:
                issues.append("contradictory_stage_across_steps")
                deductions.append(
                    {
                        "issue": "contradictory_stage_across_steps",
                        "amount": 0.2,
                        "gold_index": pair.gold_index,
                        "agent_index": pair.agent_index,
                        "previous_tnm": last_tnm,
                        "current_tnm": tnm_text,
                    }
                )
                score -= 0.2
                break
            last_tnm = tnm_text

    score_before_clamp = score
    score = max(0.0, min(1.0, score))
    return {
        "score": score,
        "score_before_clamp": score_before_clamp,
        "initial_score": 1.0,
        "total_deduction": sum(float(item["amount"]) for item in deductions),
        "issues": sorted(set(issues)),
        "deductions": deductions,
    }


def _first_present(state: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in state and state[key] not in (None, "", [], {}):
            return state[key]
        if key == "known_biomarkers" and isinstance(state.get("known_biomarkers"), dict):
            biomarkers = state["known_biomarkers"]
            if biomarkers:
                return biomarkers
        if key == "known_stage" and state.get("known_stage"):
            return state["known_stage"]
    return None


def _values_compatible(left: Any, right: Any) -> bool:
    left_text = _normalized(str(left))
    right_text = _normalized(str(right))
    if not left_text or not right_text:
        return False
    if left_text == right_text:
        return True
    if left_text in right_text or right_text in left_text:
        return True
    left_tokens = set(re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", left_text))
    right_tokens = set(re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", right_text))
    overlap = left_tokens & right_tokens
    return len(overlap) >= min(2, len(left_tokens), len(right_tokens))


def _missing_result(run_dir: Path, reason: str) -> dict[str, Any]:
    result = {
        "status": "missing",
        "score_type": "dyn_traj_alignment_v1",
        "macro_trajectory_total": None,
        "reason": reason,
    }
    write_json(run_dir / "trajectory_scores.json", result)
    return result
