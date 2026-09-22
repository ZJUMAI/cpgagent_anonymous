"""Deterministic hierarchical sampling for V2 guideline training."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class SamplingConfigurationError(ValueError):
    pass


def largest_remainder_counts(
    total: int,
    ratios: Mapping[str, float],
) -> dict[str, int]:
    """Allocate exactly ``total`` items according to non-negative ratios."""

    if total < 0:
        raise SamplingConfigurationError("total must be non-negative.")
    cleaned = {str(key): max(float(value), 0.0) for key, value in ratios.items()}
    denominator = sum(cleaned.values())
    if denominator <= 0:
        raise SamplingConfigurationError("At least one sampling ratio must be positive.")
    exact = {key: total * value / denominator for key, value in cleaned.items()}
    counts = {key: math.floor(value) for key, value in exact.items()}
    remainder = total - sum(counts.values())
    order = sorted(
        cleaned,
        key=lambda key: (-(exact[key] - counts[key]), key),
    )
    for key in order[:remainder]:
        counts[key] += 1
    return counts


@dataclass(frozen=True)
class SamplingAudit:
    requested_task_counts: dict[str, int]
    realized_task_counts: dict[str, int]
    realized_group_counts: dict[str, int]
    world_size: int
    rank: int
    seed: int

    @property
    def exact(self) -> bool:
        return self.requested_task_counts == self.realized_task_counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_task_counts": dict(self.requested_task_counts),
            "realized_task_counts": dict(self.realized_task_counts),
            "realized_group_counts": dict(self.realized_group_counts),
            "world_size": self.world_size,
            "rank": self.rank,
            "seed": self.seed,
            "exact": self.exact,
        }


class DeterministicTaskSampler:
    """Exact task schedule plus square-root group-temperature sampling.

    A global schedule is created first, then sharded by rank. This makes the
    configured task ratios describe the actual distributed examples rather than
    metadata attached to records.
    """

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        task_ratios: Mapping[str, float],
        seed: int = 17,
        group_temperature: float = 0.5,
    ) -> None:
        self.records = [dict(record) for record in records]
        self.task_ratios = {
            str(key): float(value) for key, value in task_ratios.items() if float(value) > 0
        }
        self.seed = int(seed)
        self.group_temperature = float(group_temperature)
        if not self.records:
            raise SamplingConfigurationError("Cannot sample an empty record set.")
        if not 0.0 <= self.group_temperature <= 1.0:
            raise SamplingConfigurationError("group_temperature must be between 0 and 1.")
        self._task_groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for record in self.records:
            task = str(record.get("task") or "")
            if task not in self.task_ratios:
                continue
            self._task_groups[task][_record_group(record)].append(record)
        missing = [task for task in self.task_ratios if not self._task_groups.get(task)]
        if missing:
            raise SamplingConfigurationError(
                "No records exist for configured tasks: " + ", ".join(sorted(missing))
            )

    def sample(
        self,
        steps_per_rank: int,
        *,
        world_size: int = 1,
        rank: int = 0,
        epoch: int = 0,
    ) -> tuple[list[dict[str, Any]], SamplingAudit]:
        if steps_per_rank <= 0:
            return [], SamplingAudit({}, {}, {}, world_size, rank, self.seed)
        if world_size <= 0 or not 0 <= rank < world_size:
            raise SamplingConfigurationError("Invalid distributed world_size/rank.")
        global_steps = int(steps_per_rank) * int(world_size)
        requested = largest_remainder_counts(global_steps, self.task_ratios)
        task_schedule = [task for task, count in requested.items() for _ in range(count)]
        rng = random.Random(self.seed + int(epoch) * 1_000_003)
        rng.shuffle(task_schedule)
        global_records = self._materialize_schedule(task_schedule, rng)
        local_records = global_records[rank::world_size]
        if len(local_records) != steps_per_rank:
            raise SamplingConfigurationError("Distributed sampler produced the wrong local length.")
        global_task_counts = Counter(str(record.get("task") or "") for record in global_records)
        global_group_counts = Counter(_record_group(record) for record in global_records)
        audit = SamplingAudit(
            requested_task_counts=requested,
            realized_task_counts=dict(sorted(global_task_counts.items())),
            realized_group_counts=dict(sorted(global_group_counts.items())),
            world_size=world_size,
            rank=rank,
            seed=self.seed,
        )
        if not audit.exact:
            raise SamplingConfigurationError(
                "Realized task counts do not match the exact requested schedule."
            )
        return local_records, audit

    def _materialize_schedule(
        self,
        task_schedule: Sequence[str],
        rng: random.Random,
    ) -> list[dict[str, Any]]:
        pools: dict[tuple[str, str], list[dict[str, Any]]] = {}
        offsets: Counter[tuple[str, str]] = Counter()
        for task, groups in self._task_groups.items():
            for group, records in groups.items():
                pool = list(records)
                rng.shuffle(pool)
                pools[(task, group)] = pool
        result = []
        for task in task_schedule:
            groups = self._task_groups[task]
            names = sorted(groups)
            weights = [len(groups[name]) ** self.group_temperature for name in names]
            group = rng.choices(names, weights=weights, k=1)[0]
            key = (task, group)
            pool = pools[key]
            offset = offsets[key]
            if offset and offset % len(pool) == 0:
                rng.shuffle(pool)
            result.append(dict(pool[offset % len(pool)]))
            offsets[key] += 1
        return result


def balanced_validation_records(
    records: Sequence[Mapping[str, Any]],
    max_examples: int,
    *,
    seed: int = 17,
) -> list[dict[str, Any]]:
    """Select validation examples round-robin across task and cancer/guideline group."""

    rows = [dict(record) for record in records]
    if max_examples <= 0 or max_examples >= len(rows):
        return rows
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("task") or "unknown"), _record_group(row))].append(row)
    rng = random.Random(seed)
    for pool in grouped.values():
        rng.shuffle(pool)
    keys = sorted(grouped)
    selected: list[dict[str, Any]] = []
    while len(selected) < max_examples:
        progressed = False
        for key in keys:
            pool = grouped[key]
            if not pool:
                continue
            selected.append(pool.pop())
            progressed = True
            if len(selected) == max_examples:
                break
        if not progressed:
            break
    return selected


def realized_distribution(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    return {
        "tasks": dict(sorted(Counter(str(row.get("task") or "unknown") for row in rows).items())),
        "groups": dict(sorted(Counter(_record_group(row) for row in rows).items())),
    }


class PlannerTrajectorySampler:
    """Deterministically balance Planner records by cancer, then phase/action bucket."""

    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        seed: int = 17,
    ) -> None:
        self.records = [dict(record) for record in records]
        self.seed = int(seed)
        if not self.records:
            raise SamplingConfigurationError("Cannot sample an empty Planner dataset.")
        self._families: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for record in self.records:
            state = record.get("state_before") or record.get("patient_state") or {}
            if not isinstance(state, Mapping):
                raise SamplingConfigurationError("Planner record state must be an object.")
            family = str(state.get("cancer_family") or "")
            phase = str(state.get("current_phase") or "")
            action_set = record.get("action_set") or {}
            if not family or not phase or not isinstance(action_set, Mapping):
                raise SamplingConfigurationError(
                    "Planner records require cancer_family, current_phase, and action_set."
                )
            buckets = [
                bucket
                for bucket in ("required", "acceptable", "conditional", "premature", "unsafe")
                if action_set.get(bucket)
            ]
            if not buckets:
                raise SamplingConfigurationError("Planner record has no action supervision bucket.")
            for bucket in buckets:
                self._families[family][f"{phase}/{bucket}"].append(record)

    def sample(
        self,
        steps_per_rank: int,
        *,
        world_size: int = 1,
        rank: int = 0,
        epoch: int = 0,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if world_size <= 0 or not 0 <= rank < world_size:
            raise SamplingConfigurationError("Invalid distributed world_size/rank.")
        total = int(steps_per_rank) * int(world_size)
        families = sorted(self._families)
        family_counts = largest_remainder_counts(total, {name: 1.0 for name in families})
        schedule = [name for name, count in family_counts.items() for _ in range(count)]
        rng = random.Random(self.seed + int(epoch) * 1_000_003)
        rng.shuffle(schedule)
        pools: dict[tuple[str, str], list[dict[str, Any]]] = {}
        offsets: Counter[tuple[str, str]] = Counter()
        group_offsets: Counter[str] = Counter()
        for family, groups in self._families.items():
            for group, values in groups.items():
                pool = list(values)
                rng.shuffle(pool)
                pools[(family, group)] = pool
        global_rows = []
        selected_groups = []
        for family in schedule:
            groups = sorted(self._families[family])
            group = groups[group_offsets[family] % len(groups)]
            group_offsets[family] += 1
            key = (family, group)
            pool = pools[key]
            offset = offsets[key]
            if offset and offset % len(pool) == 0:
                rng.shuffle(pool)
            global_rows.append(dict(pool[offset % len(pool)]))
            selected_groups.append(f"{family}/{group}")
            offsets[key] += 1
        local_rows = global_rows[rank::world_size]
        if len(local_rows) != int(steps_per_rank):
            raise SamplingConfigurationError("Planner rank shard has the wrong length.")
        realized_families = Counter(
            str((row.get("state_before") or row.get("patient_state"))["cancer_family"])
            for row in global_rows
        )
        if dict(realized_families) != family_counts:
            raise SamplingConfigurationError("Planner cancer balance did not match its schedule.")
        return local_rows, {
            "requested_family_counts": family_counts,
            "realized_family_counts": dict(sorted(realized_families.items())),
            "realized_phase_bucket_counts": dict(sorted(Counter(selected_groups).items())),
            "world_size": world_size,
            "rank": rank,
            "seed": self.seed,
            "exact": True,
        }


def _record_group(record: Mapping[str, Any]) -> str:
    state = record.get("patient_state") or record.get("state_before") or {}
    if not isinstance(state, Mapping):
        state = {}
    guideline_context = record.get("guideline_context") or state.get(
        "guideline_context"
    )
    guideline = None
    if isinstance(guideline_context, Mapping):
        candidates = guideline_context.get("guidelines")
        if isinstance(candidates, list) and candidates and isinstance(candidates[0], Mapping):
            guideline = candidates[0].get("guideline_id")
    cancer = str(
        record.get("cancer_family")
        or record.get("cancer_type")
        or state.get("cancer_family")
        or "unknown"
    )
    guideline_id = str(record.get("guideline_id") or guideline or "unknown")
    return f"{cancer}/{guideline_id}"
