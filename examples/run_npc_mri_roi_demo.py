"""Run the Qwen-driven NPC MRI ROI demo for the synthetic NPC-DEMO case."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.run_qwen_agent import build_qwen_agent
from medclaw.core.agent_loop import AgentLoopError
from medclaw.llm.protocols import LLMAPIError, LLMConfigurationError
from medclaw.stores.artifact_store import ArtifactStore


CASE_ID = "NPC-DEMO"
TARGET_SKILL = "radiology.npc_mri_roi"
DEFAULT_PROMPT = (
    "For the current nasopharyngeal carcinoma case, use the available NPC MRI ROI "
    "tool to export T1C primary-tumor review images, inspect the generated slice "
    "and overlay PNG artifacts, then summarize what was produced. Clearly state "
    "that this is research model output rather than clinical advice."
)


def run_demo(
    *,
    project_root: Path = PROJECT_ROOT,
    prompt: str = DEFAULT_PROMPT,
    runs_root: Path | None = None,
    artifacts_root: Path | None = None,
) -> dict[str, Any]:
    """Run one Qwen turn and require the model to choose the NPC MRI ROI skill."""

    project_root = Path(project_root).resolve()
    runs_root = Path(runs_root or project_root / "runs").resolve()
    artifacts_root = Path(artifacts_root or project_root / "artifacts").resolve()
    agent, config = build_qwen_agent(
        project_root=project_root,
        case_id=CASE_ID,
        runs_root=runs_root,
        artifacts_root=artifacts_root,
    )
    turn = agent.chat(prompt)

    target_records = [
        record for record in turn.tool_results if record.get("skill_name") == TARGET_SKILL
    ]
    if not target_records:
        raise RuntimeError(
            f"Qwen did not call the required skill {TARGET_SKILL!r} with tool_choice=auto."
        )
    result = target_records[-1]["result"]
    if result.get("status") != "success":
        raise RuntimeError(f"{TARGET_SKILL} failed: {result}")

    artifact_store = ArtifactStore(artifacts_root)
    artifact_paths = [
        str(artifact_store.resolve_uri(artifact["uri"]))
        for artifact in result.get("artifacts", [])
        if isinstance(artifact, dict) and isinstance(artifact.get("uri"), str)
    ]
    return {
        "case_id": CASE_ID,
        "model": config.public_summary(),
        "answer": turn.content,
        "tool_result": result,
        "evidence_board": agent.evidence_board.summary(),
        "evidence_board_path": str(agent.evidence_board.path),
        "audit_log_path": str(runs_root / CASE_ID / "audit_log.jsonl"),
        "conversation_log_path": str(agent.conversation_log_path),
        "turn_id": turn.turn_id,
        "artifact_paths": artifact_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    try:
        demo = run_demo(prompt=args.prompt)
    except (
        AgentLoopError,
        OSError,
        LLMAPIError,
        LLMConfigurationError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Final Qwen answer:")
    print(demo["answer"])
    print("\nEvidence board:")
    print(json.dumps(demo["evidence_board"], ensure_ascii=False, indent=2))
    print(f"\nAudit log: {demo['audit_log_path']}")
    print(f"Conversation audit log: {demo['conversation_log_path']}")
    print(f"Conversation turn: {demo['turn_id']}")
    print("Artifacts:")
    for path in demo["artifact_paths"]:
        print(f"  - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
