"""Chat with a locally served or API-backed MedClaw agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from medclaw.core.agent_loop import AgentLoop, AgentLoopError
from medclaw.core.evidence_board import EvidenceBoard
from medclaw.core.tool_router import ToolRouter
from medclaw.llm.factory import (
    DEFAULT_PROVIDER,
    SUPPORTED_PROVIDERS,
    create_llm_client,
    load_config_from_env,
    normalize_provider,
    resolve_provider,
)
from medclaw.llm.protocols import LLMAPIError, LLMConfigurationError
from medclaw.registry.skill_registry import SkillRegistry
from medclaw.runtime.runtime_manager import RuntimeManager
from medclaw.runtime.skill_runner import SkillRunner
from medclaw.stores.artifact_store import ArtifactStore
from medclaw.stores.audit_log import AuditLog
from medclaw.stores.case_store import CaseStore
from medclaw.stores.conversation_log import ConversationLog


def build_qwen_agent(
    *,
    project_root: Path = PROJECT_ROOT,
    case_id: str = "TCGA-38-4626",
    config: Any | None = None,
    provider: str = DEFAULT_PROVIDER,
    runs_root: Path | None = None,
    artifacts_root: Path | None = None,
) -> tuple[AgentLoop, Any]:
    """Build a model-driven MedClaw agent for one benchmark case."""

    project_root = Path(project_root).resolve()
    runs_root = Path(runs_root or project_root / "runs").resolve()
    artifacts_root = Path(artifacts_root or project_root / "artifacts").resolve()

    case = CaseStore(project_root / "examples" / "cases").load(case_id)
    registry = SkillRegistry(project_root / "medclaw" / "skills")
    artifact_store = ArtifactStore(artifacts_root)
    audit_log = AuditLog(runs_root)
    conversation_log = ConversationLog(runs_root)
    runtime_manager = RuntimeManager(
        registry,
        SkillRunner(),
        artifact_store,
        audit_log,
        runs_root,
    )
    evidence_board = EvidenceBoard(case_id, runs_root, load_existing=True)

    provider = normalize_provider(provider)
    llm_config = config or load_config_from_env(provider)
    system_prompt = _system_prompt(case)
    agent = AgentLoop(
        ToolRouter(registry, runtime_manager),
        evidence_board,
        llm_client=create_llm_client(provider, config=llm_config),
        system_prompt=system_prompt,
        artifact_uri_resolver=artifact_store.resolve_uri,
        conversation_log=conversation_log,
    )
    return agent, llm_config


def _system_prompt(case: dict[str, Any]) -> str:
    return f"""You are MedClaw, a multimodal medical benchmark agent.
You can directly inspect images supplied by the user and call registered tools
to collect reproducible evidence. Use tools when they add evidence, never invent
tool results, distinguish mock evidence from real evidence, and state important
uncertainty. This runtime is for research benchmarking and must not present its
output as clinical advice.

Current benchmark case:
{json.dumps(case, ensure_ascii=False, sort_keys=True)}

Always pass the exact current case_id to tools that require it."""


def _run_turn(
    agent: AgentLoop,
    message: str,
    *,
    image_paths: list[Path],
    image_urls: list[str],
) -> None:
    result = agent.chat(
        message,
        image_paths=image_paths,
        image_urls=image_urls,
    )
    print(f"\nMedClaw> {result.content}")
    print(f"\nAudit turn: {result.turn_id}")
    if result.tool_results:
        names = [
            record["skill_name"] or record["function_name"] or "unknown"
            for record in result.tool_results
        ]
        print(f"\nTools used: {', '.join(names)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default="TCGA-38-4626")
    parser.add_argument(
        "--provider",
        choices=list(SUPPORTED_PROVIDERS),
        default=resolve_provider(None),
    )
    parser.add_argument("--message", help="Run one turn instead of interactive chat.")
    parser.add_argument(
        "--image",
        action="append",
        type=Path,
        default=[],
        help="Attach a local image to the first turn. May be repeated.",
    )
    parser.add_argument(
        "--image-url",
        action="append",
        default=[],
        help="Attach an HTTPS image URL to the first turn. May be repeated.",
    )
    args = parser.parse_args()

    try:
        agent, config = build_qwen_agent(
            case_id=args.case_id,
            provider=args.provider,
        )
        print(
            "LLM configuration: "
            + json.dumps(config.public_summary(), ensure_ascii=False, sort_keys=True)
        )
        print(f"Conversation audit log: {agent.conversation_log_path}")

        if args.message:
            _run_turn(
                agent,
                args.message,
                image_paths=args.image,
                image_urls=args.image_url,
            )
            return 0

        print("Interactive chat started. Type /exit to stop.")
        first_images = args.image
        first_image_urls = args.image_url
        while True:
            try:
                message = input("\nYou> ").strip()
            except EOFError:
                break
            if message.lower() in {"/exit", "/quit"}:
                break
            if not message and not first_images and not first_image_urls:
                continue
            _run_turn(
                agent,
                message,
                image_paths=first_images,
                image_urls=first_image_urls,
            )
            first_images = []
            first_image_urls = []
        return 0
    except (AgentLoopError, OSError, LLMAPIError, LLMConfigurationError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
