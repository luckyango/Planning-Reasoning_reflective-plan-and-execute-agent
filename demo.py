from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plan_and_execute_agent import AgentState


DEFAULT_TASK = (
    "Compare LangGraph, CrewAI, and AutoGen for building an internal enterprise "
    "knowledge-base agent. Recommend one framework and explain the tradeoffs."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reflective plan-and-execute agent demo."
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        help="Task for the agent to solve.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1",
        help="Main model used by composer, planner, executor, and synthesizer.",
    )
    parser.add_argument(
        "--critic-model",
        default="gpt-4.1-mini",
        help="Model used by critic, reflector, and path evaluator roles.",
    )
    parser.add_argument(
        "--candidate-plans",
        type=int,
        default=3,
        help="Number of candidate reasoning paths to generate before execution.",
    )
    parser.add_argument(
        "--max-replans",
        type=int,
        default=3,
        help="Maximum number of replans allowed during execution.",
    )
    parser.add_argument(
        "--trace-out",
        type=Path,
        default=None,
        help="Optional path for writing a full JSON trace.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from plan_and_execute_agent import PlanAndExecuteAgent

    agent = PlanAndExecuteAgent(
        model=args.model,
        critic_model=args.critic_model,
        max_replans=args.max_replans,
        candidate_plan_count=args.candidate_plans,
    )

    final_answer = agent.run(args.task)
    state = agent.last_state
    if state is None:
        raise RuntimeError("Agent finished without a recorded state.")

    print_demo_summary(state, final_answer)

    if args.trace_out:
        write_trace(args.trace_out, state)
        print(f"\nTrace written to: {args.trace_out}")


def print_demo_summary(state: "AgentState", final_answer: str) -> None:
    print("\n=== Demo Summary ===")
    print(f"Task type: {state.task.task_type}")
    print(f"Goal: {state.task.goal}")

    print("\nCandidate reasoning paths:")
    for path in state.candidate_paths:
        evaluation = path.evaluation
        score = evaluation.total_score if evaluation else None
        marker = "selected" if path.selected else "not selected"
        print(f"- {path.path_id}: {path.strategy} | score={score} | {marker}")

    print("\nExecution trace:")
    for index, result in enumerate(state.executed_steps, 1):
        critique = next(
            (
                item
                for item in state.critiques
                if item.step_id == result.step.id
            ),
            None,
        )
        score = critique.quality_score if critique else None
        print(f"- Step {index}: {result.step.description} | quality={score}")

    print("\nWorking memory:")
    memory = state.working_memory.to_dict()
    for category, items in memory.items():
        print(f"- {category}: {len(items)} item(s)")

    print("\nFinal answer:")
    print(final_answer)


def write_trace(path: Path, state: "AgentState") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(state)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
