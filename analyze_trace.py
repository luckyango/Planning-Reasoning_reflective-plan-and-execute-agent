from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a reflective plan-and-execute agent trace."
    )
    parser.add_argument(
        "trace_path",
        type=Path,
        help="Path to a JSON trace produced by demo.py --trace-out.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the analysis as JSON instead of a text report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace = load_trace(args.trace_path)
    summary = analyze_trace(trace)

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(format_summary(summary))


def load_trace(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("Trace file must contain a JSON object.")
    return payload


def analyze_trace(trace: dict[str, Any]) -> dict[str, Any]:
    task = trace.get("task", {})
    candidate_paths = trace.get("candidate_paths", [])
    selected_path = find_selected_path(trace)
    executed_steps = trace.get("executed_steps", [])
    critiques = trace.get("critiques", [])
    reflections = trace.get("reflections", [])
    working_memory = trace.get("working_memory", {})
    trace_events = trace.get("trace", [])

    return {
        "task": {
            "type": task.get("task_type"),
            "goal": task.get("goal"),
            "constraints": task.get("constraints", []),
            "success_criteria": task.get("success_criteria", []),
        },
        "selected_path": summarize_path(selected_path),
        "candidate_paths": [summarize_path(path) for path in candidate_paths],
        "execution": {
            "executed_step_count": len(executed_steps),
            "retry_count": count_events(trace_events, "step_retry_scheduled"),
            "replan_count": trace.get("replan_count", 0),
            "final_answer_present": bool(trace.get("final_answer")),
        },
        "critic": summarize_critiques(critiques),
        "reflections": summarize_reflections(reflections),
        "working_memory": summarize_working_memory(working_memory),
        "trace_event_count": len(trace_events),
    }


def find_selected_path(trace: dict[str, Any]) -> dict[str, Any] | None:
    selected_path = trace.get("selected_path")
    if isinstance(selected_path, dict):
        return selected_path

    for path in trace.get("candidate_paths", []):
        if isinstance(path, dict) and path.get("selected"):
            return path

    return None


def summarize_path(path: dict[str, Any] | None) -> dict[str, Any] | None:
    if not path:
        return None

    evaluation = path.get("evaluation") or {}
    steps = path.get("steps") or []
    return {
        "path_id": path.get("path_id"),
        "strategy": path.get("strategy"),
        "selected": bool(path.get("selected")),
        "step_count": len(steps),
        "total_score": evaluation.get("total_score"),
        "goal_alignment_score": evaluation.get("goal_alignment_score"),
        "feasibility_score": evaluation.get("feasibility_score"),
        "evidence_potential_score": evaluation.get("evidence_potential_score"),
        "risk_score": evaluation.get("risk_score"),
        "strengths": evaluation.get("strengths", []),
        "weaknesses": evaluation.get("weaknesses", []),
    }


def summarize_critiques(critiques: list[dict[str, Any]]) -> dict[str, Any]:
    quality_scores = collect_scores(critiques, "quality_score")
    alignment_scores = collect_scores(critiques, "goal_alignment_score")
    evidence_scores = collect_scores(critiques, "evidence_score")
    issues = [
        issue
        for critique in critiques
        for issue in critique.get("issues", [])
        if issue
    ]

    return {
        "critique_count": len(critiques),
        "average_quality_score": average_or_none(quality_scores),
        "average_goal_alignment_score": average_or_none(alignment_scores),
        "average_evidence_score": average_or_none(evidence_scores),
        "issue_count": len(issues),
        "issues": issues,
    }


def summarize_reflections(reflections: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "reflection_count": len(reflections),
        "lessons": [
            reflection.get("lesson")
            for reflection in reflections
            if reflection.get("lesson")
        ],
        "failure_modes": [
            reflection.get("failure_mode")
            for reflection in reflections
            if reflection.get("failure_mode")
        ],
        "correction_strategies": [
            reflection.get("correction_strategy")
            for reflection in reflections
            if reflection.get("correction_strategy")
        ],
    }


def summarize_working_memory(working_memory: dict[str, Any]) -> dict[str, Any]:
    categories = ["observations", "decisions", "failed_attempts", "lessons"]
    return {
        category: len(working_memory.get(category, []))
        for category in categories
    }


def collect_scores(items: list[dict[str, Any]], key: str) -> list[float]:
    scores = []
    for item in items:
        value = item.get(key)
        if isinstance(value, (int, float)):
            scores.append(float(value))
    return scores


def average_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(mean(values), 3)


def count_events(events: list[dict[str, Any]], event_type: str) -> int:
    return sum(1 for event in events if event.get("event_type") == event_type)


def format_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Reasoning Run Summary",
        "=====================",
        "",
        f"Task type: {summary['task'].get('type')}",
        f"Goal: {summary['task'].get('goal')}",
        "",
    ]

    selected_path = summary.get("selected_path") or {}
    lines.extend(
        [
            "Selected reasoning path:",
            (
                f"- {selected_path.get('path_id')} / {selected_path.get('strategy')} "
                f"(score={selected_path.get('total_score')})"
            ),
            "",
            "Candidate paths:",
        ]
    )

    for path in summary.get("candidate_paths", []):
        marker = "selected" if path.get("selected") else "not selected"
        lines.append(
            f"- {path.get('path_id')}: {path.get('strategy')} | "
            f"score={path.get('total_score')} | {marker}"
        )

    execution = summary["execution"]
    critic = summary["critic"]
    memory = summary["working_memory"]
    lines.extend(
        [
            "",
            "Execution:",
            f"- Executed steps: {execution['executed_step_count']}",
            f"- Retries: {execution['retry_count']}",
            f"- Replans: {execution['replan_count']}",
            f"- Final answer present: {execution['final_answer_present']}",
            "",
            "Critic scores:",
            f"- Average quality: {critic['average_quality_score']}",
            f"- Average goal alignment: {critic['average_goal_alignment_score']}",
            f"- Average evidence: {critic['average_evidence_score']}",
            f"- Issues: {critic['issue_count']}",
            "",
            "Working memory:",
            f"- Observations: {memory['observations']}",
            f"- Decisions: {memory['decisions']}",
            f"- Failed attempts: {memory['failed_attempts']}",
            f"- Lessons: {memory['lessons']}",
        ]
    )

    lessons = summary["reflections"]["lessons"]
    if lessons:
        lines.extend(["", "Key lessons:"])
        lines.extend(f"- {lesson}" for lesson in lessons)

    return "\n".join(lines)


if __name__ == "__main__":
    main()
