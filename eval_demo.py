from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from analyze_trace import analyze_trace


@dataclass(frozen=True)
class EvaluationMetric:
    """A weighted metric used to score an agent variant."""

    name: str
    weight: float


@dataclass(frozen=True)
class EvaluationCase:
    """One named trace to include in a variant comparison."""

    name: str
    trace: dict[str, Any]


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation scores for one agent variant."""

    name: str
    overall_score: float
    metric_scores: dict[str, float]
    summary: dict[str, Any]


DEFAULT_METRICS = [
    EvaluationMetric("answer_present", 0.10),
    EvaluationMetric("critic_quality", 0.20),
    EvaluationMetric("goal_alignment", 0.20),
    EvaluationMetric("evidence_strength", 0.20),
    EvaluationMetric("path_search", 0.10),
    EvaluationMetric("memory_usage", 0.10),
    EvaluationMetric("reflection_usage", 0.10),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare reasoning traces across agent variants."
    )
    parser.add_argument(
        "variants",
        nargs="*",
        help="Variant trace in the form name=path/to/trace.json.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print evaluation results as JSON.",
    )
    parser.add_argument(
        "--use-samples",
        action="store_true",
        help="Use built-in sample traces instead of reading trace files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = sample_cases() if args.use_samples or not args.variants else load_cases(args.variants)
    results = evaluate_cases(cases)

    if args.json:
        print(json.dumps(results_to_dict(results), indent=2, ensure_ascii=False))
    else:
        print(format_results(results))


def load_cases(variant_args: list[str]) -> list[EvaluationCase]:
    cases = []
    for variant_arg in variant_args:
        if "=" not in variant_arg:
            raise ValueError(
                "Variant traces must use the format name=path/to/trace.json."
            )

        name, raw_path = variant_arg.split("=", 1)
        name = name.strip()
        path = Path(raw_path.strip())
        if not name:
            raise ValueError("Variant name cannot be empty.")

        trace = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(trace, dict):
            raise ValueError(f"Trace for variant '{name}' must be a JSON object.")

        cases.append(EvaluationCase(name=name, trace=trace))

    return cases


def evaluate_cases(
    cases: list[EvaluationCase],
    metrics: list[EvaluationMetric] | None = None,
) -> list[EvaluationResult]:
    active_metrics = metrics or DEFAULT_METRICS
    results = [evaluate_case(case, active_metrics) for case in cases]
    return sorted(results, key=lambda result: result.overall_score, reverse=True)


def evaluate_case(
    case: EvaluationCase,
    metrics: list[EvaluationMetric],
) -> EvaluationResult:
    summary = analyze_trace(case.trace)
    metric_scores = {
        metric.name: score_metric(metric.name, summary)
        for metric in metrics
    }
    total_weight = sum(metric.weight for metric in metrics)
    weighted_score = sum(
        metric_scores[metric.name] * metric.weight
        for metric in metrics
    )
    overall_score = round(weighted_score / total_weight, 3) if total_weight else 0.0

    return EvaluationResult(
        name=case.name,
        overall_score=overall_score,
        metric_scores=metric_scores,
        summary=summary,
    )


def score_metric(metric_name: str, summary: dict[str, Any]) -> float:
    execution = summary["execution"]
    critic = summary["critic"]
    memory = summary["working_memory"]
    selected_path = summary.get("selected_path") or {}

    if metric_name == "answer_present":
        return 1.0 if execution["final_answer_present"] else 0.0
    if metric_name == "critic_quality":
        return score_or_zero(critic["average_quality_score"])
    if metric_name == "goal_alignment":
        return score_or_zero(critic["average_goal_alignment_score"])
    if metric_name == "evidence_strength":
        return score_or_zero(critic["average_evidence_score"])
    if metric_name == "path_search":
        path_count_signal = min(1.0, len(summary["candidate_paths"]) / 3)
        selected_score = score_or_zero(selected_path.get("total_score"))
        return round((path_count_signal + selected_score) / 2, 3)
    if metric_name == "memory_usage":
        return min(1.0, sum(memory.values()) / 8)
    if metric_name == "reflection_usage":
        return min(1.0, summary["reflections"]["reflection_count"] / 3)

    raise ValueError(f"Unknown evaluation metric: {metric_name}")


def score_or_zero(value: Any) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    return 0.0


def results_to_dict(results: list[EvaluationResult]) -> dict[str, Any]:
    return {
        "best_variant": results[0].name if results else None,
        "variant_count": len(results),
        "variants": [
            {
                "name": result.name,
                "overall_score": result.overall_score,
                "metric_scores": result.metric_scores,
                "selected_path": (result.summary.get("selected_path") or {}).get("strategy"),
                "candidate_path_count": len(result.summary["candidate_paths"]),
                "executed_step_count": result.summary["execution"]["executed_step_count"],
                "retry_count": result.summary["execution"]["retry_count"],
                "replan_count": result.summary["execution"]["replan_count"],
                "critic_issue_count": result.summary["critic"]["issue_count"],
                "memory_item_count": sum(result.summary["working_memory"].values()),
                "reflection_count": result.summary["reflections"]["reflection_count"],
            }
            for result in results
        ],
    }


def format_results(results: list[EvaluationResult]) -> str:
    payload = results_to_dict(results)
    lines = [
        "Agent Variant Evaluation",
        "========================",
        "",
        f"Variants compared: {payload['variant_count']}",
        f"Best variant: {payload['best_variant']}",
        "",
        "Scores:",
    ]

    for variant in payload["variants"]:
        metric_text = " ".join(
            f"{name}={score}"
            for name, score in variant["metric_scores"].items()
        )
        lines.extend(
            [
                f"- {variant['name']}: overall={variant['overall_score']}",
                f"  {metric_text}",
                (
                    f"  paths={variant['candidate_path_count']} "
                    f"steps={variant['executed_step_count']} "
                    f"retries={variant['retry_count']} "
                    f"replans={variant['replan_count']}"
                ),
                (
                    f"  memory_items={variant['memory_item_count']} "
                    f"reflections={variant['reflection_count']} "
                    f"issues={variant['critic_issue_count']}"
                ),
            ]
        )

    return "\n".join(lines)


def sample_cases() -> list[EvaluationCase]:
    return [
        EvaluationCase(
            name="baseline",
            trace=build_sample_trace(
                selected_score=0.55,
                quality=0.58,
                alignment=0.6,
                evidence=0.52,
                candidate_paths=1,
                memory_items=0,
                reflection_count=0,
                retry_count=0,
                replan_count=0,
                issues=["Plan used a shallow comparison frame."],
            ),
        ),
        EvaluationCase(
            name="reflective",
            trace=build_sample_trace(
                selected_score=0.76,
                quality=0.74,
                alignment=0.8,
                evidence=0.72,
                candidate_paths=1,
                memory_items=5,
                reflection_count=2,
                retry_count=1,
                replan_count=0,
                issues=["First pass needed more specific evidence."],
            ),
        ),
        EvaluationCase(
            name="search_memory",
            trace=build_sample_trace(
                selected_score=0.92,
                quality=0.88,
                alignment=0.91,
                evidence=0.86,
                candidate_paths=3,
                memory_items=8,
                reflection_count=3,
                retry_count=1,
                replan_count=1,
                issues=["One downstream step needed replanning."],
            ),
        ),
    ]


def build_sample_trace(
    selected_score: float,
    quality: float,
    alignment: float,
    evidence: float,
    candidate_paths: int,
    memory_items: int,
    reflection_count: int,
    retry_count: int,
    replan_count: int,
    issues: list[str],
) -> dict[str, Any]:
    paths = []
    for index in range(candidate_paths):
        selected = index == 0
        score = selected_score if selected else max(0.1, selected_score - 0.2 - index * 0.05)
        paths.append(
            {
                "path_id": chr(ord("A") + index),
                "strategy": "evidence-first" if selected else "alternate strategy",
                "selected": selected,
                "steps": [{"id": 1}, {"id": 2}],
                "evaluation": {
                    "total_score": round(score, 3),
                    "goal_alignment_score": round(score, 3),
                    "feasibility_score": round(score, 3),
                    "evidence_potential_score": round(score, 3),
                    "risk_score": round(1.0 - score, 3),
                    "strengths": ["Clear evaluation path"],
                    "weaknesses": [],
                },
            }
        )

    return {
        "task": {
            "goal": "Compare agent framework variants",
            "task_type": "evaluation",
            "constraints": [],
            "success_criteria": ["Identify the strongest variant"],
        },
        "candidate_paths": paths,
        "selected_path": None,
        "executed_steps": [{"step": {"id": 1}}, {"step": {"id": 2}}],
        "critiques": [
            {
                "quality_score": quality,
                "goal_alignment_score": alignment,
                "evidence_score": evidence,
                "issues": issues,
            }
        ],
        "reflections": [
            {
                "lesson": f"Lesson {index + 1}",
                "failure_mode": "none",
                "correction_strategy": "Continue with stronger evidence.",
            }
            for index in range(reflection_count)
        ],
        "working_memory": split_memory_items(memory_items),
        "trace": (
            [{"event_type": "step_retry_scheduled"} for _ in range(retry_count)]
            + [{"event_type": "plan_revised"} for _ in range(replan_count)]
        ),
        "replan_count": replan_count,
        "final_answer": "Final recommendation.",
    }


def split_memory_items(total_count: int) -> dict[str, list[dict[str, str]]]:
    categories = ["observations", "decisions", "failed_attempts", "lessons"]
    memory = {category: [] for category in categories}
    for index in range(total_count):
        category = categories[index % len(categories)]
        memory[category].append({"content": f"{category} item {index + 1}"})
    return memory


if __name__ == "__main__":
    main()
