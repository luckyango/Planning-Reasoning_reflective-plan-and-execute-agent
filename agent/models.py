from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.memory import WorkingMemory
from agent.utils import clamp_score, normalize_bool, normalize_string_list, utc_now


@dataclass
class Task:
    """A structured task profile that anchors an agent run."""

    original_input: str
    goal: str
    task_type: str
    constraints: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw_task: dict[str, Any], original_input: str) -> "Task":
        return cls(
            original_input=original_input,
            goal=str(raw_task.get("goal", original_input)).strip(),
            task_type=str(raw_task.get("task_type", "general")).strip(),
            constraints=normalize_string_list(raw_task.get("constraints", [])),
            success_criteria=normalize_string_list(raw_task.get("success_criteria", [])),
            assumptions=normalize_string_list(raw_task.get("assumptions", [])),
        )


@dataclass
class PlanStep:
    """A single executable step in a plan."""

    id: int
    description: str
    tool: str
    expected_output: str
    depends_on: list[int]

    @classmethod
    def from_dict(cls, raw_step: dict[str, Any], fallback_id: int) -> "PlanStep":
        return cls(
            id=int(raw_step.get("id", fallback_id)),
            description=str(raw_step.get("description", "")).strip(),
            tool=str(raw_step.get("tool", "general reasoning")).strip(),
            expected_output=str(raw_step.get("expected_output", "")).strip(),
            depends_on=[
                int(step_id)
                for step_id in raw_step.get("depends_on", [])
                if str(step_id).isdigit()
            ],
        )


@dataclass
class PlanEvaluation:
    """Scores assigned to one candidate reasoning path."""

    goal_alignment_score: float
    feasibility_score: float
    evidence_potential_score: float
    risk_score: float
    total_score: float
    strengths: list[str]
    weaknesses: list[str]

    @classmethod
    def from_dict(cls, raw_evaluation: dict[str, Any]) -> "PlanEvaluation":
        goal_alignment_score = clamp_score(raw_evaluation.get("goal_alignment_score", 0.0))
        feasibility_score = clamp_score(raw_evaluation.get("feasibility_score", 0.0))
        evidence_potential_score = clamp_score(
            raw_evaluation.get("evidence_potential_score", 0.0)
        )
        risk_score = clamp_score(raw_evaluation.get("risk_score", 1.0))
        default_total = (
            goal_alignment_score
            + feasibility_score
            + evidence_potential_score
            + (1.0 - risk_score)
        ) / 4

        return cls(
            goal_alignment_score=goal_alignment_score,
            feasibility_score=feasibility_score,
            evidence_potential_score=evidence_potential_score,
            risk_score=risk_score,
            total_score=clamp_score(raw_evaluation.get("total_score", default_total)),
            strengths=normalize_string_list(raw_evaluation.get("strengths", [])),
            weaknesses=normalize_string_list(raw_evaluation.get("weaknesses", [])),
        )


@dataclass
class ReasoningPath:
    """A candidate plan produced during search-based reasoning."""

    path_id: str
    strategy: str
    rationale: str
    steps: list[PlanStep]
    evaluation: PlanEvaluation | None = None
    selected: bool = False

    @classmethod
    def from_dict(cls, raw_path: dict[str, Any], fallback_id: int) -> "ReasoningPath":
        raw_steps = raw_path.get("steps", [])
        steps = [
            PlanStep.from_dict(raw_step, fallback_id=index + 1)
            for index, raw_step in enumerate(raw_steps)
            if isinstance(raw_step, dict)
        ]
        return cls(
            path_id=str(raw_path.get("path_id", f"path_{fallback_id}")).strip(),
            strategy=str(raw_path.get("strategy", "")).strip(),
            rationale=str(raw_path.get("rationale", "")).strip(),
            steps=steps,
        )


@dataclass
class ExecutionResult:
    """The result of executing one plan step."""

    step: PlanStep
    result: str
    status: str = "completed"
    created_at: str = field(default_factory=utc_now)


@dataclass
class Critique:
    """A structured quality assessment for an execution result."""

    step_id: int
    quality_score: float
    goal_alignment_score: float
    evidence_score: float
    issues: list[str]
    recommendation: str
    should_retry: bool
    should_replan: bool
    created_at: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, raw_critique: dict[str, Any], step_id: int) -> "Critique":
        return cls(
            step_id=step_id,
            quality_score=clamp_score(raw_critique.get("quality_score", 0.0)),
            goal_alignment_score=clamp_score(raw_critique.get("goal_alignment_score", 0.0)),
            evidence_score=clamp_score(raw_critique.get("evidence_score", 0.0)),
            issues=normalize_string_list(raw_critique.get("issues", [])),
            recommendation=str(raw_critique.get("recommendation", "continue")).strip().lower(),
            should_retry=normalize_bool(raw_critique.get("should_retry", False)),
            should_replan=normalize_bool(raw_critique.get("should_replan", False)),
        )


@dataclass
class Reflection:
    """A self-correction note generated by the reflection agent."""

    step_id: int
    lesson: str
    failure_mode: str
    correction_strategy: str
    next_action: str
    created_at: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, raw_reflection: dict[str, Any], step_id: int) -> "Reflection":
        return cls(
            step_id=step_id,
            lesson=str(raw_reflection.get("lesson", "")).strip(),
            failure_mode=str(raw_reflection.get("failure_mode", "")).strip(),
            correction_strategy=str(raw_reflection.get("correction_strategy", "")).strip(),
            next_action=str(raw_reflection.get("next_action", "continue")).strip().lower(),
        )


@dataclass
class TraceEvent:
    """A structured event emitted during an agent run."""

    event_type: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)


@dataclass
class AgentState:
    """Mutable state for one plan-and-execute run."""

    task: Task
    plan: list[PlanStep] = field(default_factory=list)
    candidate_paths: list[ReasoningPath] = field(default_factory=list)
    selected_path: ReasoningPath | None = None
    executed_steps: list[ExecutionResult] = field(default_factory=list)
    critiques: list[Critique] = field(default_factory=list)
    reflections: list[Reflection] = field(default_factory=list)
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    trace: list[TraceEvent] = field(default_factory=list)
    replan_count: int = 0
    final_answer: str | None = None

    def add_trace(
        self,
        event_type: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.trace.append(
            TraceEvent(
                event_type=event_type,
                message=message,
                payload=payload or {},
            )
        )

    def remaining_steps_after(self, step_index: int) -> list[PlanStep]:
        return self.plan[step_index + 1 :]

    def trace_summary(self) -> list[dict[str, Any]]:
        """Return a serializable summary of the execution trace."""

        return [
            {
                "event_type": event.event_type,
                "message": event.message,
                "payload": event.payload,
                "created_at": event.created_at,
            }
            for event in self.trace
        ]
