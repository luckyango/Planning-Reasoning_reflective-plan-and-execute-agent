from __future__ import annotations

import json
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from agent.memory import WorkingMemory
from agent.models import (
    AgentState,
    Critique,
    ExecutionResult,
    PlanEvaluation,
    PlanStep,
    ReasoningPath,
    Reflection,
    Task,
)

DEFAULT_MODEL = "gpt-4.1"
CRITIC_MODEL = "gpt-4.1-mini"


class TaskComposer:
    """Convert raw user input into a structured task profile."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def compose(self, user_input: str) -> Task:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a task composition engine for a planning agent.

Convert the user request into a structured task profile.

User request:
{user_input}

Classify task_type as one of:
- research
- analysis
- decision
- coding
- planning
- writing
- general

Requirements:
1. Preserve the user's core goal.
2. Extract explicit constraints from the request.
3. Infer only practical success criteria that are needed to judge completion.
4. List assumptions only when the request leaves important details unspecified.
5. Do not solve the task yet.

Return only a JSON object with this shape:
{{
  "goal": "Clear goal statement",
  "task_type": "research",
  "constraints": ["Constraint"],
  "success_criteria": ["Criterion"],
  "assumptions": ["Assumption"]
}}""",
                }
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("The task composer returned an empty response.")

        try:
            raw_task = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"The task composer returned invalid JSON: {content}") from exc

        if not isinstance(raw_task, dict):
            raise ValueError("The task composer response must be a JSON object.")

        return Task.from_dict(raw_task, original_input=user_input)


class Critic:
    """Evaluate execution results and recommend the next control action."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def critique(
        self,
        task_profile: str,
        execution_history: str,
        result: ExecutionResult,
        remaining_steps: list[PlanStep],
    ) -> Critique:
        remaining_step_descriptions = [
            step.description for step in remaining_steps
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a critic for a reflective planning agent.

Evaluate the latest execution result and recommend what the agent should do next.

Task profile:
{task_profile}

Execution history:
{execution_history or "No earlier steps."}

Latest step:
{result.step.description}

Expected output:
{result.step.expected_output}

Actual result:
{result.result[:1200]}

Remaining steps:
{json.dumps(remaining_step_descriptions, ensure_ascii=False, indent=2)}

Scoring guidance:
- quality_score: Does the result satisfy the step objective?
- goal_alignment_score: Does the result move the agent toward the overall goal?
- evidence_score: Is the result specific, supported, and usable by later steps?

Recommendation must be one of:
- continue
- retry
- replan

Return only a JSON object with this shape:
{{
  "quality_score": 0.0,
  "goal_alignment_score": 0.0,
  "evidence_score": 0.0,
  "issues": ["Issue"],
  "recommendation": "continue",
  "should_retry": false,
  "should_replan": false
}}""",
                }
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("The critic returned an empty response.")

        try:
            raw_critique = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"The critic returned invalid JSON: {content}") from exc

        if not isinstance(raw_critique, dict):
            raise ValueError("The critic response must be a JSON object.")

        critique = Critique.from_dict(raw_critique, step_id=result.step.id)
        if critique.recommendation == "retry":
            critique.should_retry = True
        if critique.recommendation == "replan":
            critique.should_replan = True
        return critique


class Reflector:
    """Generate reflexion records from execution results and critiques."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def reflect(
        self,
        task_profile: str,
        execution_history: str,
        result: ExecutionResult,
        critique: Critique,
    ) -> Reflection:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a reflection agent for a plan-and-execute system.

Create a concise reflexion record that helps the agent improve the next action.

Task profile:
{task_profile}

Execution history:
{execution_history or "No earlier steps."}

Latest step:
{result.step.description}

Expected output:
{result.step.expected_output}

Actual result:
{result.result[:1200]}

Critique:
{json.dumps(self._critique_payload(critique), ensure_ascii=False, indent=2)}

Requirements:
1. If the result is acceptable, capture what worked and how to continue.
2. If the result is weak, identify the failure mode and a correction strategy.
3. Do not solve the original task. Focus on improving the agent's process.
4. Keep each field concise and actionable.

Return only a JSON object with this shape:
{{
  "lesson": "Reusable lesson for this run",
  "failure_mode": "What went wrong, or none",
  "correction_strategy": "How to correct or continue",
  "next_action": "continue"
}}""",
                }
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("The reflector returned an empty response.")

        try:
            raw_reflection = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"The reflector returned invalid JSON: {content}") from exc

        if not isinstance(raw_reflection, dict):
            raise ValueError("The reflector response must be a JSON object.")

        return Reflection.from_dict(raw_reflection, step_id=result.step.id)

    def _critique_payload(self, critique: Critique) -> dict[str, Any]:
        return {
            "step_id": critique.step_id,
            "quality_score": critique.quality_score,
            "goal_alignment_score": critique.goal_alignment_score,
            "evidence_score": critique.evidence_score,
            "issues": critique.issues,
            "recommendation": critique.recommendation,
            "should_retry": critique.should_retry,
            "should_replan": critique.should_replan,
        }


class PlanSearch:
    """Generate and evaluate candidate reasoning paths before execution."""

    def __init__(self, client: OpenAI, generator_model: str, evaluator_model: str) -> None:
        self.client = client
        self.generator_model = generator_model
        self.evaluator_model = evaluator_model

    def search(
        self,
        task_profile: str,
        working_memory: str,
        candidate_count: int,
    ) -> list[ReasoningPath]:
        candidate_paths = self._generate_candidate_paths(
            task_profile=task_profile,
            working_memory=working_memory,
            candidate_count=candidate_count,
        )
        for path in candidate_paths:
            path.evaluation = self._evaluate_path(
                task_profile=task_profile,
                working_memory=working_memory,
                path=path,
            )
        return sorted(
            candidate_paths,
            key=lambda path: path.evaluation.total_score if path.evaluation else 0.0,
            reverse=True,
        )

    def _generate_candidate_paths(
        self,
        task_profile: str,
        working_memory: str,
        candidate_count: int,
    ) -> list[ReasoningPath]:
        response = self.client.chat.completions.create(
            model=self.generator_model,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a search-based planning agent.

Generate {candidate_count} distinct reasoning paths for the task. Each path should use a different planning strategy.

Task profile:
{task_profile}

Working memory:
{working_memory}

Useful strategy examples:
- evidence-first investigation
- hypothesis-driven reasoning
- compare-and-rank analysis
- risk-first validation
- decomposition-by-dependencies

Requirements:
1. Produce genuinely different candidate paths.
2. Each path must contain 3 to 8 executable steps.
3. Each step must specify what to do, what tool or method to use, and the expected output.
4. Do not execute any step. Only create candidate plans.

Return only a JSON object with this shape:
{{
  "paths": [
    {{
      "path_id": "A",
      "strategy": "Strategy name",
      "rationale": "Why this path may work",
      "steps": [
        {{
          "id": 1,
          "description": "Step description",
          "tool": "Tool or method",
          "expected_output": "Expected output",
          "depends_on": []
        }}
      ]
    }}
  ]
}}""",
                }
            ],
            response_format={"type": "json_object"},
        )

        raw_json = self._parse_json_response(response.choices[0].message.content)
        raw_paths = raw_json.get("paths", [])
        if not isinstance(raw_paths, list):
            raise ValueError("The plan search response must include a 'paths' list.")

        paths = [
            ReasoningPath.from_dict(raw_path, fallback_id=index + 1)
            for index, raw_path in enumerate(raw_paths)
            if isinstance(raw_path, dict)
        ]
        valid_paths = [path for path in paths if path.steps]
        if not valid_paths:
            raise ValueError("The plan search response did not include any valid paths.")

        return valid_paths

    def _evaluate_path(
        self,
        task_profile: str,
        working_memory: str,
        path: ReasoningPath,
    ) -> PlanEvaluation:
        response = self.client.chat.completions.create(
            model=self.evaluator_model,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are a plan evaluator for a search-based reasoning agent.

Score the candidate reasoning path before execution.

Task profile:
{task_profile}

Working memory:
{working_memory}

Candidate path:
{json.dumps(self._path_payload(path), ensure_ascii=False, indent=2)}

Scoring guidance:
- goal_alignment_score: Does this path directly serve the user's goal and success criteria?
- feasibility_score: Can the steps be executed in a clear sequence?
- evidence_potential_score: Is the path likely to produce specific, useful evidence?
- risk_score: How likely is this path to fail, drift, or produce unsupported conclusions?
- total_score: Overall score from 0.0 to 1.0. Higher is better.

Return only a JSON object with this shape:
{{
  "goal_alignment_score": 0.0,
  "feasibility_score": 0.0,
  "evidence_potential_score": 0.0,
  "risk_score": 0.0,
  "total_score": 0.0,
  "strengths": ["Strength"],
  "weaknesses": ["Weakness"]
}}""",
                }
            ],
            response_format={"type": "json_object"},
        )

        raw_json = self._parse_json_response(response.choices[0].message.content)
        return PlanEvaluation.from_dict(raw_json)

    def _path_payload(self, path: ReasoningPath) -> dict[str, Any]:
        return {
            "path_id": path.path_id,
            "strategy": path.strategy,
            "rationale": path.rationale,
            "steps": [
                {
                    "id": step.id,
                    "description": step.description,
                    "tool": step.tool,
                    "expected_output": step.expected_output,
                    "depends_on": step.depends_on,
                }
                for step in path.steps
            ],
        }

    def _parse_json_response(self, content: str | None) -> dict[str, Any]:
        if not content:
            raise ValueError("The plan search agent returned an empty response.")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"The plan search agent returned invalid JSON: {content}") from exc

        if not isinstance(parsed, dict):
            raise ValueError("The plan search response must be a JSON object.")

        return parsed


class PlanAndExecuteAgent:
    """A baseline plan-and-execute agent with simple replanning."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        critic_model: str = CRITIC_MODEL,
        max_replans: int = 3,
        candidate_plan_count: int = 3,
        client: OpenAI | None = None,
    ) -> None:
        self.model = model
        self.critic_model = critic_model
        self.max_replans = max_replans
        self.candidate_plan_count = candidate_plan_count
        if client is None and OpenAI is None:
            raise RuntimeError(
                "The openai package is required when no custom client is provided. "
                "Install dependencies with 'pip install -r requirements.txt'."
            )
        self.client = client or OpenAI()
        self.task_composer = TaskComposer(client=self.client, model=self.model)
        self.critic = Critic(client=self.client, model=self.critic_model)
        self.reflector = Reflector(client=self.client, model=self.critic_model)
        self.plan_search = PlanSearch(
            client=self.client,
            generator_model=self.model,
            evaluator_model=self.critic_model,
        )
        self.last_state: AgentState | None = None

    def run(self, task: str) -> str:
        """Run the plan-execute-replan loop for a task."""

        composed_task = self.task_composer.compose(task)
        state = AgentState(task=composed_task)
        self.last_state = state
        state.add_trace(
            "task_composed",
            "User input composed into a structured task profile.",
            {
                "original_input": composed_task.original_input,
                "goal": composed_task.goal,
                "task_type": composed_task.task_type,
                "constraints": composed_task.constraints,
                "success_criteria": composed_task.success_criteria,
                "assumptions": composed_task.assumptions,
            },
        )
        state.working_memory.add_decision(
            f"Composed the user request as a {composed_task.task_type} task: "
            f"{composed_task.goal}",
            importance=0.9,
        )
        state.add_trace("run_started", "Agent run started.", {"task": composed_task.goal})

        state.plan = self._search_plan(state)
        state.working_memory.add_decision(
            f"Selected an initial plan with {len(state.plan)} steps from "
            f"{len(state.candidate_paths)} candidate reasoning paths.",
            importance=0.7,
        )
        state.add_trace(
            "plan_created",
            "Initial plan selected from candidate reasoning paths.",
            {
                "step_count": len(state.plan),
                "candidate_path_count": len(state.candidate_paths),
                "selected_path_id": state.selected_path.path_id if state.selected_path else None,
            },
        )
        self._print_plan("Initial plan", state.plan)

        step_index = 0
        retry_counts: dict[int, int] = {}

        while step_index < len(state.plan):
            step = state.plan[step_index]
            print(f"\nExecuting step {step_index + 1}/{len(state.plan)}: {step.description}")
            state.add_trace(
                "step_started",
                "Plan step execution started.",
                {"step_id": step.id, "description": step.description},
            )

            result = self._execute_step(step, state)
            execution_result = ExecutionResult(step=step, result=result)
            state.executed_steps.append(execution_result)
            state.working_memory.add_observation(
                f"Step {step.id} result: {result[:500]}",
                source_step_id=step.id,
                importance=0.6,
            )
            state.add_trace(
                "step_completed",
                "Plan step execution completed.",
                {
                    "step_id": step.id,
                    "description": step.description,
                    "result_preview": result[:240],
                },
            )

            preview = result[:120].replace("\n", " ")
            print(f"Result preview: {preview}...")

            remaining_steps = state.remaining_steps_after(step_index)
            critique = self.critic.critique(
                task_profile=self._format_task_profile(state.task),
                execution_history=self._format_execution_history(
                    state.executed_steps[:-1],
                    max_chars=300,
                ),
                result=execution_result,
                remaining_steps=remaining_steps,
            )
            state.critiques.append(critique)
            reflection = self.reflector.reflect(
                task_profile=self._format_task_profile(state.task),
                execution_history=self._format_execution_history(
                    state.executed_steps[:-1],
                    max_chars=300,
                ),
                result=execution_result,
                critique=critique,
            )
            state.reflections.append(reflection)
            if critique.issues:
                state.working_memory.add_failed_attempt(
                    f"Step {step.id} issues: {'; '.join(critique.issues)}",
                    source_step_id=step.id,
                    importance=0.8,
                )
            state.working_memory.add_lesson(
                (
                    f"Step {step.id} lesson: {reflection.lesson}. "
                    f"Correction strategy: {reflection.correction_strategy}"
                ),
                source_step_id=step.id,
                importance=0.8,
            )
            state.add_trace(
                "critique_completed",
                "Execution result critique completed.",
                {
                    "step_id": step.id,
                    "quality_score": critique.quality_score,
                    "goal_alignment_score": critique.goal_alignment_score,
                    "evidence_score": critique.evidence_score,
                    "recommendation": critique.recommendation,
                    "should_retry": critique.should_retry,
                    "should_replan": critique.should_replan,
                    "remaining_step_count": len(remaining_steps),
                },
            )
            state.add_trace(
                "reflection_created",
                "Reflection created from critique.",
                {
                    "step_id": step.id,
                    "lesson": reflection.lesson,
                    "failure_mode": reflection.failure_mode,
                    "correction_strategy": reflection.correction_strategy,
                    "next_action": reflection.next_action,
                },
            )

            if critique.should_retry and retry_counts.get(step.id, 0) < 1:
                retry_counts[step.id] = retry_counts.get(step.id, 0) + 1
                print(f"\nRetrying step {step.id} based on critic feedback")
                state.working_memory.add_decision(
                    f"Retry step {step.id} because the critic recommended retry.",
                    source_step_id=step.id,
                    importance=0.9,
                )
                state.add_trace(
                    "step_retry_scheduled",
                    "Critic recommended retrying the current step.",
                    {"step_id": step.id, "issues": critique.issues},
                )
                state.executed_steps.pop()
                step_index -= 1

            elif critique.should_replan and state.replan_count < self.max_replans:
                state.replan_count += 1
                print(f"\nReplanning remaining work ({state.replan_count}/{self.max_replans})")
                new_remaining_plan = self._replan(state, remaining_steps)
                state.plan = state.plan[: step_index + 1] + new_remaining_plan
                state.working_memory.add_decision(
                    (
                        f"Replanned remaining work after step {step.id}; "
                        f"new remaining step count: {len(new_remaining_plan)}."
                    ),
                    source_step_id=step.id,
                    importance=0.9,
                )
                state.add_trace(
                    "plan_revised",
                    "Remaining plan revised.",
                    {
                        "replan_count": state.replan_count,
                        "new_remaining_step_count": len(new_remaining_plan),
                    },
                )
                self._print_plan("Updated plan", state.plan[step_index + 1 :], start=step_index + 2)

            step_index += 1

        state.final_answer = self._synthesize(state)
        state.add_trace(
            "run_completed",
            "Agent run completed.",
            {"executed_step_count": len(state.executed_steps)},
        )
        return state.final_answer

    def _search_plan(self, state: AgentState) -> list[PlanStep]:
        """Search over candidate reasoning paths and select the best plan."""

        candidate_paths = self.plan_search.search(
            task_profile=self._format_task_profile(state.task),
            working_memory=self._format_working_memory(state.working_memory),
            candidate_count=self.candidate_plan_count,
        )
        state.candidate_paths = candidate_paths
        selected_path = candidate_paths[0]
        selected_path.selected = True
        state.selected_path = selected_path

        state.add_trace(
            "candidate_paths_evaluated",
            "Candidate reasoning paths generated and evaluated.",
            {
                "paths": [
                    {
                        "path_id": path.path_id,
                        "strategy": path.strategy,
                        "total_score": (
                            path.evaluation.total_score if path.evaluation else None
                        ),
                        "selected": path.selected,
                    }
                    for path in candidate_paths
                ]
            },
        )
        state.working_memory.add_decision(
            (
                f"Selected reasoning path {selected_path.path_id} using strategy "
                f"'{selected_path.strategy}' with score "
                f"{selected_path.evaluation.total_score if selected_path.evaluation else 'unknown'}."
            ),
            importance=0.9,
        )
        return selected_path.steps

    def _plan(self, state: AgentState) -> list[PlanStep]:
        """Generate an executable plan for the task."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are an expert task planner.

Create a detailed execution plan for the structured task profile below.

{self._format_task_profile(state.task)}

Working memory:
{self._format_working_memory(state.working_memory)}

Requirements:
1. Break the task into 3 to 8 executable steps.
2. Each step must specify what to do, what tool or method to use, and the expected output.
3. Use the success criteria to decide what evidence or outputs are needed.
4. Respect the constraints and assumptions.
5. Make dependencies between steps clear.
6. Do not execute any step. Only create the plan.

Return only a JSON object with this shape:
{{
  "steps": [
    {{
      "id": 1,
      "description": "Step description",
      "tool": "Tool or method",
      "expected_output": "Expected output",
      "depends_on": []
    }}
  ]
}}""",
                }
            ],
            response_format={"type": "json_object"},
        )

        raw_json = self._parse_json_response(response.choices[0].message.content)
        return self._parse_plan(raw_json)

    def _execute_step(self, step: PlanStep, state: AgentState) -> str:
        """Execute one plan step using the current execution history."""

        history_text = self._format_execution_history(state.executed_steps, max_chars=250)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Execute the following plan step.

Step:
{step.description}

Tool or method:
{step.tool}

Expected output:
{step.expected_output}

Task profile:
{self._format_task_profile(state.task)}

Working memory:
{self._format_working_memory(state.working_memory)}

Completed steps:
{history_text or "No steps have been completed yet."}

Return the execution result directly.""",
                }
            ],
        )

        return response.choices[0].message.content or ""

    def _replan(
        self,
        state: AgentState,
        old_remaining: list[PlanStep],
    ) -> list[PlanStep]:
        """Generate a new plan for the remaining work."""

        history_text = self._format_execution_history(state.executed_steps, max_chars=300)
        remaining_text = json.dumps(
            [step.description for step in old_remaining],
            ensure_ascii=False,
            indent=2,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Replan the remaining work based on the execution history.

Task profile:
{self._format_task_profile(state.task)}

Working memory:
{self._format_working_memory(state.working_memory)}

Completed steps:
{history_text}

Old remaining steps:
{remaining_text}

Create a revised plan for only the remaining work.

Return only a JSON object with this shape:
{{
  "steps": [
    {{
      "id": 1,
      "description": "Step description",
      "tool": "Tool or method",
      "expected_output": "Expected output",
      "depends_on": []
    }}
  ]
}}""",
                }
            ],
            response_format={"type": "json_object"},
        )

        raw_json = self._parse_json_response(response.choices[0].message.content)
        return self._parse_plan(raw_json)

    def _synthesize(self, state: AgentState) -> str:
        """Synthesize all execution results into the final answer."""

        history_text = self._format_execution_history(state.executed_steps, max_chars=700)
        reflection_text = self._format_reflections(state.reflections)
        search_summary = self._format_reasoning_paths(state.candidate_paths)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Answer the original task based on the execution results.

Task profile:
{self._format_task_profile(state.task)}

Working memory:
{self._format_working_memory(state.working_memory)}

Search-based reasoning summary:
{search_summary or "No candidate reasoning paths were recorded."}

Execution results:
{history_text}

Reflections:
{reflection_text or "No reflections were recorded."}

Provide a complete, structured final answer.""",
                }
            ],
        )

        return response.choices[0].message.content or ""

    def _parse_json_response(self, content: str | None) -> dict[str, Any]:
        """Parse a JSON object returned by the model."""

        if not content:
            raise ValueError("The model returned an empty response.")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"The model returned invalid JSON: {content}") from exc

        if not isinstance(parsed, dict):
            raise ValueError("The model response must be a JSON object.")

        return parsed

    def _parse_plan(self, raw_json: dict[str, Any]) -> list[PlanStep]:
        """Convert a JSON plan payload into typed plan steps."""

        raw_steps = raw_json.get("steps") or raw_json.get("plan")
        if not isinstance(raw_steps, list):
            raise ValueError("The plan response must include a 'steps' list.")

        plan = [
            PlanStep.from_dict(raw_step, fallback_id=index + 1)
            for index, raw_step in enumerate(raw_steps)
            if isinstance(raw_step, dict)
        ]

        if not plan:
            raise ValueError("The plan response did not include any valid steps.")

        return plan

    def _format_execution_history(
        self,
        history: list[ExecutionResult],
        max_chars: int,
    ) -> str:
        """Format execution history for model context."""

        lines = []
        for index, record in enumerate(history, 1):
            result_preview = record.result[:max_chars]
            lines.append(
                f"Step {index} - {record.step.description}\n"
                f"Result: {result_preview}"
            )
        return "\n\n".join(lines)

    def _format_task_profile(self, task: Task) -> str:
        """Format a structured task profile for model prompts."""

        return json.dumps(
            {
                "original_input": task.original_input,
                "goal": task.goal,
                "task_type": task.task_type,
                "constraints": task.constraints,
                "success_criteria": task.success_criteria,
                "assumptions": task.assumptions,
            },
            ensure_ascii=False,
            indent=2,
        )

    def _format_reflections(self, reflections: list[Reflection]) -> str:
        """Format reflection records for model context."""

        lines = []
        for reflection in reflections:
            lines.append(
                f"Step {reflection.step_id}\n"
                f"Lesson: {reflection.lesson}\n"
                f"Failure mode: {reflection.failure_mode}\n"
                f"Correction strategy: {reflection.correction_strategy}\n"
                f"Next action: {reflection.next_action}"
            )
        return "\n\n".join(lines)

    def _format_working_memory(self, working_memory: WorkingMemory) -> str:
        """Format working memory as compact model context."""

        memory_payload = working_memory.to_dict()
        has_memory = any(memory_payload.values())
        if not has_memory:
            return "No working memory has been recorded yet."

        return json.dumps(memory_payload, ensure_ascii=False, indent=2)

    def _format_reasoning_paths(self, paths: list[ReasoningPath]) -> str:
        """Format candidate reasoning paths and scores for model context."""

        if not paths:
            return ""

        payload = []
        for path in paths:
            evaluation = path.evaluation
            payload.append(
                {
                    "path_id": path.path_id,
                    "strategy": path.strategy,
                    "rationale": path.rationale,
                    "selected": path.selected,
                    "total_score": evaluation.total_score if evaluation else None,
                    "goal_alignment_score": (
                        evaluation.goal_alignment_score if evaluation else None
                    ),
                    "feasibility_score": evaluation.feasibility_score if evaluation else None,
                    "evidence_potential_score": (
                        evaluation.evidence_potential_score if evaluation else None
                    ),
                    "risk_score": evaluation.risk_score if evaluation else None,
                    "strengths": evaluation.strengths if evaluation else [],
                    "weaknesses": evaluation.weaknesses if evaluation else [],
                }
            )

        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _print_plan(
        self,
        title: str,
        plan: list[PlanStep],
        start: int = 1,
    ) -> None:
        """Print a compact plan summary."""

        print(f"\n{title}:")
        for index, step in enumerate(plan, start):
            print(f"  {index}. {step.description}")


if __name__ == "__main__":
    agent = PlanAndExecuteAgent()
    final_answer = agent.run(
        "Analyze the 2025 global AI agent market size, compare it with 2024, "
        "and identify the fastest-growing segments and key growth drivers."
    )
    print(f"\nFinal answer:\n{final_answer}")
