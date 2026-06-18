from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


DEFAULT_MODEL = "gpt-4.1"
CRITIC_MODEL = "gpt-4.1-mini"


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
class ExecutionRecord:
    """The result of executing one plan step."""

    step: PlanStep
    result: str


class PlanAndExecuteAgent:
    """A baseline plan-and-execute agent with simple replanning."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        critic_model: str = CRITIC_MODEL,
        max_replans: int = 3,
        client: OpenAI | None = None,
    ) -> None:
        self.model = model
        self.critic_model = critic_model
        self.max_replans = max_replans
        self.client = client or OpenAI()

    def run(self, task: str) -> str:
        """Run the plan-execute-replan loop for a task."""

        plan = self._plan(task)
        self._print_plan("Initial plan", plan)

        executed_steps: list[ExecutionRecord] = []
        replan_count = 0
        step_index = 0

        while step_index < len(plan):
            step = plan[step_index]
            print(f"\nExecuting step {step_index + 1}/{len(plan)}: {step.description}")

            result = self._execute_step(step, executed_steps)
            executed_steps.append(ExecutionRecord(step=step, result=result))

            preview = result[:120].replace("\n", " ")
            print(f"Result preview: {preview}...")

            remaining_steps = plan[step_index + 1 :]
            should_replan = self._should_replan(task, executed_steps, remaining_steps)

            if should_replan and replan_count < self.max_replans:
                replan_count += 1
                print(f"\nReplanning remaining work ({replan_count}/{self.max_replans})")
                new_remaining_plan = self._replan(task, executed_steps, remaining_steps)
                plan = plan[: step_index + 1] + new_remaining_plan
                self._print_plan("Updated plan", plan[step_index + 1 :], start=step_index + 2)

            step_index += 1

        return self._synthesize(task, executed_steps)

    def _plan(self, task: str) -> list[PlanStep]:
        """Generate an executable plan for the task."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"""You are an expert task planner.

Create a detailed execution plan for the task below.

Task:
{task}

Requirements:
1. Break the task into 3 to 8 executable steps.
2. Each step must specify what to do, what tool or method to use, and the expected output.
3. Make dependencies between steps clear.
4. Do not execute any step. Only create the plan.

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

    def _execute_step(self, step: PlanStep, history: list[ExecutionRecord]) -> str:
        """Execute one plan step using the current execution history."""

        history_text = self._format_execution_history(history, max_chars=250)

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

Completed steps:
{history_text or "No steps have been completed yet."}

Return the execution result directly.""",
                }
            ],
        )

        return response.choices[0].message.content or ""

    def _should_replan(
        self,
        task: str,
        executed: list[ExecutionRecord],
        remaining: list[PlanStep],
    ) -> bool:
        """Decide whether the agent should replan the remaining steps."""

        if not remaining:
            return False

        last_record = executed[-1]

        response = self.client.chat.completions.create(
            model=self.critic_model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Decide whether the latest execution result seriously diverges from the expected output.

Original task:
{task}

Step objective:
{last_record.step.description}

Expected output:
{last_record.step.expected_output}

Actual result:
{last_record.result[:800]}

Answer YES only if the result failed, used incorrect assumptions, or makes the remaining plan unreliable.
Answer NO if the result is acceptable and the current plan can continue.

Return only YES or NO.""",
                }
            ],
            max_tokens=10,
        )

        decision = response.choices[0].message.content or ""
        return decision.strip().upper().startswith("YES")

    def _replan(
        self,
        task: str,
        executed: list[ExecutionRecord],
        old_remaining: list[PlanStep],
    ) -> list[PlanStep]:
        """Generate a new plan for the remaining work."""

        history_text = self._format_execution_history(executed, max_chars=300)
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

Original task:
{task}

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

    def _synthesize(self, task: str, executed: list[ExecutionRecord]) -> str:
        """Synthesize all execution results into the final answer."""

        history_text = self._format_execution_history(executed, max_chars=700)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Answer the original task based on the execution results.

Original task:
{task}

Execution results:
{history_text}

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
        history: list[ExecutionRecord],
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
