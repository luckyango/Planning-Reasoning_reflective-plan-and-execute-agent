from __future__ import annotations

import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plan_and_execute_agent import PlanAndExecuteAgent


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def create(self, **kwargs) -> FakeResponse:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Fake client ran out of responses.")
        return FakeResponse(self.responses.pop(0))


class FakeChat:
    def __init__(self, responses: list[str]) -> None:
        self.completions = FakeCompletions(responses)


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.chat = FakeChat(responses)


def json_response(payload: dict) -> str:
    return json.dumps(payload)


def task_response() -> str:
    return json_response(
        {
            "goal": "Compare agent frameworks",
            "task_type": "decision",
            "constraints": ["Prefer maintainable orchestration"],
            "success_criteria": ["Recommend one framework"],
            "assumptions": ["Use public framework capabilities"],
        }
    )


def paths_response() -> str:
    return json_response(
        {
            "paths": [
                {
                    "path_id": "A",
                    "strategy": "evidence-first",
                    "rationale": "Collect concrete comparison evidence first.",
                    "steps": [
                        {
                            "id": 1,
                            "description": "Compare framework capabilities",
                            "tool": "analysis",
                            "expected_output": "Capability comparison",
                            "depends_on": [],
                        }
                    ],
                },
                {
                    "path_id": "B",
                    "strategy": "assumption-first",
                    "rationale": "Start from assumptions and validate later.",
                    "steps": [
                        {
                            "id": 1,
                            "description": "List framework assumptions",
                            "tool": "analysis",
                            "expected_output": "Assumption list",
                            "depends_on": [],
                        }
                    ],
                },
            ]
        }
    )


def evaluation_response(total_score: float) -> str:
    return json_response(
        {
            "goal_alignment_score": total_score,
            "feasibility_score": total_score,
            "evidence_potential_score": total_score,
            "risk_score": 1.0 - total_score,
            "total_score": total_score,
            "strengths": ["Clear path"],
            "weaknesses": [],
        }
    )


def critique_response(
    recommendation: str = "continue",
    should_retry: bool = False,
    should_replan: bool = False,
    issues: list[str] | None = None,
) -> str:
    return json_response(
        {
            "quality_score": 0.8,
            "goal_alignment_score": 0.9,
            "evidence_score": 0.8,
            "issues": issues or [],
            "recommendation": recommendation,
            "should_retry": should_retry,
            "should_replan": should_replan,
        }
    )


def reflection_response(next_action: str = "continue") -> str:
    return json_response(
        {
            "lesson": "Keep the comparison evidence specific.",
            "failure_mode": "none",
            "correction_strategy": "Continue with the selected path.",
            "next_action": next_action,
        }
    )


def run_silently(agent: PlanAndExecuteAgent, task: str) -> str:
    with redirect_stdout(StringIO()):
        return agent.run(task)


class AgentControlFlowTests(unittest.TestCase):
    def test_selects_highest_scoring_candidate_path(self) -> None:
        client = FakeClient(
            [
                task_response(),
                paths_response(),
                evaluation_response(0.9),
                evaluation_response(0.4),
                "Execution result for selected path.",
                critique_response(),
                reflection_response(),
                "Final answer.",
            ]
        )
        agent = PlanAndExecuteAgent(client=client)

        final_answer = run_silently(agent, "Compare frameworks")

        self.assertEqual(final_answer, "Final answer.")
        self.assertIsNotNone(agent.last_state)
        state = agent.last_state
        assert state is not None
        self.assertEqual(state.selected_path.path_id, "A")
        self.assertEqual(state.plan[0].description, "Compare framework capabilities")
        self.assertEqual(len(state.candidate_paths), 2)
        self.assertEqual(len(state.executed_steps), 1)
        self.assertGreater(len(state.working_memory.observations), 0)

    def test_retry_uses_critic_feedback_once(self) -> None:
        client = FakeClient(
            [
                task_response(),
                paths_response(),
                evaluation_response(0.9),
                evaluation_response(0.4),
                "Weak execution result.",
                critique_response(
                    recommendation="retry",
                    should_retry=True,
                    issues=["Result was too shallow"],
                ),
                reflection_response(next_action="retry"),
                "Improved execution result.",
                critique_response(),
                reflection_response(),
                "Final answer after retry.",
            ]
        )
        agent = PlanAndExecuteAgent(client=client)

        final_answer = run_silently(agent, "Compare frameworks")

        self.assertEqual(final_answer, "Final answer after retry.")
        self.assertIsNotNone(agent.last_state)
        state = agent.last_state
        assert state is not None
        self.assertEqual(len(state.executed_steps), 1)
        self.assertEqual(state.executed_steps[0].result, "Improved execution result.")
        self.assertEqual(len(state.critiques), 2)
        self.assertGreater(len(state.working_memory.failed_attempts), 0)

    def test_replan_replaces_remaining_steps(self) -> None:
        client = FakeClient(
            [
                task_response(),
                paths_response(),
                evaluation_response(0.9),
                evaluation_response(0.4),
                "First step result.",
                critique_response(
                    recommendation="replan",
                    should_replan=True,
                    issues=["Remaining plan needs adjustment"],
                ),
                reflection_response(next_action="replan"),
                json_response(
                    {
                        "steps": [
                            {
                                "id": 2,
                                "description": "Run revised comparison",
                                "tool": "analysis",
                                "expected_output": "Revised comparison",
                                "depends_on": [1],
                            }
                        ]
                    }
                ),
                "Replanned step result.",
                critique_response(),
                reflection_response(),
                "Final answer after replan.",
            ]
        )
        agent = PlanAndExecuteAgent(client=client)

        final_answer = run_silently(agent, "Compare frameworks")

        self.assertEqual(final_answer, "Final answer after replan.")
        self.assertIsNotNone(agent.last_state)
        state = agent.last_state
        assert state is not None
        self.assertEqual(state.replan_count, 1)
        self.assertEqual(len(state.executed_steps), 2)
        self.assertEqual(state.executed_steps[1].step.description, "Run revised comparison")
        self.assertGreater(len(state.working_memory.decisions), 0)


if __name__ == "__main__":
    unittest.main()
