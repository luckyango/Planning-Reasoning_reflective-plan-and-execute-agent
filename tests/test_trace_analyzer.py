from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze_trace import analyze_trace, format_summary, load_trace


def sample_trace() -> dict:
    return {
        "task": {
            "goal": "Compare agent frameworks",
            "task_type": "decision",
            "constraints": ["Prefer maintainable orchestration"],
            "success_criteria": ["Recommend one framework"],
        },
        "candidate_paths": [
            {
                "path_id": "A",
                "strategy": "evidence-first",
                "selected": True,
                "steps": [{"id": 1}, {"id": 2}],
                "evaluation": {
                    "total_score": 0.91,
                    "goal_alignment_score": 0.95,
                    "feasibility_score": 0.9,
                    "evidence_potential_score": 0.92,
                    "risk_score": 0.12,
                    "strengths": ["Specific evidence"],
                    "weaknesses": [],
                },
            },
            {
                "path_id": "B",
                "strategy": "assumption-first",
                "selected": False,
                "steps": [{"id": 1}],
                "evaluation": {
                    "total_score": 0.64,
                    "goal_alignment_score": 0.7,
                    "feasibility_score": 0.8,
                    "evidence_potential_score": 0.5,
                    "risk_score": 0.4,
                    "strengths": [],
                    "weaknesses": ["Less evidence-driven"],
                },
            },
        ],
        "selected_path": None,
        "executed_steps": [{"step": {"id": 1}}, {"step": {"id": 2}}],
        "critiques": [
            {
                "quality_score": 0.8,
                "goal_alignment_score": 0.9,
                "evidence_score": 0.7,
                "issues": ["Needs more source detail"],
            },
            {
                "quality_score": 0.9,
                "goal_alignment_score": 0.95,
                "evidence_score": 0.85,
                "issues": [],
            },
        ],
        "reflections": [
            {
                "lesson": "Use source-specific comparison criteria.",
                "failure_mode": "Insufficient evidence detail",
                "correction_strategy": "Ask for more concrete comparison evidence.",
            }
        ],
        "working_memory": {
            "observations": [{"content": "Observation"}],
            "decisions": [{"content": "Decision"}],
            "failed_attempts": [{"content": "Failed attempt"}],
            "lessons": [{"content": "Lesson"}],
        },
        "trace": [
            {"event_type": "run_started"},
            {"event_type": "step_retry_scheduled"},
            {"event_type": "run_completed"},
        ],
        "replan_count": 1,
        "final_answer": "Final recommendation.",
    }


class TraceAnalyzerTests(unittest.TestCase):
    def test_analyze_trace_summarizes_reasoning_run(self) -> None:
        summary = analyze_trace(sample_trace())

        self.assertEqual(summary["task"]["type"], "decision")
        self.assertEqual(summary["selected_path"]["path_id"], "A")
        self.assertEqual(summary["execution"]["executed_step_count"], 2)
        self.assertEqual(summary["execution"]["retry_count"], 1)
        self.assertEqual(summary["execution"]["replan_count"], 1)
        self.assertEqual(summary["critic"]["average_quality_score"], 0.85)
        self.assertEqual(summary["critic"]["issue_count"], 1)
        self.assertEqual(summary["working_memory"]["lessons"], 1)

    def test_format_summary_contains_key_sections(self) -> None:
        summary = analyze_trace(sample_trace())
        report = format_summary(summary)

        self.assertIn("Reasoning Run Summary", report)
        self.assertIn("Selected reasoning path", report)
        self.assertIn("Critic scores", report)
        self.assertIn("Working memory", report)
        self.assertIn("Key lessons", report)

    def test_load_trace_reads_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = Path(tmp_dir) / "trace.json"
            trace_path.write_text(json.dumps(sample_trace()), encoding="utf-8")

            loaded = load_trace(trace_path)

        self.assertEqual(loaded["task"]["goal"], "Compare agent frameworks")


if __name__ == "__main__":
    unittest.main()
