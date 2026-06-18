from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval_demo import (
    evaluate_cases,
    format_results,
    load_cases,
    results_to_dict,
    sample_cases,
)


class EvalDemoTests(unittest.TestCase):
    def test_sample_evaluation_ranks_search_memory_variant_first(self) -> None:
        results = evaluate_cases(sample_cases())
        payload = results_to_dict(results)

        self.assertEqual(payload["variant_count"], 3)
        self.assertEqual(payload["best_variant"], "search_memory")
        self.assertGreater(
            payload["variants"][0]["overall_score"],
            payload["variants"][-1]["overall_score"],
        )

    def test_format_results_includes_variant_metrics(self) -> None:
        results = evaluate_cases(sample_cases())
        report = format_results(results)

        self.assertIn("Agent Variant Evaluation", report)
        self.assertIn("Best variant: search_memory", report)
        self.assertIn("answer_present=", report)
        self.assertIn("memory_items=", report)

    def test_load_cases_reads_named_trace_files(self) -> None:
        trace = sample_cases()[0].trace
        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_path = Path(tmp_dir) / "baseline.json"
            trace_path.write_text(json.dumps(trace), encoding="utf-8")

            cases = load_cases([f"baseline={trace_path}"])

        self.assertEqual(cases[0].name, "baseline")
        self.assertEqual(
            cases[0].trace["task"]["goal"],
            "Compare agent framework variants",
        )


if __name__ == "__main__":
    unittest.main()
