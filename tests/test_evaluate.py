from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import new_math_ops.evaluate as evaluate  # noqa: E402
from new_math_ops.contracts import (  # noqa: E402
    ArithmeticFamily,
    ChatMessage,
    DatasetRow,
    DifficultyLevel,
    Metrics,
    Operator,
    PredictionRow,
    RunConfig,
)


class _FakeClient:
    async def complete(self, messages: Sequence[ChatMessage]) -> str:
        user_content = messages[1]["content"]
        if "8 ## 3" in user_content:
            return "<final_answer>5</final_answer>"
        if "1 @@ 9" in user_content:
            return "not-valid"
        return "<final_answer>0</final_answer>"


class EvaluateUnitTests(unittest.TestCase):
    def test_parse_final_answer_accepts_integer_tag(self) -> None:
        self.assertEqual(evaluate.parse_final_answer("<final_answer>-17</final_answer>"), -17)

    def test_parse_final_answer_accepts_embedded_lenient_tag(self) -> None:
        self.assertEqual(
            evaluate.parse_final_answer("Answer: <FINAL_ANSWER> 42 </FINAL_ANSWER>"),
            42,
        )


class EvaluateAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_evaluate_dataset_rows_computes_metrics(self) -> None:
        rows: list[DatasetRow] = [
            {
                "input": "8 ## 3",
                "expected_output": 5,
                "metadata": {
                    "id": "s1_0",
                    "arithmetic_family": ArithmeticFamily.NEW,
                    "difficulty_level": DifficultyLevel.L1,
                    "n_ops": 1,
                    "op_seq": [Operator.ABS_DIFF],
                },
            },
            {
                "input": "1 @@ 9",
                "expected_output": 9,
                "metadata": {
                    "id": "s2_0",
                    "arithmetic_family": ArithmeticFamily.NEW,
                    "difficulty_level": DifficultyLevel.L2,
                    "n_ops": 1,
                    "op_seq": [Operator.MAX],
                },
            },
        ]
        predictions, metrics = await evaluate.evaluate_dataset_rows(
            rows=rows,
            client=_FakeClient(),
            concurrency=2,
        )

        self.assertEqual(len(predictions), 2)
        self.assertEqual(metrics["total"], 2)
        self.assertEqual(metrics["correct"], 1)
        self.assertEqual(metrics["format_errors"], 1)
        accuracy = float(metrics["accuracy"])
        format_error_rate = float(metrics["format_error_rate"])
        by_family = dict(metrics["by_family"])
        by_difficulty = dict(metrics["by_difficulty"])
        by_n_ops = dict(metrics["by_n_ops"])

        self.assertAlmostEqual(accuracy, 0.5)
        self.assertAlmostEqual(format_error_rate, 0.5)
        self.assertIn(ArithmeticFamily.NEW, by_family)
        self.assertIn(DifficultyLevel.L1, by_difficulty)
        self.assertIn(DifficultyLevel.L2, by_difficulty)
        self.assertIn(1, by_n_ops)

    def test_write_run_artifacts_writes_metrics_and_predictions(self) -> None:
        predictions: list[PredictionRow] = [
            {
                "id": "s1_0",
                "input": "8 ## 3",
                "expected_output": 5,
                "predicted_output": 5,
                "is_correct": True,
                "format_error": False,
                "raw_response": "<final_answer>5</final_answer>",
                "arithmetic_family": ArithmeticFamily.NEW,
                "difficulty_level": DifficultyLevel.L1,
                "n_ops": 1,
                "latency_seconds": 0.01,
            }
        ]
        metrics: Metrics = {
            "total": 1,
            "correct": 1,
            "accuracy": 1.0,
            "format_errors": 0,
            "format_error_rate": 0.0,
            "by_family": {},
            "by_difficulty": {},
            "by_n_ops": {},
        }
        run_config: RunConfig = {
            "model": "openai/test",
            "temperature": 1.0,
            "max_tokens": 64,
            "concurrency": 2,
            "dataset": "data/new_math_ops_v2/dataset.jsonl",
            "manifest": None,
            "limit": None,
            "prompt_version": "v2",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = evaluate.write_run_artifacts(
                output_root=Path(tmp_dir),
                predictions=predictions,
                metrics=metrics,
                run_config=run_config,
                run_id="unit_test_run",
            )

            self.assertTrue((run_dir / "predictions.jsonl").exists())
            self.assertTrue((run_dir / "metrics.json").exists())
            self.assertTrue((run_dir / "run_config.json").exists())

            lines = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            row = json.loads(lines[0])
            self.assertEqual(row["id"], "s1_0")


if __name__ == "__main__":
    unittest.main()
