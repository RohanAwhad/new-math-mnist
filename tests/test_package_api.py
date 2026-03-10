from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import new_math_ops


class PackageAPITests(unittest.TestCase):
    def test_exports_prompt_contract(self) -> None:
        messages = new_math_ops.build_messages("9 / 2 ## 3")
        self.assertEqual(new_math_ops.PROMPT_VERSION, "v2")
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")

    def test_exports_parse_final_answer(self) -> None:
        self.assertEqual(new_math_ops.parse_final_answer("<final_answer>-8</final_answer>"), -8)
        self.assertIsNone(new_math_ops.parse_final_answer("-8"))

    def test_exports_dataset_loader(self) -> None:
        row = {
            "input": "1 + 2",
            "expected_output": 3,
            "metadata": {
                "id": "sample_1",
                "arithmetic_family": "normal",
                "difficulty_level": "L1",
                "n_ops": 1,
                "op_seq": ["+"],
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = Path(tmp_dir) / "dataset.jsonl"
            dataset_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            rows = new_math_ops.load_dataset_rows(dataset_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["metadata"]["id"], "sample_1")


if __name__ == "__main__":
    unittest.main()
