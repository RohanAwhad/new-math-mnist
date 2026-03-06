from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import generate_dataset
from contracts import (
    ArithmeticFamily,
    DifficultyLevel,
    Operator,
)


class GenerateDatasetTests(unittest.TestCase):
    def test_allocate_family_counts_requires_even_total(self) -> None:
        with self.assertRaises(ValueError):
            generate_dataset.allocate_family_counts(101)

        self.assertEqual(
            generate_dataset.allocate_family_counts(120),
            {
                ArithmeticFamily.NORMAL: 60,
                ArithmeticFamily.NEW: 60,
            },
        )

    def test_allocate_level_counts_uses_equal_thirds(self) -> None:
        counts = generate_dataset.allocate_level_counts(60)
        self.assertEqual(sum(counts.values()), 60)
        self.assertEqual(counts[DifficultyLevel.L1], 20)
        self.assertEqual(counts[DifficultyLevel.L2], 20)
        self.assertEqual(counts[DifficultyLevel.L3], 20)

    def test_build_label_histogram_supports_multi_digit_and_negative(self) -> None:
        samples = [
            generate_dataset.Sample(
                sample_id="a",
                arithmetic_family=ArithmeticFamily.NEW,
                difficulty_level=DifficultyLevel.L1,
                input="1 ## 3",
                expected_output=2,
                n_ops=1,
                op_seq=(Operator.ABS_DIFF,),
            ),
            generate_dataset.Sample(
                sample_id="b",
                arithmetic_family=ArithmeticFamily.NORMAL,
                difficulty_level=DifficultyLevel.L2,
                input="9 - 15",
                expected_output=-6,
                n_ops=1,
                op_seq=(Operator.SUBTRACT,),
            ),
            generate_dataset.Sample(
                sample_id="c",
                arithmetic_family=ArithmeticFamily.NORMAL,
                difficulty_level=DifficultyLevel.L2,
                input="4 * 5",
                expected_output=20,
                n_ops=1,
                op_seq=(Operator.MULTIPLY,),
            ),
        ]

        histogram = generate_dataset.build_label_histogram(samples)
        self.assertEqual(histogram["-6"], 1)
        self.assertEqual(histogram["2"], 1)
        self.assertEqual(histogram["20"], 1)

    def test_cli_writes_single_dataset_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = MODULE_ROOT / "generate_dataset.py"
            output_dir = Path(tmp_dir) / "out"
            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--output-dir",
                    str(output_dir),
                    "--seed",
                    "9",
                    "--num-examples",
                    "120",
                ],
                check=True,
            )

            dataset_path = output_dir / "dataset.jsonl"
            manifest_path = output_dir / "manifest.json"

            self.assertTrue(dataset_path.exists())
            self.assertTrue(manifest_path.exists())

            dataset_lines = (
                dataset_path.read_text(encoding="utf-8").strip().splitlines()
            )
            self.assertEqual(len(dataset_lines), 120)
            rows = [json.loads(line) for line in dataset_lines]

            counts_by_family: Counter[str] = Counter()
            counts_by_difficulty: Counter[str] = Counter()
            counts_by_pair: Counter[tuple[str, str]] = Counter()

            level_bounds: dict[str, tuple[int, int]] = {
                "L1": (1, 5),
                "L2": (6, 10),
                "L3": (11, 20),
            }
            normal_ops = {"+", "-", "*", "/"}
            new_ops = {"##", "@@", "$$"}

            for row in rows:
                self.assertEqual(
                    set(row.keys()), {"input", "expected_output", "metadata"}
                )

                metadata = row["metadata"]
                family = metadata["arithmetic_family"]
                difficulty = metadata["difficulty_level"]
                n_ops = metadata["n_ops"]
                op_seq = metadata["op_seq"]
                expression_tokens = row["input"].split()

                counts_by_family[family] += 1
                counts_by_difficulty[difficulty] += 1
                counts_by_pair[(family, difficulty)] += 1

                self.assertEqual(n_ops, len(op_seq))
                self.assertEqual(op_seq, expression_tokens[1::2])

                min_ops, max_ops = level_bounds[difficulty]
                self.assertGreaterEqual(n_ops, min_ops)
                self.assertLessEqual(n_ops, max_ops)

                if family == "normal":
                    self.assertTrue(set(op_seq).issubset(normal_ops))
                else:
                    self.assertTrue(set(op_seq).issubset(new_ops))

                for index, operator in enumerate(op_seq):
                    if operator == "/":
                        rhs = int(expression_tokens[(index * 2) + 2])
                        self.assertNotEqual(rhs, 0)

            self.assertEqual(counts_by_family["normal"], 60)
            self.assertEqual(counts_by_family["new"], 60)
            self.assertEqual(counts_by_difficulty["L1"], 40)
            self.assertEqual(counts_by_difficulty["L2"], 40)
            self.assertEqual(counts_by_difficulty["L3"], 40)
            self.assertEqual(counts_by_pair[("normal", "L1")], 20)
            self.assertEqual(counts_by_pair[("normal", "L2")], 20)
            self.assertEqual(counts_by_pair[("normal", "L3")], 20)
            self.assertEqual(counts_by_pair[("new", "L1")], 20)
            self.assertEqual(counts_by_pair[("new", "L2")], 20)
            self.assertEqual(counts_by_pair[("new", "L3")], 20)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["benchmark_name"], "new_math_ops_v2")
            self.assertEqual(manifest["generated_num_examples"], 120)
            self.assertEqual(manifest["files"], ["dataset.jsonl", "manifest.json"])
            self.assertEqual(manifest["counts_by_family"]["normal"], 60)
            self.assertEqual(manifest["counts_by_family"]["new"], 60)
            self.assertEqual(manifest["counts_by_difficulty"]["L1"], 40)
            self.assertEqual(manifest["counts_by_difficulty"]["L2"], 40)
            self.assertEqual(manifest["counts_by_difficulty"]["L3"], 40)
            self.assertEqual(
                manifest["counts_by_family_and_difficulty"]["normal"]["L1"], 20
            )
            self.assertEqual(
                manifest["counts_by_family_and_difficulty"]["new"]["L3"], 20
            )
            self.assertEqual(
                manifest["row_fields"]["metadata"]["arithmetic_family"], "str"
            )

            self.assertEqual(sum(manifest["label_histogram"].values()), 120)

    def test_generation_is_deterministic_for_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = MODULE_ROOT / "generate_dataset.py"
            output_a = Path(tmp_dir) / "a"
            output_b = Path(tmp_dir) / "b"

            for output_dir in (output_a, output_b):
                subprocess.run(
                    [
                        sys.executable,
                        str(script_path),
                        "--output-dir",
                        str(output_dir),
                        "--seed",
                        "13",
                        "--num-examples",
                        "90",
                    ],
                    check=True,
                )

            dataset_a = (output_a / "dataset.jsonl").read_text(encoding="utf-8")
            dataset_b = (output_b / "dataset.jsonl").read_text(encoding="utf-8")
            manifest_a = (output_a / "manifest.json").read_text(encoding="utf-8")
            manifest_b = (output_b / "manifest.json").read_text(encoding="utf-8")

            self.assertEqual(dataset_a, dataset_b)
            self.assertEqual(manifest_a, manifest_b)


if __name__ == "__main__":
    unittest.main()
