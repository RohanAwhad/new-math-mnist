from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import generate_dataset


class GenerateDatasetTests(unittest.TestCase):
    def test_allocate_level_counts_respects_total_and_s1_cap(self) -> None:
        counts = generate_dataset.allocate_level_counts(1_000_000)

        self.assertEqual(sum(counts.values()), 1_000_000)
        self.assertLessEqual(
            counts[generate_dataset.LEVEL_S1], generate_dataset.MAX_S1_EXPRESSIONS
        )

    def test_generate_s1_primitive_returns_empty_when_size_zero(self) -> None:
        samples = generate_dataset.generate_s1_primitive(0, random.Random(7), set())
        self.assertEqual(samples, [])

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

            first_row = json.loads(dataset_lines[0])
            self.assertEqual(
                set(first_row.keys()), {"input", "expected_output", "metadata"}
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["generated_num_examples"], 120)
            self.assertEqual(manifest["files"], ["dataset.jsonl", "manifest.json"])


if __name__ == "__main__":
    unittest.main()
