from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import prompts


class PromptTests(unittest.TestCase):
    def test_build_messages_includes_rules_and_expression(self) -> None:
        messages = prompts.build_messages("9 / 2 ## 3")

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("a / b = floor(a / b)", messages[0]["content"])
        self.assertIn("a ## b = abs(a - b)", messages[0]["content"])

        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Expression: 9 / 2 ## 3", messages[1]["content"])
        self.assertIn("<final_answer>integer</final_answer>", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
