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
        messages = prompts.build_messages("3 ## 8 @@ 2")

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("a ## b = abs(a - b)", messages[0]["content"])
        self.assertIn("<final_answer><digit></final_answer>", messages[0]["content"])

        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("Expression: 3 ## 8 @@ 2", messages[1]["content"])

    def test_prompt_version_is_tag_contract_version(self) -> None:
        self.assertEqual(prompts.PROMPT_VERSION, "v2_final_answer_tag")


if __name__ == "__main__":
    unittest.main()
