from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

import new_math_ops.llm_client as llm_client  # noqa: E402


class LiteLLMClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_uses_temperature_default_one(self) -> None:
        captured: dict[str, object] = {}

        async def fake_acompletion(**kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return {
                "choices": [
                    {
                        "message": {
                            "content": "<final_answer>7</final_answer>",
                        }
                    }
                ]
            }

        client = llm_client.LiteLLMClient(model="openai/test", acompletion_fn=fake_acompletion)
        response_text = await client.complete([{"role": "user", "content": "x"}])

        self.assertEqual(response_text, "<final_answer>7</final_answer>")
        self.assertEqual(captured["temperature"], 1.0)

    async def test_complete_flattens_list_content(self) -> None:
        async def fake_acompletion(**_: object) -> dict[str, object]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "<final_answer>"},
                                {"type": "text", "text": "4"},
                                {"type": "text", "text": "</final_answer>"},
                            ]
                        }
                    }
                ]
            }

        client = llm_client.LiteLLMClient(model="openai/test", acompletion_fn=fake_acompletion)
        response_text = await client.complete([{"role": "user", "content": "x"}])

        self.assertEqual(response_text, "<final_answer>4</final_answer>")


if __name__ == "__main__":
    unittest.main()
