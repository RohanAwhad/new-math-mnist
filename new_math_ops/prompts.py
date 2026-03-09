from __future__ import annotations

from .contracts import ChatMessage

PROMPT_VERSION = "v2"

SYSTEM_PROMPT = """You are evaluating expressions in a synthetic arithmetic system.

Evaluation order: strictly left-to-right for all operators.
Do not apply precedence. No parentheses.

You may think through the problem step by step before responding.

Return your final answer as exactly one XML tag:
<final_answer>integer</final_answer>
"""

USER_TEMPLATE = """Expression: {expression}
Return <final_answer>integer</final_answer>."""


def build_messages(expression: str) -> list[ChatMessage]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(expression=expression)},
    ]


__all__ = [
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "USER_TEMPLATE",
    "build_messages",
]
