from __future__ import annotations

from contracts import ChatMessage

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are evaluating expressions in a synthetic arithmetic system.

Rules:
1) a ## b = abs(a - b)
2) a @@ b = max(a, b)
3) a $$ b = min(a, b)

Evaluation order: strictly left-to-right. No precedence. No parentheses.

You may think through the problem step by step before responding.

Return your final answer as exactly one XML tag:
<final_answer>[0-9]</final_answer>
"""

USER_TEMPLATE = """Expression: {expression}
Return <final_answer>[0-9]</final_answer>."""


def build_messages(expression: str) -> list[ChatMessage]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(expression=expression)},
    ]
