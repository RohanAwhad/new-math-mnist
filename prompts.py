from __future__ import annotations

from contracts import ChatMessage

PROMPT_VERSION = "v2"

SYSTEM_PROMPT = """You are evaluating expressions in a synthetic arithmetic system.

Rules:
1) a + b = a + b
2) a - b = a - b
3) a * b = a * b
4) a / b = floor(a / b)
5) a ## b = abs(a - b)
6) a @@ b = max(a, b)
7) a $$ b = min(a, b)

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
