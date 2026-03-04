from __future__ import annotations

PROMPT_VERSION = "v2_final_answer_tag"

SYSTEM_PROMPT = """You are evaluating expressions in a synthetic arithmetic system.

Rules:
1) a ## b = abs(a - b)
2) a @@ b = max(a, b)
3) a $$ b = min(a, b)

Evaluation order: strictly left-to-right. No precedence. No parentheses.

Return exactly one XML tag and nothing else:
<final_answer><digit></final_answer>

<digit> must be a single integer from 0 to 9.
Do not include any explanation or extra text.
"""

USER_TEMPLATE = """Expression: {expression}
Return only <final_answer><digit></final_answer>."""


def build_messages(expression: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(expression=expression)},
    ]
