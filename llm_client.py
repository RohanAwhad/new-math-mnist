from __future__ import annotations

from collections.abc import Awaitable, Callable


Message = dict[str, str]
ACompletionFn = Callable[..., Awaitable[object]]


def _default_acompletion(**kwargs: object) -> Awaitable[object]:
    from litellm import acompletion

    return acompletion(**kwargs)


def _extract_content(response: object) -> str:
    choices: object
    if isinstance(response, dict):
        choices = response["choices"]
    else:
        choices = getattr(response, "choices")

    first_choice = choices[0]
    if isinstance(first_choice, dict):
        message = first_choice["message"]
    else:
        message = getattr(first_choice, "message")

    if isinstance(message, dict):
        content = message["content"]
    else:
        content = getattr(message, "content")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text_parts.append(str(part.get("text", "")))
            else:
                text_parts.append(str(getattr(part, "text", "")))
        return "".join(text_parts)

    return str(content)


class LiteLLMClient:
    def __init__(
        self,
        *,
        model: str,
        temperature: float = 1.0,
        max_tokens: int = 64,
        acompletion_fn: ACompletionFn | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._acompletion_fn = acompletion_fn or _default_acompletion

    async def complete(self, messages: list[Message]) -> str:
        response = await self._acompletion_fn(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return _extract_content(response).strip()
