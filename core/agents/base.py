"""Base agent using LlamaIndex framework."""

from __future__ import annotations

from typing import Any, Optional


class LlamaIndexAgent:
    """Base agent wrapper that uses LLM directly with system prompts."""

    def __init__(
        self,
        *,
        llm: Any,
        system_prompt: str,
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.verbose = verbose

    async def achat(self, message: str) -> str:
        """Chat with the agent asynchronously."""
        full_prompt = f"{self.system_prompt}\n\n{message}"
        result = await self.llm.acomplete(full_prompt)
        return self._extract_text(result)

    def chat(self, message: str) -> str:
        """Chat with the agent synchronously."""
        full_prompt = f"{self.system_prompt}\n\n{message}"
        result = self.llm.complete(full_prompt)
        return self._extract_text(result)

    @staticmethod
    def _extract_text(raw: Any) -> str:
        """Extract text from LLM response."""
        if isinstance(raw, str):
            return raw.strip()
        if hasattr(raw, "text"):
            return getattr(raw, "text", "").strip()
        if hasattr(raw, "response"):
            return getattr(raw, "response", "").strip()
        return str(raw).strip()
