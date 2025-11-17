"""Base class for LlamaIndex powered agents."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, Optional

class LlamaJsonAgent:
    """Thin wrapper around an LlamaIndex LLM that enforces JSON output."""

    def __init__(
        self,
        *,
        system_prompt: str,
        llm: Optional[Any] = None,
        verbose: bool = False,
    ) -> None:
        if llm is None:
            raise ValueError("llm is required for LlamaJsonAgent")
        self.llm = llm
        self.system_prompt = system_prompt
        self.verbose = verbose

    async def aask(self, prompt: str) -> Dict[str, Any]:
        full_prompt = f"{self.system_prompt}\n\n{prompt}"
        if hasattr(self.llm, "acomplete"):
            raw = await self.llm.acomplete(full_prompt)
        else:
            raise AttributeError("LLM does not support async completion")
        return self._ensure_json(raw)

    def ask(self, prompt: str) -> Dict[str, Any]:
        full_prompt = f"{self.system_prompt}\n\n{prompt}"
        if hasattr(self.llm, "complete"):
            raw = self.llm.complete(full_prompt)
        else:
            raise AttributeError("LLM does not support sync completion")
        return self._ensure_json(raw)

    @staticmethod
    def _ensure_json(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if hasattr(raw, "text"):
            candidate = raw.text
        elif hasattr(raw, "message"):
            candidate = getattr(raw.message, "content", raw.message)
        elif hasattr(raw, "response"):
            candidate = raw.response
        else:
            candidate = str(raw)
        json_str = LlamaJsonAgent._extract_json(candidate)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Agent returned non-JSON payload: {candidate}") from exc

    @staticmethod
    def _extract_json(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            return cleaned
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return match.group(0)
        return cleaned


def run_sync(awaitable: Any) -> Any:
    """Helper to execute async code when sync interface is required."""
    return asyncio.get_event_loop().run_until_complete(awaitable)

