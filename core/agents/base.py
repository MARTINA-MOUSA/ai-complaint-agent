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
        elif hasattr(self.llm, "complete"):
            raw = await asyncio.to_thread(self.llm.complete, full_prompt)
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
        if raw is None:
            raise ValueError("Agent returned empty payload.")

        if hasattr(raw, "text"):
            candidate = getattr(raw, "text", None) or ""
        elif hasattr(raw, "message") and raw.message:
            candidate = getattr(raw.message, "content", raw.message)
        elif hasattr(raw, "response") and raw.response:
            candidate = raw.response
        elif hasattr(raw, "candidates") and raw.candidates:
            parts = getattr(raw.candidates[0].content, "parts", [])
            candidate = "".join(getattr(part, "text", str(part)) for part in parts)
        else:
            candidate = str(raw) if raw else ""

        json_str = LlamaJsonAgent._extract_json(candidate)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Agent returned non-JSON payload: {candidate}") from exc

    @staticmethod
    def _extract_json(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            segments = cleaned.split("```")
            if len(segments) >= 2:
                cleaned = segments[1]
            cleaned = re.sub(r"^\s*json", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = cleaned.strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            return LlamaJsonAgent._balance_braces(cleaned)
        match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
        if match:
            return LlamaJsonAgent._balance_braces(match.group(0))
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return LlamaJsonAgent._balance_braces(cleaned[start : end + 1])
        return "{}"

    @staticmethod
    def _balance_braces(text: str) -> str:
        open_count = text.count("{")
        close_count = text.count("}")
        if close_count < open_count:
            text += "}" * (open_count - close_count)
        elif close_count > open_count:
            text = "{" * (close_count - open_count) + text
        return text


def run_sync(awaitable: Any) -> Any:
    """Helper to execute async code when sync interface is required."""
    return asyncio.get_event_loop().run_until_complete(awaitable)

