"""Base class for LlamaIndex powered agents with robust JSON extraction."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Dict, Optional


class LlamaJsonAgent:
    """Thin wrapper around an LLM that enforces JSON output."""

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

        # Async-compatible LLMs
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
        """Normalizes all possible LLM output formats to a JSON dict."""

        # Already JSON
        if isinstance(raw, dict):
            return raw

        if raw is None:
            raise ValueError("Agent returned empty payload.")

        # ---- GeminiWrapper Support ----
        # Response from Gemini looks like:
        # {
        #   "candidates": [
        #       { "content": { "parts": [ {"text": "..."} ] } }
        #   ]
        # }
        candidate = ""

        # 1) Gemini candidates
        if hasattr(raw, "candidates"):
            try:
                parts = raw.candidates[0].content.parts
                candidate = "".join(
                    getattr(p, "text", str(p)) for p in parts
                )
            except Exception:
                candidate = str(raw)

        # 2) Generic "text" field
        elif hasattr(raw, "text"):
            candidate = raw.text or ""

        # 3) Response.message.content
        elif hasattr(raw, "message"):
            candidate = getattr(raw.message, "content", raw.message)

        # 4) Fallback
        else:
            candidate = str(raw)

        # Extract JSON content
        json_str = LlamaJsonAgent._extract_json(candidate)

        try:
            return json.loads(json_str)
        except Exception as exc:
            raise ValueError(f"Agent returned non-JSON payload: {candidate}") from exc

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract valid JSON from any messy model output."""
        cleaned = text.strip()

        # Strip code fences
        if cleaned.startswith("```"):
            segments = cleaned.split("```")
            if len(segments) >= 2:
                cleaned = segments[1]
            cleaned = re.sub(r"^\s*json", "", cleaned, flags=re.IGNORECASE).strip()

        cleaned = cleaned.strip()

        # If whole content is JSON
        if cleaned.startswith("{") and cleaned.endswith("}"):
            return LlamaJsonAgent._balance_braces(cleaned)

        # Find JSON inside text
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return LlamaJsonAgent._balance_braces(match.group(0))

        # Fallback (no JSON found)
        return "{}"

    @staticmethod
    def _balance_braces(text: str) -> str:
        """Fix unbalanced braces from model output."""
        open_count = text.count("{")
        close_count = text.count("}")

        if close_count < open_count:
            text += "}" * (open_count - close_count)
        elif close_count > open_count:
            text = "{" * (close_count - open_count) + text

        return text


def run_sync(awaitable: Any) -> Any:
    return asyncio.get_event_loop().run_until_complete(awaitable)
