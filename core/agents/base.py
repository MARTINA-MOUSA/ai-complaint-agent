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

        # Try async first, fallback to sync in thread pool
        if hasattr(self.llm, "acomplete"):
            raw = await self.llm.acomplete(full_prompt)
        elif hasattr(self.llm, "complete"):
            raw = await asyncio.to_thread(self.llm.complete, full_prompt)
        else:
            raise AttributeError("LLM does not support async or sync completion")

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

        # 2) Generic "text" field (for GeminiWrapper SimpleNamespace)
        elif hasattr(raw, "text"):
            text_val = getattr(raw, "text", None)
            candidate = text_val if text_val else ""

        # 3) Response.message.content
        elif hasattr(raw, "message") and raw.message:
            candidate = getattr(raw.message, "content", str(raw.message))

        # 4) Fallback
        else:
            candidate = str(raw) if raw else ""

        # Validate we have content
        if not candidate or not candidate.strip():
            raise ValueError("Agent returned empty or invalid response. Raw: " + str(raw)[:200])

        # Extract JSON content
        json_str = LlamaJsonAgent._extract_json(candidate)

        try:
            return json.loads(json_str)
        except Exception as exc:
            raise ValueError(f"Agent returned non-JSON payload: {candidate}") from exc

    @staticmethod
    def _extract_json(text: str) -> str:
        cleaned = text.strip()

        # 1) استخراج أي بلوك بين ``` ```
        if "```" in cleaned:
            segments = cleaned.split("```")
            # بلوك JSON هو أي جزء يحتوي قوسين {}
            for seg in segments:
                if "{" in seg and "}" in seg:
                    cleaned = seg
                    break

            # إزالة كلمة json أو JSON في بداية البلوك
            cleaned = re.sub(r"^\s*(json)?", "", cleaned, flags=re.IGNORECASE).strip()

        # 2) قص أول JSON كامل في النص
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return LlamaJsonAgent._balance_braces(cleaned[start:end+1])

        # 3) fallback
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
