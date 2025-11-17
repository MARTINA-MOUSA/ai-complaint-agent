"""Base agent for text-based output."""

from __future__ import annotations

import asyncio
from typing import Any, Optional


class TextAgent:
    """Agent that returns plain text output instead of JSON."""

    def __init__(
        self,
        *,
        system_prompt: str,
        llm: Optional[Any] = None,
        verbose: bool = False,
    ) -> None:
        if llm is None:
            raise ValueError("llm is required for TextAgent")
        self.llm = llm
        self.system_prompt = system_prompt
        self.verbose = verbose

    async def aask(self, prompt: str) -> str:
        """Ask the LLM and return plain text."""
        full_prompt = f"{self.system_prompt}\n\n{prompt}"

        # Try async first, fallback to sync in thread pool
        if hasattr(self.llm, "acomplete"):
            raw = await self.llm.acomplete(full_prompt)
        elif hasattr(self.llm, "complete"):
            raw = await asyncio.to_thread(self.llm.complete, full_prompt)
        else:
            raise AttributeError("LLM does not support async or sync completion")

        return self._extract_text(raw)

    def ask(self, prompt: str) -> str:
        """Ask the LLM synchronously and return plain text."""
        full_prompt = f"{self.system_prompt}\n\n{prompt}"

        if hasattr(self.llm, "complete"):
            raw = self.llm.complete(full_prompt)
        else:
            raise AttributeError("LLM does not support sync completion")

        return self._extract_text(raw)

    @staticmethod
    def _extract_text(raw: Any) -> str:
        """Extract text from various LLM response formats."""
        if isinstance(raw, str):
            return raw.strip()

        if raw is None:
            raise ValueError("Agent returned empty payload.")

        # Extract text from different response formats
        text = ""
        if hasattr(raw, "candidates"):
            try:
                parts = raw.candidates[0].content.parts
                text = "".join(getattr(p, "text", str(p)) for p in parts)
            except Exception:
                text = str(raw)
        elif hasattr(raw, "text"):
            text = getattr(raw, "text", "")
        elif hasattr(raw, "response"):
            text = getattr(raw, "response", "")
        elif hasattr(raw, "message") and raw.message:
            text = getattr(raw.message, "content", str(raw.message))
        else:
            text = str(raw)

        # Clean up namespace wrappers if present
        if "namespace(" in text:
            # Try to extract content from namespace(response='...') or namespace(text='...')
            if "response=" in text or "text=" in text:
                # Find the JSON or text content
                start = text.find("{")
                if start == -1:
                    # No JSON, try to find quoted content
                    import re
                    match = re.search(r"(?:response|text)\s*=\s*['\"](.*?)['\"]", text, re.DOTALL)
                    if match:
                        text = match.group(1)
                else:
                    # Extract JSON object
                    brace_count = 0
                    for i in range(start, len(text)):
                        if text[i] == "{":
                            brace_count += 1
                        elif text[i] == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                text = text[start:i+1]
                                break

        if not text.strip():
            raise ValueError("Agent returned empty or invalid response.")

        return text.strip()
