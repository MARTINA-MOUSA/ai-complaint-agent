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

        # 3) Support "response" field as fallback (old format)
        elif hasattr(raw, "response"):
            response_val = getattr(raw, "response", None)
            candidate = response_val if response_val else ""

        # 4) Response.message.content
        elif hasattr(raw, "message") and raw.message:
            candidate = getattr(raw.message, "content", str(raw.message))

        # 5) Fallback
        else:
            candidate = str(raw) if raw else ""

        # Validate we have content
        if not candidate or not candidate.strip():
            raise ValueError("Agent returned empty or invalid response. Raw: " + str(raw)[:200])

        # Extract JSON content
        json_str = LlamaJsonAgent._extract_json(candidate)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as exc:
            # Show first 500 chars of the problematic JSON for debugging
            preview = json_str[:500] if len(json_str) > 500 else json_str
            raise ValueError(
                f"Agent returned invalid JSON. Error: {exc.msg} at line {exc.lineno}, col {exc.colno}. "
                f"JSON preview: {preview}... Original candidate: {candidate[:200]}..."
            ) from exc
        except Exception as exc:
            raise ValueError(f"Agent returned non-JSON payload: {candidate[:500]}...") from exc

    @staticmethod
    def _extract_json(text: str) -> str:
        cleaned = text.strip()

        # 0) Remove namespace(...) wrapper patterns
        # Handle patterns like: namespace(response='{...}') or namespace(text='{...}')
        # Simple approach: find the JSON object directly, ignoring wrapper text
        if "namespace(" in cleaned:
            # Extract content between quotes after response= or text=
            # Use a simpler approach: find the JSON object by looking for { and }
            json_start = cleaned.find("{")
            if json_start != -1:
                # Find the matching closing brace
                brace_count = 0
                for i in range(json_start, len(cleaned)):
                    if cleaned[i] == "{":
                        brace_count += 1
                    elif cleaned[i] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            cleaned = cleaned[json_start:i+1]
                            break
                else:
                    # No matching brace, use from { to end
                    cleaned = cleaned[json_start:]

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

        # 2) Remove any leading/trailing quotes or whitespace
        cleaned = cleaned.strip().strip("'\"")
        
        # 3) قص أول JSON كامل في النص
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            extracted = cleaned[start:end+1]
            # Convert escape sequences to actual characters (but preserve them inside JSON string values)
            # Only convert \n and \t that are outside of string values
            # Simple approach: replace common escape sequences
            extracted = extracted.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
            # Remove any trailing commas before closing braces
            extracted = re.sub(r',(\s*[}\]])', r'\1', extracted)
            return LlamaJsonAgent._balance_braces(extracted)

        # 4) fallback
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
