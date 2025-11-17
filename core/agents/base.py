import json
import re
from typing import Any, Dict

class LlamaJsonAgent:
    """محسّنة: تحويل أي output من LLM لـ JSON صالح."""

    @staticmethod
    def _ensure_json(raw: Any) -> Dict[str, Any]:
        """Normalizes all LLM output formats to a JSON dict."""
        if isinstance(raw, dict):
            return raw

        if raw is None:
            raise ValueError("Agent returned empty payload.")

        # تحويل أي نوع output لـ string candidate
        candidate = ""
        if hasattr(raw, "candidates"):
            try:
                parts = raw.candidates[0].content.parts
                candidate = "".join(getattr(p, "text", str(p)) for p in parts)
            except Exception:
                candidate = str(raw)
        elif hasattr(raw, "text"):
            candidate = getattr(raw, "text", "")
        elif hasattr(raw, "response"):
            candidate = getattr(raw, "response", "")
        elif hasattr(raw, "message") and raw.message:
            candidate = getattr(raw.message, "content", str(raw.message))
        else:
            candidate = str(raw)

        if not candidate.strip():
            raise ValueError("Agent returned empty or invalid response.")

        # استخراج JSON
        json_str = LlamaJsonAgent._extract_json(candidate)

        # تحويل علامات الاقتباس المفردة لـ مزدوجة
        json_str = json_str.replace("'", '"')

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as exc:
            print("Failed JSON preview:", json_str[:500])
            # fallback مؤقت
            return {}

    @staticmethod
    def _extract_json(text: str) -> str:
        """استخراج JSON من أي نص."""
        cleaned = text.strip()

        # إزالة namespace(...) wrappers
        if "namespace(" in cleaned:
            json_start = cleaned.find("{")
            if json_start != -1:
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
                    cleaned = cleaned[json_start:]

        # استخراج أول JSON كامل
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            extracted = cleaned[start:end+1]
            extracted = re.sub(r',(\s*[}\]])', r'\1', extracted)  # إزالة trailing commas
            return LlamaJsonAgent._balance_braces(extracted)

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
