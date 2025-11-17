"""Configuration helpers for the AI complaint agent."""

from __future__ import annotations

from functools import lru_cache
import asyncio
from types import SimpleNamespace
from typing import Literal, Optional

import google.generativeai as genai
from llama_index.llms.openai import OpenAI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_api_key: str = Field("", alias="LLM_API_KEY")
    llm_provider: Literal["openai", "gemini"] = Field("gemini", alias="LLM_PROVIDER")
    llm_model: str = Field("gemini-2.5-flash", alias="LLM_MODEL")
    llm_base_url: Optional[str] = Field(default=None, alias="LLM_BASE_URL")

    backend_host: str = Field("0.0.0.0", alias="BACKEND_HOST")
    backend_port: int = Field(8080, alias="BACKEND_PORT")

    def build_llm(self):
        if not self.llm_api_key:
            raise ValueError("LLM_API_KEY missing. Please set it in your environment.")
        if self.llm_provider == "gemini":
            return _GeminiWrapper(model=self.llm_model, api_key=self.llm_api_key, temperature=0.2)
        return OpenAI(
            model=self.llm_model,
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            temperature=0.2,
        )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()  # type: ignore[arg-type]


class _GeminiWrapper:
    """Minimal wrapper exposing complete/acomplete for Gemini via google.generativeai."""

    def __init__(self, model: str, api_key: str, temperature: float = 0.2) -> None:
        genai.configure(api_key=api_key)
        # Normalize model name: add 'models/' prefix if missing
        if not model.startswith("models/"):
            model_name = f"models/{model}"
        else:
            model_name = model
        self._model = genai.GenerativeModel(model_name=model_name)
        self._generation_config = {"temperature": temperature}

    def complete(self, prompt: str):
        text = self._generate_text(prompt)
        return SimpleNamespace(text=text)

    async def acomplete(self, prompt: str):
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, self._generate_text, prompt)
        return SimpleNamespace(text=text)

    def _generate_text(self, prompt: str) -> str:
        response = self._model.generate_content(prompt, generation_config=self._generation_config)
        if hasattr(response, "text") and response.text:
            return response.text
        if response.candidates:
            parts = response.candidates[0].content.parts
            return "".join(getattr(part, "text", str(part)) for part in parts)
        return ""

