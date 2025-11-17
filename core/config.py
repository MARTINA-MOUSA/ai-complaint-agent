"""Configuration helpers for the AI complaint agent."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from llama_index.llms.openai import OpenAI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_api_key: str = Field("", alias="LLM_API_KEY")
    llm_provider: Literal["openai"] = Field("openai", alias="LLM_PROVIDER")
    llm_model: str = Field("gpt-4o-mini", alias="LLM_MODEL")
    llm_base_url: Optional[str] = Field(default=None, alias="LLM_BASE_URL")

    backend_host: str = Field("0.0.0.0", alias="BACKEND_HOST")
    backend_port: int = Field(8080, alias="BACKEND_PORT")

    def build_llm(self) -> OpenAI:
        if not self.llm_api_key:
            raise ValueError("LLM_API_KEY missing. Please set it in your environment.")
        return OpenAI(
            model=self.llm_model,
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            temperature=0.2,
        )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()  # type: ignore[arg-type]

