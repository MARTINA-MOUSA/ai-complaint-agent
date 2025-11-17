"""Simplified orchestration for text-based output."""

from __future__ import annotations

from typing import Any, Optional

from core.agents.base import TextAgent
from core.config import AppSettings
from core.prompts.templates import build_analysis_prompt
from core.schemas import ComplaintPayload
from core.services.logging import get_logger

logger = get_logger(__name__)


class ComplaintOrchestrator:
    """Simplified orchestrator that returns plain Arabic text."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        llm: Optional[Any] = None,
        verbose_agents: bool = False,
    ) -> None:
        self.settings = settings
        if llm is None:
            llm = settings.build_llm()
        self.llm = llm

        # Create a single text agent
        system_prompt = "أنت مساعد ذكي متخصص في تحليل شكاوى العملاء بالعربية."
        self.agent = TextAgent(
            system_prompt=system_prompt,
            llm=self.llm,
            verbose=verbose_agents,
        )

    async def aanalyze(self, payload: ComplaintPayload) -> str:
        """Analyze complaint and return plain Arabic text."""
        logger.info("orchestrator.start", complaint=len(payload.complaint_text))

        # Build the analysis prompt
        prompt = build_analysis_prompt(
            complaint=payload.complaint_text,
            company=payload.company.as_label(),
            notes=payload.notes,
        )

        # Get analysis as plain text
        analysis_text = await self.agent.aask(prompt)

        logger.info("orchestrator.end", response_length=len(analysis_text))
        return analysis_text
