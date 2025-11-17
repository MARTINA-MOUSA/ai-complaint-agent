"""Policy guard to ensure final response respects company constraints."""

from __future__ import annotations

from typing import Any

from core.agents.base import LlamaJsonAgent
from core.prompts.templates import build_policy_prompt
from core.schemas import ClassificationResult, ComplaintPayload, ResolutionBundle


POLICY_SYSTEM_PROMPT = """
انتِ Claude-4-Opus. راجعي المخرجات للتأكد من توافقها مع سياسات الشركة
ونبرة العلامة التجارية، وأعيدي أي تعديلات بصيغة JSON في حال الضرورة.
"""


class PolicyAgent:
    def __init__(self, *, llm: Any, verbose: bool = False) -> None:
        self.agent = LlamaJsonAgent(
            system_prompt=POLICY_SYSTEM_PROMPT,
            llm=llm,
            verbose=verbose,
        )

    async def areview(
        self,
        payload: ComplaintPayload,
        classification: ClassificationResult,
        resolution: ResolutionBundle,
    ) -> dict:
        prompt = build_policy_prompt(
            company=payload.company.as_label(),
            notes=payload.notes,
            summary=classification.summary,
            emotions=classification.emotions,
            strategy=[step.model_dump() for step in resolution.strategy],
            formal_reply=resolution.formal_reply,
        )
        return await self.agent.aask(prompt)

