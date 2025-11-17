"""Resolution agent generates strategy steps and formal reply."""

from __future__ import annotations

from typing import Any

from core.agents.base import LlamaJsonAgent
from core.prompts.templates import build_resolution_prompt
from core.schemas import ClassificationResult, ComplaintPayload, ResolutionBundle, StrategyStep


RESOLUTION_SYSTEM_PROMPT = """
انتِ Claude-4-Opus، مديرة عمليات قادرة على تحويل التحليل إلى خطة تنفيذية.
ضعي خطة بحد أدنى ثلاثة خطوات واضحة قابلة للقياس، ثم اكتبي ردًا رسميًا
باللغة العربية. التزمي بـ JSON صالح.
"""


class ResolutionAgent:
    def __init__(self, *, llm: Any, verbose: bool = False) -> None:
        self.agent = LlamaJsonAgent(
            system_prompt=RESOLUTION_SYSTEM_PROMPT,
            llm=llm,
            verbose=verbose,
        )

    async def aresolve(
        self,
        payload: ComplaintPayload,
        classification: ClassificationResult,
    ) -> ResolutionBundle:
        prompt = build_resolution_prompt(
            complaint=payload.complaint_text,
            company=payload.company.as_label(),
            notes=payload.notes,
            category=classification.category.value,
            summary=classification.summary,
            emotions=classification.emotions,
        )
        data = await self.agent.aask(prompt)
        strategy_payload = data.get("strategy", [])
        steps = [
            StrategyStep(**step)
            for step in strategy_payload
        ]
        return ResolutionBundle(strategy=steps, formal_reply=data.get("formal_reply", ""))

