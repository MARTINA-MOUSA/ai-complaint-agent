"""Routing agent selects the most likely complaint category."""

from __future__ import annotations

from typing import Any

from core.agents.base import LlamaJsonAgent
from core.prompts.templates import build_router_prompt
from core.schemas import ComplaintPayload, ComplaintCategory, RouterDecision


ROUTER_SYSTEM_PROMPT = """
انتِ نموذج ذكاء اصطناعي. دورك اختيار أفضل قسم داخلي للتعامل مع الشكوى.
التصنيفات المتاحة:
- delivery_issue (مشاكل التوصيل، التأخير، تلف الشحنة)
- payment_issue (الفوترة، المبالغ المستردة، البطاقات)
- technical_issue (تطبيق، موقع، تتبع)
- general_inquiry (أسئلة عامة، معلومات)
- return_exchange (مرتجع، استبدال)
- other (كل ما سبق غير مناسب)
أعيدي الناتج بصيغة JSON فقط.
"""


class RouterAgent:
    def __init__(self, *, llm: Any, verbose: bool = False) -> None:
        self.agent = LlamaJsonAgent(system_prompt=ROUTER_SYSTEM_PROMPT, llm=llm, verbose=verbose)

    async def aroute(self, payload: ComplaintPayload) -> RouterDecision:
        prompt = build_router_prompt(
            complaint=payload.complaint_text,
            company=payload.company.as_label(),
            notes=payload.notes,
        )
        data = await self.agent.aask(prompt)
        return RouterDecision(
            category=ComplaintCategory(data.get("category", ComplaintCategory.OTHER)),
            confidence=float(data.get("confidence", 0.5)),
            rationale=data.get("rationale", "غير محدد"),
        )

