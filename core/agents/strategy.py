"""Agent for creating resolution strategy."""

from __future__ import annotations

from typing import Any

from core.agents.base import TextAgent
from core.schemas import ComplaintPayload


STRATEGY_SYSTEM_PROMPT = """
أنت خبير في وضع خطط حل المشاكل. مهمتك إنشاء خطط عملية وقابلة للتنفيذ.
"""


class StrategyAgent:
    """Agent that creates resolution strategy."""

    def __init__(self, *, llm: Any, verbose: bool = False) -> None:
        self.agent = TextAgent(
            system_prompt=STRATEGY_SYSTEM_PROMPT,
            llm=llm,
            verbose=verbose,
        )

    async def acreate_strategy(
        self,
        payload: ComplaintPayload,
        classification: str,
        emotions: str,
    ) -> str:
        """Create resolution strategy and return strategy text."""
        extra = f"\nملاحظات إضافية: {payload.notes}" if payload.notes else ""
        prompt = f"""
        قم بإنشاء خطة حل للمشكلة التالية:

        الشركة: {payload.company.as_label()}
        نص الشكوى: {payload.complaint_text}
        التصنيف: {classification}
        تحليل المشاعر: {emotions}
        {extra}

        المطلوب:
        اكتب خطة حل عملية مكونة من 3-5 خطوات. لكل خطوة اذكر:
        - الإجراء المطلوب
        - المسؤول (مثل: فريق خدمة العملاء، قسم الشحن، إلخ)
        - الجدول الزمني (مثل: خلال 24 ساعة، خلال 3 أيام عمل)
        - معيار النجاح

        اكتب الإجابة بالعربية فقط.
        """
        return await self.agent.aask(prompt)

