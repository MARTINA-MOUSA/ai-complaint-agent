"""Strategy agent using LlamaIndex."""

from __future__ import annotations

from typing import Any

from core.agents.base import LlamaIndexAgent
from core.schemas import ComplaintPayload


STRATEGY_SYSTEM_PROMPT = """
أنت خبير في وضع خطط حل المشاكل. مهمتك إنشاء خطط عملية وقابلة للتنفيذ.
أجب دائماً بالعربية فقط.
"""


class StrategyAgent:
    """Agent that creates resolution strategy using LlamaIndex."""

    def __init__(self, *, llm: Any, verbose: bool = False) -> None:
        self.agent_wrapper = LlamaIndexAgent(
            llm=llm,
            system_prompt=STRATEGY_SYSTEM_PROMPT,
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
        message = f"""
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
        return await self.agent_wrapper.achat(message)
