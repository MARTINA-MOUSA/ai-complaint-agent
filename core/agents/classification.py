"""Agent for classifying complaints."""

from __future__ import annotations

from typing import Any

from core.agents.base import TextAgent
from core.schemas import ComplaintPayload


CLASSIFICATION_SYSTEM_PROMPT = """
أنت خبير في تصنيف شكاوى العملاء. مهمتك تحديد نوع الشكوى بدقة.
"""


class ClassificationAgent:
    """Agent that classifies the complaint type."""

    def __init__(self, *, llm: Any, verbose: bool = False) -> None:
        self.agent = TextAgent(
            system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
            llm=llm,
            verbose=verbose,
        )

    async def aclassify(self, payload: ComplaintPayload) -> str:
        """Classify the complaint and return classification text."""
        prompt = f"""
        قم بتصنيف الشكوى التالية:

        الشركة: {payload.company.as_label()}
        نص الشكوى: {payload.complaint_text}

        المطلوب:
        1. حدد نوع الشكوى من: (مشكلة في التوصيل | مشكلة في الدفع | مشكلة تقنية | استفسار عام | استرجاع/استبدال | أخرى)
        2. اشرح سبب التصنيف في جملة أو جملتين بالعربية.

        اكتب الإجابة بالعربية فقط.
        """
        return await self.agent.aask(prompt)

