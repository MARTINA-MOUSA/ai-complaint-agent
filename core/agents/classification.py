"""Classification agent using LlamaIndex."""

from __future__ import annotations

from typing import Any

from core.agents.base import LlamaIndexAgent
from core.schemas import ComplaintPayload


CLASSIFICATION_SYSTEM_PROMPT = """
أنت خبير في تصنيف شكاوى العملاء. مهمتك تحديد نوع الشكوى بدقة.
أجب دائماً بالعربية فقط.
"""


class ClassificationAgent:
    """Agent that classifies the complaint type using LlamaIndex."""

    def __init__(self, *, llm: Any, verbose: bool = False) -> None:
        self.agent_wrapper = LlamaIndexAgent(
            llm=llm,
            system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
            verbose=verbose,
        )

    async def aclassify(self, payload: ComplaintPayload) -> str:
        """Classify the complaint and return classification text."""
        message = f"""
        قم بتصنيف الشكوى التالية:

        الشركة: {payload.company.as_label()}
        نص الشكوى: {payload.complaint_text}

        المطلوب:
        1. حدد نوع الشكوى من: (مشكلة في التوصيل | مشكلة في الدفع | مشكلة تقنية | استفسار عام | استرجاع/استبدال | أخرى)
        2. اشرح سبب التصنيف في جملة أو جملتين بالعربية.

        اكتب الإجابة بالعربية فقط.
        """
        return await self.agent_wrapper.achat(message)
