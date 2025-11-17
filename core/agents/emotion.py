"""Agent for understanding customer emotions."""

from __future__ import annotations

from typing import Any

from core.agents.base import TextAgent
from core.schemas import ComplaintPayload


EMOTION_SYSTEM_PROMPT = """
أنت خبير في تحليل المشاعر والعواطف. مهمتك فهم مشاعر العملاء من خلال شكواهم.
"""


class EmotionAgent:
    """Agent that analyzes customer emotions."""

    def __init__(self, *, llm: Any, verbose: bool = False) -> None:
        self.agent = TextAgent(
            system_prompt=EMOTION_SYSTEM_PROMPT,
            llm=llm,
            verbose=verbose,
        )

    async def aanalyze_emotions(self, payload: ComplaintPayload, classification: str) -> str:
        """Analyze emotions and return emotion analysis text."""
        prompt = f"""
        قم بتحليل المشاعر في الشكوى التالية:

        الشركة: {payload.company.as_label()}
        نص الشكوى: {payload.complaint_text}
        التصنيف: {classification}

        المطلوب:
        1. حدد المشاعر الأساسية للعميل (مثل: غضب، إحباط، قلق، خيبة أمل، رضا، خوف)
        2. اشرح كيف يجب التعامل مع هذه المشاعر
        3. اقترح نبرة مناسبة للرد

        اكتب الإجابة بالعربية فقط.
        """
        return await self.agent.aask(prompt)

