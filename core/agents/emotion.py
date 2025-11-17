"""Emotion analysis agent using LlamaIndex."""

from __future__ import annotations

from typing import Any

from core.agents.base import LlamaIndexAgent
from core.schemas import ComplaintPayload


EMOTION_SYSTEM_PROMPT = """
أنت خبير في تحليل المشاعر والعواطف. مهمتك فهم مشاعر العملاء من خلال شكواهم.
أجب دائماً بالعربية فقط.
"""


class EmotionAgent:
    """Agent that analyzes customer emotions using LlamaIndex."""

    def __init__(self, *, llm: Any, verbose: bool = False) -> None:
        self.agent_wrapper = LlamaIndexAgent(
            llm=llm,
            system_prompt=EMOTION_SYSTEM_PROMPT,
            verbose=verbose,
        )

    async def aanalyze_emotions(self, payload: ComplaintPayload, classification: str) -> str:
        """Analyze emotions and return emotion analysis text."""
        message = f"""
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
        return await self.agent_wrapper.achat(message)
