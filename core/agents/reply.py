"""Reply agent using LlamaIndex."""

from __future__ import annotations

from typing import Any

from core.agents.base import LlamaIndexAgent
from core.schemas import ComplaintPayload


REPLY_SYSTEM_PROMPT = """
أنت خبير في كتابة ردود خدمة العملاء. مهمتك كتابة ردود رسمية، لطيفة، وتعاطفية.
أجب دائماً بالعربية فقط.
"""


class ReplyAgent:
    """Agent that creates formal customer reply using LlamaIndex."""

    def __init__(self, *, llm: Any, verbose: bool = False) -> None:
        self.agent_wrapper = LlamaIndexAgent(
            llm=llm,
            system_prompt=REPLY_SYSTEM_PROMPT,
            verbose=verbose,
        )

    async def acreate_reply(
        self,
        payload: ComplaintPayload,
        classification: str,
        emotions: str,
        strategy: str,
    ) -> str:
        """Create formal reply and return reply text."""
        extra = f"\nملاحظات إضافية: {payload.notes}" if payload.notes else ""
        message = f"""
        قم بكتابة رد رسمي للعميل بناءً على التحليل التالي:

        الشركة: {payload.company.as_label()}
        نص الشكوى: {payload.complaint_text}
        التصنيف: {classification}
        تحليل المشاعر: {emotions}
        خطة المعالجة: {strategy}
        {extra}

        المطلوب:
        اكتب ردًا رسميًا للعميل بالعربية يتسم باللباقة والتعاطف ويوضح الحل. يجب أن:
        - يشكر العميل على تواصله
        - يعترف بمشاعره
        - يشرح الإجراءات التي ستتخذها الشركة
        - يقدم جدولًا زمنيًا واقعيًا
        - يقدم خيارات للتواصل المستقبلي

        اكتب الرد بالعربية فقط، واجعله 3-4 فقرات.
        """
        return await self.agent_wrapper.achat(message)
