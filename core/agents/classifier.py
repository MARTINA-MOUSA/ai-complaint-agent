"""Classifier agent enriches the complaint details."""

from __future__ import annotations

from typing import Any

from core.agents.base import LlamaJsonAgent
from core.prompts.templates import build_classifier_prompt
from core.schemas import ClassificationResult, ComplaintCategory, ComplaintPayload


CLASSIFIER_SYSTEM_PROMPT = """
انتِ نموذج ذكاء اصطناعي، خبيرة تجربة عملاء. حلّلي الشكوى بعمق
واستخرجي الملخص والمشاعر ومؤشر المخاطر. استمعي لتوجيه الراوتر لكن
صحّحيه إذا كان خاطئًا. أجيبي بـ JSON صالح.
"""


class ClassifierAgent:
    def __init__(self, *, llm: Any, verbose: bool = False) -> None:
        self.agent = LlamaJsonAgent(
            system_prompt=CLASSIFIER_SYSTEM_PROMPT,
            llm=llm,
            verbose=verbose,
        )

    async def aclassify(
        self,
        payload: ComplaintPayload,
        routed_category: ComplaintCategory,
    ) -> ClassificationResult:
        prompt = build_classifier_prompt(
            complaint=payload.complaint_text,
            company=payload.company.as_label(),
            notes=payload.notes,
            routed_category=routed_category.value,
        )
        data = await self.agent.aask(prompt)
        return ClassificationResult(
            category=ComplaintCategory(data.get("category", routed_category.value)),
            summary=data.get("summary", "لم يتمكن النظام من توليد ملخص."),
            emotions=list(data.get("emotions", [])),
            risk_level=data.get("risk_level", "medium"),
        )

