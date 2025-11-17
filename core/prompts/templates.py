"""Prompt builders leveraging the shared persona snippets."""

from __future__ import annotations

from textwrap import dedent

from .base import (
    BASE_PERSONA,
    FORMAL_REPLY_STYLE,
    JSON_STYLE,
    OUTPUT_REQUIREMENTS,
    STRATEGY_TEMPLATE,
)


def build_router_prompt(complaint: str, company: str, notes: str | None = None) -> str:
    extra = f"\nملاحظات إضافية: {notes}" if notes else ""
    return dedent(
        f"""
        {BASE_PERSONA}
        {OUTPUT_REQUIREMENTS}

        بيانات الطلب:
        - الشركة: {company}
        - نص الشكوى: {complaint}
        {extra}

        المطلوب:
        1. استنتج نوع الشكوى (delivery_issue | payment_issue | technical_issue |
           general_inquiry | return_exchange | other).
        2. قيّم الثقة (0-1).
        3. فسّر سبب التصنيف في جملة قصيرة.

        أجب بصيغة JSON:
        {{
          "category": "...",
          "confidence": 0.00,
          "rationale": "..."
        }}
        """
    ).strip()


def build_classifier_prompt(
    complaint: str,
    company: str,
    notes: str | None,
    routed_category: str,
) -> str:
    extra = f"\nملاحظات إضافية: {notes}" if notes else ""
    return dedent(
        f"""
        {BASE_PERSONA}
        {OUTPUT_REQUIREMENTS}
        {JSON_STYLE}

        المهمة: تحليل الشكوى التالية بالتفصيل وضمان أن التصنيف النهائي يطابق
        أي سياسات إضافية. التصنيف المبدئي من الروبوت: {routed_category}.

        الشركة: {company}
        نص الشكوى: {complaint}
        {extra}

        أجب بصيغة JSON مع الحقول:
        {{
          "category": "...",
          "summary": "...",
          "emotions": ["غضب", "قلق", ...],
          "risk_level": "low|medium|high"
        }}
        """
    ).strip()


def build_resolution_prompt(
    complaint: str,
    company: str,
    notes: str | None,
    category: str,
    summary: str,
    emotions: list[str],
) -> str:
    extra = f"\nملاحظات إضافية: {notes}" if notes else ""
    emotion_line = "، ".join(emotions) if emotions else "غير محدد"
    return dedent(
        f"""
        {BASE_PERSONA}
        {OUTPUT_REQUIREMENTS}
        {STRATEGY_TEMPLATE}
        {FORMAL_REPLY_STYLE}

        السياق:
        - الشركة: {company}
        - نوع الشكوى: {category}
        - الملخص: {summary}
        - المشاعر: {emotion_line}
        {extra}

        أجب بصيغة JSON:
        {{
          "strategy": [
             {{
               "action_title": "...",
               "owner_role": "...",
               "timeline": "...",
               "success_metric": "..."
             }}
          ],
          "formal_reply": "..."
        }}
        """
    ).strip()


def build_policy_prompt(
    company: str,
    notes: str | None,
    summary: str,
    emotions: list[str],
    strategy: list[dict],
    formal_reply: str,
) -> str:
    extra = f"\nسياسات إضافية: {notes}" if notes else ""
    return dedent(
        f"""
        {BASE_PERSONA}
        {OUTPUT_REQUIREMENTS}

        راجع النتيجة التالية طبقًا لسياسات الشركة.
        شركة: {company}{extra}

        الملخص: {summary}
        المشاعر: {', '.join(emotions)}
        الاستراتيجية: {strategy}
        الرد الرسمي: {formal_reply}

        المطلوب:
        - عدّل الصياغة إذا لزم الأمر.
        - تأكد من أن الرد يحتوي على خطوات التنفيذ كاملة.

        أجب بصيغة JSON:
        {{
          "summary": "...",
          "emotions": [...],
          "strategy": [...],
          "formal_reply": "..."
        }}
        """
    ).strip()

