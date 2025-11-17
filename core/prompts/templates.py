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
        1. استنتج نوع الشكوى من القائمة التالية:
           - delivery_issue (مشكلة في التوصيل)
           - payment_issue (مشكلة في الدفع)
           - technical_issue (مشكلة تقنية)
           - general_inquiry (استفسار عام)
           - return_exchange (استرجاع/استبدال)
           - other (أخرى)
        2. قيّم مستوى الثقة في التصنيف (رقم بين 0.0 و 1.0).
        3. اشرح سبب التصنيف في جملة واحدة بالعربية.

        IMPORTANT: أجب بصيغة JSON فقط، بدون markdown code blocks. استخدم المفتاح "category" بالإنجليزية والقيمة بالعربية.
        {{
          "category": "delivery_issue",
          "confidence": 0.95,
          "rationale": "الشكوى تتعلق بتأخر وصول الطلب"
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

        المطلوب:
        1. أكد أو صحح التصنيف (delivery_issue | payment_issue | technical_issue | general_inquiry | return_exchange | other).
        2. اكتب ملخصًا موجزًا للمشكلة بالعربية (2-3 جمل).
        3. حدد المشاعر الأساسية للعميل (مثل: غضب، إحباط، قلق، خيبة أمل، رضا).
        4. قيّم مستوى المخاطر (low | medium | high).

        IMPORTANT: أجب بصيغة JSON فقط، بدون markdown code blocks.
        {{
          "category": "delivery_issue",
          "summary": "العميل يشكو من تأخر وصول الطلب لمدة 5 أيام...",
          "emotions": ["غضب", "إحباط"],
          "risk_level": "high"
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

        المطلوب:
        1. أنشئ خطة حل مكونة من 3-5 خطوات عملية.
        2. اكتب ردًا رسميًا للعميل بالعربية.

        IMPORTANT: أجب بصيغة JSON فقط، بدون markdown code blocks.
        {{
          "strategy": [
             {{
               "action_title": "التحقق من حالة الطلب",
               "owner_role": "فريق خدمة العملاء",
               "timeline": "خلال 24 ساعة",
               "success_metric": "تأكيد وصول الطلب للعميل"
             }},
             {{
               "action_title": "متابعة مع شركة الشحن",
               "owner_role": "قسم الشحن",
               "timeline": "خلال 48 ساعة",
               "success_metric": "حل مشكلة التأخير"
             }}
          ],
          "formal_reply": "عزيزي العميل، نشكرك على تواصلك معنا. نحن نتفهم إحباطك من تأخر وصول الطلب..."
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
        1. راجع الملخص والمشاعر والاستراتيجية والرد الرسمي.
        2. عدّل الصياغة إذا لزم الأمر لضمان الامتثال لسياسات الشركة.
        3. تأكد من أن الرد الرسمي يحتوي على جميع خطوات التنفيذ بشكل واضح.
        4. تأكد من أن جميع النصوص بالعربية فقط.

        IMPORTANT: أجب بصيغة JSON فقط، بدون markdown code blocks.
        {{
          "summary": "الملخص المعدل...",
          "emotions": ["غضب", "إحباط"],
          "strategy": [
            {{"action_title": "...", "owner_role": "...", "timeline": "...", "success_metric": "..."}}
          ],
          "formal_reply": "الرد الرسمي المعدل..."
        }}
        """
    ).strip()

