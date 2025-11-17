import asyncio
import json
import os
from typing import Dict, List

import streamlit as st

from core.config import get_settings
from core.schemas import ComplaintPayload, CompanyDetails
from core.services.orchestrator import ComplaintOrchestrator

st.set_page_config(page_title="AI Complaint Agent", layout="wide")

st.title("منصة تحليل الشكاوى بالذكاء الاصطناعي")
st.caption("تصنيف | فهم المشاعر | خطة معالجة | رد رسمي")


def _ensure_llm_key() -> None:
    if os.environ.get("LLM_API_KEY"):
        return
    if "LLM_API_KEY" in st.secrets:
        os.environ["LLM_API_KEY"] = st.secrets["LLM_API_KEY"]


@st.cache_resource
def get_orchestrator() -> ComplaintOrchestrator:
    _ensure_llm_key()
    settings = get_settings()
    return ComplaintOrchestrator(settings=settings)


def analyze_locally(payload: ComplaintPayload):
    orchestrator = get_orchestrator()
    return asyncio.run(orchestrator.aanalyze(payload))


with st.sidebar:
    st.header("بيانات الشركة")
    company_name = st.text_input("اسم الشركة", value="سريع إكسبرس")
    service_name = st.text_input("الخدمة / المنتج", value="توصيل متاجر")
    notes = st.text_area("سياسات إضافية", value="يجب الرد خلال 24 ساعة كحد أقصى.")
    st.info("تأكدي من إضافة مفتاح LLM في `Secrets` داخل Streamlit Cloud.")
    st.markdown("---")
    st.write("أدخل نص الشكوى في النموذج الرئيسي.")

complaint_text = st.text_area(
    "نص الشكوى",
    height=200,
    placeholder="اكتب الشكوى كما وصلت من العميل...",
)

if "history" not in st.session_state:
    st.session_state.history = []


if st.button("حلّل الشكوى", use_container_width=True) and complaint_text:
    payload_model = ComplaintPayload(
        complaint_text=complaint_text,
        company=CompanyDetails(name=company_name, service=service_name or None),
        notes=notes or None,
    )

    with st.spinner("جارٍ تحليل الشكوى عبر الوكلاء محليًا..."):
        analysis = analyze_locally(payload_model)

    st.success("تم توليد الخطة والرد بنجاح.")
    st.info(
        f"التصنيف | {analysis.category.value} | "
        f"مستوى المخاطر | {analysis.risk_level} | "
        f"الثقة | {analysis.router.confidence:.2f}"
    )

    st.subheader("١. ملخص المشكلة")
    st.write(f"ملخص | {analysis.summary}")

    st.subheader("٢. المشاعر الأساسية")
    emotions_line = "، ".join(analysis.emotions) if analysis.emotions else "غير محدد"
    st.write(f"مشاعر | {emotions_line}")

    st.subheader("٣. إستراتيجية الحل المقترحة")
    strategy_lines: List[str] = [
        f"- {step.action_title} | المسؤول: {step.owner_role} | "
        f"الجدول: {step.timeline} | المعيار: {step.success_metric}"
        for step in analysis.strategy
    ]
    st.markdown("\n".join(strategy_lines))

    st.subheader("٤. الرد الرسمي")
    st.write(analysis.formal_reply)

    st.session_state.history.append(
        {
            "summary": analysis.summary,
            "category": analysis.category.value,
            "reply": analysis.formal_reply,
        }
    )

if st.session_state.history:
    st.markdown("## السجل الأخير")
    for idx, item in enumerate(reversed(st.session_state.history[-5:]), start=1):
        st.markdown(f"### طلب {idx}")
        st.text(f"التصنيف | {item['category']}")
        st.text(f"الملخص | {item['summary']}")
        st.text(f"الرد | {item['reply'][:140]}...")

