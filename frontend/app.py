import json
from typing import Dict

import httpx
import streamlit as st

st.set_page_config(page_title="AI Complaint Agent", layout="wide")

st.title("منصة تحليل الشكاوى بالذكاء الاصطناعي")
st.caption("تصنيف | فهم المشاعر | خطة معالجة | رد رسمي")

with st.sidebar:
    st.header("بيانات الشركة")
    company_name = st.text_input("اسم الشركة", value="سريع إكسبرس")
    service_name = st.text_input("الخدمة / المنتج", value="توصيل متاجر")
    notes = st.text_area("سياسات إضافية", value="يجب الرد خلال 24 ساعة كحد أقصى.")
    backend_url = st.text_input("رابط واجهة البرمجة", value="http://localhost:8080")
    st.markdown("---")
    st.write("أدخل نص الشكوى في النموذج الرئيسي.")

complaint_text = st.text_area(
    "نص الشكوى",
    height=200,
    placeholder="اكتب الشكوى كما وصلت من العميل...",
)

if "history" not in st.session_state:
    st.session_state.history = []


def trigger_analysis(payload: Dict) -> Dict:
    stream_endpoint = f"{backend_url}/analyze/stream"
    chunks: Dict[str, str] = {}
    with httpx.Client(timeout=httpx.Timeout(120.0)) as client:
        with client.stream("POST", stream_endpoint, json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                chunks[data["section"]] = data["payload"]
                yield data["section"], data["payload"]
    return chunks


if st.button("حلّل الشكوى", use_container_width=True) and complaint_text:
    payload = {
        "complaint_text": complaint_text,
        "company": {"name": company_name, "service": service_name},
        "notes": notes,
    }
    placeholder_summary = st.empty()
    placeholder_emotions = st.empty()
    placeholder_strategy = st.empty()
    placeholder_reply = st.empty()

    collected = {}
    with st.spinner("جارٍ تحليل الشكوى عبر الوكلاء..."):
        for section, value in trigger_analysis(payload):
            if section == "summary":
                placeholder_summary.subheader("١. ملخص المشكلة")
                placeholder_summary.write(f"ملخص | {value}")
            elif section == "emotions":
                emotions = ", ".join(json.loads(value))
                placeholder_emotions.subheader("٢. المشاعر الأساسية")
                placeholder_emotions.write(f"مشاعر | {emotions}")
            elif section == "strategy":
                steps = json.loads(value)
                lines = [
                    f"- {step['action_title']} | المسؤول: {step['owner_role']} | "
                    f"الجدول: {step['timeline']} | المعيار: {step['success_metric']}"
                    for step in steps
                ]
                placeholder_strategy.subheader("٣. إستراتيجية الحل المقترحة")
                placeholder_strategy.markdown("\n".join(lines))
            elif section == "formal_reply":
                placeholder_reply.subheader("٤. الرد الرسمي")
                placeholder_reply.write(value)
            collected[section] = value

    if collected:
        st.success("تم توليد الخطة والرد بنجاح.")
        meta_raw = collected.get("meta", "{}")
        meta = json.loads(meta_raw or "{}")
        st.info(
            f"التصنيف | {meta.get('category', 'غير محدد')} | "
            f"مستوى المخاطر | {meta.get('risk_level', 'غير محدد')} | "
            f"الثقة | {meta.get('confidence', '0')}"
        )
        st.session_state.history.append(
            {
                "summary": collected.get("summary", ""),
                "category": meta.get("category", ""),
                "reply": collected.get("formal_reply", ""),
            }
        )

if st.session_state.history:
    st.markdown("## السجل الأخير")
    for idx, item in enumerate(reversed(st.session_state.history[-5:]), start=1):
        st.markdown(f"### طلب {idx}")
        st.text(f"التصنيف | {item['category']}")
        st.text(f"الملخص | {item['summary']}")
        st.text(f"الرد | {item['reply'][:140]}...")

