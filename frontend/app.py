import asyncio
import os
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
    # Also check for GOOGLE_API_KEY (Gemini)
    if "GOOGLE_API_KEY" in st.secrets and "LLM_API_KEY" not in os.environ:
        os.environ["LLM_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


@st.cache_resource
def get_orchestrator() -> ComplaintOrchestrator:
    _ensure_llm_key()
    settings = get_settings()
    return ComplaintOrchestrator(settings=settings, verbose_agents=True)


def analyze_locally(payload: ComplaintPayload) -> str:
    """Analyze complaint and return plain Arabic text."""
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

    with st.spinner("جارٍ تحليل الشكوى..."):
        analysis_text = analyze_locally(payload_model)

    st.success("تم توليد التحليل بنجاح.")
    
    # Display the analysis text directly
    st.markdown("---")
    st.markdown(analysis_text)
    
    # Save to history
    st.session_state.history.append({
        "complaint": complaint_text[:100] + "..." if len(complaint_text) > 100 else complaint_text,
        "analysis": analysis_text[:500] + "..." if len(analysis_text) > 500 else analysis_text,
    })

if st.session_state.history:
    st.markdown("---")
    st.markdown("## السجل الأخير")
    for idx, item in enumerate(reversed(st.session_state.history[-5:]), start=1):
        with st.expander(f"طلب {idx}: {item['complaint']}"):
            st.markdown(item['analysis'])
