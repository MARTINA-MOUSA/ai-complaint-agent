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

# Page configuration
st.set_page_config(
    page_title="منصة تحليل الشكاوى بالذكاء الاصطناعي",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    /* Main title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #1565a0;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Success message */
    .stSuccess {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    
    /* Analysis result container */
    .analysis-container {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 2rem;
        margin-top: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Header in sidebar */
    h3 {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-title">📋 منصة تحليل الشكاوى بالذكاء الاصطناعي</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">تحليل ذكي شامل | تصنيف تلقائي | خطة معالجة | رد رسمي</p>', unsafe_allow_html=True)

def _ensure_llm_key() -> None:
    """Ensure LLM API key is available."""
    if os.environ.get("LLM_API_KEY"):
        return
    if "LLM_API_KEY" in st.secrets:
        os.environ["LLM_API_KEY"] = st.secrets["LLM_API_KEY"]
    # Also check for GOOGLE_API_KEY (Gemini)
    if "GOOGLE_API_KEY" in st.secrets and "LLM_API_KEY" not in os.environ:
        os.environ["LLM_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


@st.cache_resource
def get_orchestrator() -> ComplaintOrchestrator:
    """Get cached orchestrator instance."""
    _ensure_llm_key()
    settings = get_settings()
    return ComplaintOrchestrator(settings=settings, verbose_agents=True)


def analyze_locally(payload: ComplaintPayload) -> str:
    """Analyze complaint and return plain Arabic text."""
    orchestrator = get_orchestrator()
    return asyncio.run(orchestrator.aanalyze(payload))


# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ إعدادات الشركة")
    company_name = st.text_input(
        "🏢 اسم الشركة",
        value="سريع إكسبرس",
        help="أدخل اسم الشركة أو العلامة التجارية"
    )
    service_name = st.text_input(
        "📦 الخدمة / المنتج",
        value="توصيل متاجر",
        help="نوع الخدمة أو المنتج الذي تقدمه الشركة"
    )
    notes = st.text_area(
        "📝 سياسات إضافية",
        value="يجب الرد خلال 24 ساعة كحد أقصى.",
        help="أي سياسات أو قيود إضافية يجب مراعاتها في التحليل",
        height=100
    )
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        💡 أدخل نص الشكوى في النموذج الرئيسي
    </div>
    """, unsafe_allow_html=True)

# Main content
st.markdown("### 📝 إدخال الشكوى")

complaint_text = st.text_area(
    "اكتب نص الشكوى كما وصلت من العميل:",
    height=200,
    placeholder="مثال: العميل يشكو من تأخر وصول الطلب لمدة 5 أيام رغم التأكيد على التسليم خلال 24 ساعة...",
    help="الصق أو اكتب نص الشكوى الكامل هنا"
)

col1, col2 = st.columns([1, 4])

with col1:
    analyze_button = st.button(
        "🔍 حلّل الشكوى",
        use_container_width=True,
        type="primary"
    )

with col2:
    if analyze_button and complaint_text:
        st.info("⏳ جارٍ التحليل... قد يستغرق الأمر بضع ثوانٍ")

if analyze_button and complaint_text:
    payload_model = ComplaintPayload(
        complaint_text=complaint_text,
        company=CompanyDetails(name=company_name, service=service_name or None),
        notes=notes or None,
    )

    try:
        with st.spinner("🔄 جارٍ تحليل الشكوى باستخدام الذكاء الاصطناعي..."):
            analysis_text = analyze_locally(payload_model)

        st.success("✅ تم توليد التحليل بنجاح!")
        
        # Display the analysis in a styled container
        st.markdown("---")
        st.markdown("### 📊 نتائج التحليل")
        st.markdown(
            f'<div class="analysis-container">{analysis_text}</div>',
            unsafe_allow_html=True
        )
        
        # Add download button for the analysis
        st.download_button(
            label="💾 تحميل التحليل",
            data=analysis_text,
            file_name=f"تحليل_شكوى_{company_name.replace(' ', '_')}.txt",
            mime="text/plain"
        )
        
    except Exception as e:
        st.error(f"❌ حدث خطأ أثناء التحليل: {str(e)}")
        st.info("تأكدي من إضافة مفتاح API في Streamlit Secrets")

elif analyze_button and not complaint_text:
    st.warning("⚠️ يرجى إدخال نص الشكوى أولاً")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #999; font-size: 0.85rem; padding: 1rem;'>
        منصة تحليل الشكاوى بالذكاء الاصطناعي | Powered by Gemini AI
    </div>
    """,
    unsafe_allow_html=True
)
