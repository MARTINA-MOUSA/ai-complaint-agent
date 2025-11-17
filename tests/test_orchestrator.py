import asyncio

import pytest

from core.schemas import (
    ClassificationResult,
    ComplaintAnalysis,
    ComplaintCategory,
    ComplaintPayload,
    CompanyDetails,
    ResolutionBundle,
    RouterDecision,
    StrategyStep,
)
from core.services.orchestrator import ComplaintOrchestrator
from core.config import AppSettings


class FakeRouter:
    async def aroute(self, payload: ComplaintPayload) -> RouterDecision:
        return RouterDecision(
            category=ComplaintCategory.DELIVERY,
            confidence=0.9,
            rationale="نص يصف تأخير التوصيل",
        )


class FakeClassifier:
    async def aclassify(self, payload: ComplaintPayload, routed_category: ComplaintCategory) -> ClassificationResult:
        return ClassificationResult(
            category=routed_category,
            summary="تأخر السائق ولم يتواصل مع العميل.",
            emotions=["غضب", "قلق"],
            risk_level="high",
        )


class FakeResolution:
    async def aresolve(self, payload: ComplaintPayload, classification: ClassificationResult) -> ResolutionBundle:
        steps = [
            StrategyStep(
                action_title="الاتصال بالعميل خلال 15 دقيقة",
                owner_role="فريق خدمة العملاء",
                timeline="فوري",
                success_metric="تأكيد الرضا في المكالمة",
            ),
            StrategyStep(
                action_title="مراجعة أداء السائق",
                owner_role="فريق العمليات",
                timeline="24 ساعة",
                success_metric="اكتمال تقرير المتابعة",
            ),
            StrategyStep(
                action_title="تعويض العميل بقسيمة",
                owner_role="فريق الولاء",
                timeline="24 ساعة",
                success_metric="إرسال القسيمة",
            ),
        ]
        return ResolutionBundle(strategy=steps, formal_reply="نعتذر عن التأخير...")


class FakePolicy:
    async def areview(self, payload, classification, resolution) -> dict:
        return {
            "summary": classification.summary,
            "emotions": classification.emotions,
            "strategy": [step.model_dump() for step in resolution.strategy],
            "formal_reply": resolution.formal_reply,
        }


def build_payload() -> ComplaintPayload:
    return ComplaintPayload(
        complaint_text="طلبت شحنة غذاء وتأخر السائق ساعتين ولم يرد على الاتصالات.",
        company=CompanyDetails(name="سريع", service="توصيل المنازل"),
        notes="يجب تعويض العميل إذا تجاوز التأخير 60 دقيقة.",
    )


@pytest.mark.asyncio
async def test_orchestrator_with_fake_agents():
    orchestrator = ComplaintOrchestrator(
        settings=AppSettings(),
        router_agent=FakeRouter(),
        classifier_agent=FakeClassifier(),
        resolution_agent=FakeResolution(),
        policy_agent=FakePolicy(),
    )

    analysis = await orchestrator.aanalyze(build_payload())
    assert isinstance(analysis, ComplaintAnalysis)
    assert analysis.category == ComplaintCategory.DELIVERY
    assert len(analysis.strategy) == 3
    assert "نعتذر" in analysis.formal_reply

