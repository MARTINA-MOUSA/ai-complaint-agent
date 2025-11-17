"""Orchestration layer combining all agents."""

from __future__ import annotations

from typing import Any, Optional

from core.agents.classifier import ClassifierAgent
from core.agents.policy import PolicyAgent
from core.agents.resolution import ResolutionAgent
from core.agents.router import RouterAgent
from core.config import AppSettings
from core.schemas import (
    ClassificationResult,
    ComplaintAnalysis,
    ComplaintPayload,
    ResolutionBundle,
    RouterDecision,
    StrategyStep,
)
from core.services.logging import get_logger

logger = get_logger(__name__)


class ComplaintOrchestrator:
    def __init__(
        self,
        *,
        settings: AppSettings,
        llm: Optional[Any] = None,
        verbose_agents: bool = False,
        router_agent: Optional[RouterAgent] = None,
        classifier_agent: Optional[ClassifierAgent] = None,
        resolution_agent: Optional[ResolutionAgent] = None,
        policy_agent: Optional[PolicyAgent] = None,
    ) -> None:
        self.settings = settings
        needs_llm = any(
            agent is None for agent in (router_agent, classifier_agent, resolution_agent, policy_agent)
        )
        if llm is None and needs_llm:
            llm = settings.build_llm()
        self.llm = llm

        if router_agent is None:
            if not self.llm:
                raise ValueError("Router agent requires an LLM instance.")
            router_agent = RouterAgent(llm=self.llm, verbose=verbose_agents)
        if classifier_agent is None:
            if not self.llm:
                raise ValueError("Classifier agent requires an LLM instance.")
            classifier_agent = ClassifierAgent(llm=self.llm, verbose=verbose_agents)
        if resolution_agent is None:
            if not self.llm:
                raise ValueError("Resolution agent requires an LLM instance.")
            resolution_agent = ResolutionAgent(llm=self.llm, verbose=verbose_agents)
        if policy_agent is None:
            if not self.llm:
                raise ValueError("Policy agent requires an LLM instance.")
            policy_agent = PolicyAgent(llm=self.llm, verbose=verbose_agents)

        self.router = router_agent
        self.classifier = classifier_agent
        self.resolution = resolution_agent
        self.policy = policy_agent

    async def aanalyze(self, payload: ComplaintPayload) -> ComplaintAnalysis:
        logger.info("orchestrator.start", complaint=len(payload.complaint_text))

        router_decision = await self.router.aroute(payload)
        logger.info("router.decision", category=router_decision.category, confidence=router_decision.confidence)

        classification = await self.classifier.aclassify(payload, router_decision.category)
        logger.info("classifier.done", category=classification.category, risk=classification.risk_level)

        resolution = await self.resolution.aresolve(payload, classification)
        logger.info("resolution.steps", steps=len(resolution.strategy))

        reviewed = await self.policy.areview(payload, classification, resolution)
        analysis = self._assemble_output(
            router_decision=router_decision,
            classification=classification,
            resolution=resolution,
            reviewed=reviewed,
        )
        logger.info("orchestrator.end", category=analysis.category, steps=len(analysis.strategy))
        return analysis

    def _assemble_output(
        self,
        *,
        router_decision: RouterDecision,
        classification: ClassificationResult,
        resolution: ResolutionBundle,
        reviewed: dict,
    ) -> ComplaintAnalysis:
        summary = reviewed.get("summary", classification.summary)
        emotions = reviewed.get("emotions", classification.emotions)
        strategy_payload = reviewed.get("strategy") or [step.model_dump() for step in resolution.strategy]
        strategy = [StrategyStep(**step) for step in strategy_payload]
        formal_reply = reviewed.get("formal_reply", resolution.formal_reply)

        return ComplaintAnalysis(
            summary=summary,
            emotions=emotions,
            strategy=strategy,
            formal_reply=formal_reply,
            category=classification.category,
            router=router_decision,
            risk_level=classification.risk_level,
        )

