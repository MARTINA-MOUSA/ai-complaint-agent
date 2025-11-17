"""Multi-agent orchestration for complaint analysis."""

from __future__ import annotations

from typing import Any, Optional

from core.agents.classification import ClassificationAgent
from core.agents.emotion import EmotionAgent
from core.agents.reply import ReplyAgent
from core.agents.strategy import StrategyAgent
from core.config import AppSettings
from core.schemas import ComplaintPayload
from core.services.logging import get_logger

logger = get_logger(__name__)


class ComplaintOrchestrator:
    """Multi-agent orchestrator that combines all agents' outputs into one response."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        llm: Optional[Any] = None,
        verbose_agents: bool = False,
    ) -> None:
        self.settings = settings
        if llm is None:
            llm = settings.build_llm()
        self.llm = llm

        # Create all agents using the same LLM
        self.classification_agent = ClassificationAgent(llm=self.llm, verbose=verbose_agents)
        self.emotion_agent = EmotionAgent(llm=self.llm, verbose=verbose_agents)
        self.strategy_agent = StrategyAgent(llm=self.llm, verbose=verbose_agents)
        self.reply_agent = ReplyAgent(llm=self.llm, verbose=verbose_agents)

    async def aanalyze(self, payload: ComplaintPayload) -> str:
        """Analyze complaint using multiple agents and combine results."""
        logger.info("orchestrator.start", complaint=len(payload.complaint_text))

        # Step 1: Classification
        logger.info("agent.classification.start")
        classification = await self.classification_agent.aclassify(payload)
        logger.info("agent.classification.done", length=len(classification))

        # Step 2: Emotion Analysis
        logger.info("agent.emotion.start")
        emotions = await self.emotion_agent.aanalyze_emotions(payload, classification)
        logger.info("agent.emotion.done", length=len(emotions))

        # Step 3: Strategy Creation
        logger.info("agent.strategy.start")
        strategy = await self.strategy_agent.acreate_strategy(payload, classification, emotions)
        logger.info("agent.strategy.done", length=len(strategy))

        # Step 4: Formal Reply
        logger.info("agent.reply.start")
        formal_reply = await self.reply_agent.acreate_reply(payload, classification, emotions, strategy)
        logger.info("agent.reply.done", length=len(formal_reply))

        # Combine all results into one comprehensive response
        final_response = self._combine_results(classification, emotions, strategy, formal_reply)

        logger.info("orchestrator.end", total_length=len(final_response))
        return final_response

    def _combine_results(
        self,
        classification: str,
        emotions: str,
        strategy: str,
        formal_reply: str,
    ) -> str:
        """Combine all agent outputs into one formatted response."""
        return f"""# تحليل شامل للشكوى

## ١. التصنيف
{classification}

---

## ٢. فهم المشاعر
{emotions}

---

## ٣. خطة المعالجة
{strategy}

---

## ٤. الرد الرسمي
{formal_reply}
"""
