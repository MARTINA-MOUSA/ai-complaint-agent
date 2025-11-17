"""Shared Pydantic models for requests and responses."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class ComplaintCategory(str, Enum):
    DELIVERY = "delivery_issue"
    PAYMENT = "payment_issue"
    TECHNICAL = "technical_issue"
    INQUIRY = "general_inquiry"
    RETURN = "return_exchange"
    OTHER = "other"


class CompanyDetails(BaseModel):
    name: str = Field(..., min_length=2)
    service: Optional[str] = Field(
        default=None, description="Product, plan, or shipment related to the complaint."
    )

    def as_label(self) -> str:
        return f"{self.name} - {self.service}" if self.service else self.name


class ComplaintPayload(BaseModel):
    complaint_text: str = Field(..., min_length=10)
    company: CompanyDetails
    notes: Optional[str] = None


class RouterDecision(BaseModel):
    category: ComplaintCategory
    confidence: float = Field(..., ge=0, le=1)
    rationale: str


class ClassificationResult(BaseModel):
    category: ComplaintCategory
    summary: str
    emotions: List[str] = Field(default_factory=list)
    risk_level: str = Field(default="medium")

    @field_validator("risk_level")
    @classmethod
    def _validate_risk(cls, value: str) -> str:
        allowed = {"low", "medium", "high"}
        if value not in allowed:
            return "medium"
        return value


class StrategyStep(BaseModel):
    action_title: str
    owner_role: str
    timeline: str
    success_metric: str


class ResolutionBundle(BaseModel):
    strategy: List[StrategyStep]
    formal_reply: str


class ComplaintAnalysis(BaseModel):
    summary: str
    emotions: List[str]
    strategy: List[StrategyStep]
    formal_reply: str
    category: ComplaintCategory
    router: RouterDecision
    risk_level: str = "medium"


class StreamChunk(BaseModel):
    section: str
    payload: str

