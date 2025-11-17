from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from core.config import AppSettings, get_settings
from core.schemas import ComplaintAnalysis, ComplaintPayload, StreamChunk
from core.services.logging import setup_logging
from core.services.orchestrator import ComplaintOrchestrator

setup_logging()

app = FastAPI(title="AI Complaint Agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_orchestrator(settings: AppSettings = Depends(get_settings)) -> ComplaintOrchestrator:
    return ComplaintOrchestrator(settings=settings)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=ComplaintAnalysis)
async def analyze(payload: ComplaintPayload, orchestrator: ComplaintOrchestrator = Depends(get_orchestrator)):
    return await orchestrator.aanalyze(payload)


@app.post("/analyze/stream")
async def analyze_stream(
    payload: ComplaintPayload,
    orchestrator: ComplaintOrchestrator = Depends(get_orchestrator),
) -> StreamingResponse:
    async def event_stream() -> AsyncGenerator[bytes, None]:
        analysis = await orchestrator.aanalyze(payload)
        sections = [
            ("summary", analysis.summary),
            ("emotions", json.dumps(analysis.emotions, ensure_ascii=False)),
            ("strategy", json.dumps([step.model_dump() for step in analysis.strategy], ensure_ascii=False)),
            ("formal_reply", analysis.formal_reply),
            (
                "meta",
                json.dumps(
                    {
                        "category": analysis.category.value,
                        "risk_level": analysis.risk_level,
                        "confidence": analysis.router.confidence,
                    },
                    ensure_ascii=False,
                ),
            ),
        ]
        for section, payload_text in sections:
            chunk = StreamChunk(section=section, payload=payload_text)
            yield (chunk.model_dump_json() + "\n").encode("utf-8")
            await asyncio.sleep(0)

    return StreamingResponse(event_stream(), media_type="application/json")

