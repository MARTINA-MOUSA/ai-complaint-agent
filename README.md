# AI Complaint Agent

Production-ready multi-agent system that classifies Arabic customer complaints for delivery/e-commerce companies, produces emotional insights, action plans, and empathetic replies, and exposes both FastAPI and Streamlit interfaces.

## Features
- LlamaIndex multi-agent orchestration (router, classifier, resolver, policy guard).
- Arabic-first prompt templates with structured output (summary | emotions | strategy | formal reply).
- FastAPI backend with streaming endpoint for Streamlit and external integrations.
- Streamlit dashboard for operations teams with live response rendering and history.
- Configurable via environment variables; supports OpenAI-compatible models (user adds key).

## Quick Start
1. Create a virtualenv (`python -m venv .venv && .venv\\Scripts\\activate`).
2. Install dependencies: `pip install -e .[dev]`.
3. Copy `env.example` to `.env` and fill `LLM_API_KEY`.
4. Run backend API: `uvicorn backend.main:app --reload`.
5. Run Streamlit UI: `streamlit run frontend/app.py`.

## Tests
```
pytest
```

