"""Logging utilities."""

from __future__ import annotations

import logging
from typing import Any

import structlog


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structlog for both backend and CLI usages."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
    )
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.EventRenamer("event"),
            structlog.processors.dict_tracebacks,
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)

