"""Structured JSON logging factory shared by every service."""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: int = logging.INFO) -> None:
    """Call once at process start-up (each service's entrypoint)."""
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(**initial_context: object) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(**initial_context)
