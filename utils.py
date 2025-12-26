# -*- coding: utf-8 -*-
"""Shared utility functions for ADK agents.

This module contains reusable helper functions that are common across
different agent implementations. These utilities are designed to be
stateless and have no side effects beyond logging.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.adk.events import Event

logger = logging.getLogger(__name__)


def print_agent_response(events: list[Event]) -> None:
    """Log text responses from agent events.

    Iterates through a list of ADK events and logs any text content
    found in the event parts. This is useful for debugging and
    displaying agent outputs in CLI applications.

    Args:
        events: List of ADK Event objects to process.

    Example:
        >>> events = []
        >>> async for event in runner.run_async(...):
        ...     events.append(event)
        >>> print_agent_response(events)
    """
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                logger.info("Agent response: %s", part)
