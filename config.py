# -*- coding: utf-8 -*-
"""Shared configuration and common imports for agents."""

import logging
import os
import sys
from dataclasses import dataclass, field

from dotenv import load_dotenv
from google.genai import types

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Application constants
DEFAULT_USER_ID = "kaus123"


@dataclass
class Config:
    """Application configuration."""

    google_api_key: str
    openai_api_key: str | None = None
    model_name: str = "gemini-2.5-flash-lite"
    retry_attempts: int = 5
    retry_exp_base: int = 7
    retry_initial_delay: int = 1
    retry_http_codes: list[int] = field(default_factory=lambda: [429, 500, 503, 504])


def load_config(require_openai: bool = False) -> Config:
    """
    Load configuration from environment variables.

    Args:
        require_openai: If True, raises error when OPENAI_API_KEY is missing.

    Returns:
        Config object with loaded values.

    Raises:
        ValueError: If required API keys are missing.
    """
    load_dotenv()

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY must be set in the .env file.")

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if require_openai and not openai_api_key:
        raise ValueError("OPENAI_API_KEY must be set in the .env file.")

    # Set in environment for SDKs that read directly from env
    os.environ["GOOGLE_API_KEY"] = google_api_key
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    logger.info("Configuration loaded successfully.")

    return Config(
        google_api_key=google_api_key,
        openai_api_key=openai_api_key,
    )


def create_retry_config(config: Config) -> types.HttpRetryOptions:
    """Create HTTP retry configuration."""
    return types.HttpRetryOptions(
        attempts=config.retry_attempts,
        exp_base=config.retry_exp_base,
        initial_delay=config.retry_initial_delay,
        http_status_codes=config.retry_http_codes,
    )
