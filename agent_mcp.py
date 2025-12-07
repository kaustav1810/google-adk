# -*- coding: utf-8 -*-
"""MCP Agent for image retrieval using Google ADK."""

import asyncio
import sys

from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from mcp import StdioServerParameters

from config import Config, create_retry_config, load_config, logger


MCP_SERVER_PACKAGE = "@modelcontextprotocol/server-everything"
MCP_TIMEOUT = 30.0


def create_mcp_toolset() -> McpToolset:
    """Create and configure MCP toolset."""
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command="npx",
                args=["-y", MCP_SERVER_PACKAGE],
            ),
            timeout=MCP_TIMEOUT,
        ),
        tool_filter=["getTinyImage"],
    )


def create_agent(config: Config, mcp_toolset: McpToolset) -> LlmAgent:
    """Create and configure the LLM agent."""
    retry_config = create_retry_config(config)
    return LlmAgent(
        name="mcp_agent",
        model=Gemini(model=config.model_name, retry_options=retry_config),
        instruction=(
            "Use the getTinyImage tool to retrieve and display images. "
            "This tool returns a small test image."
        ),
        tools=[mcp_toolset],
    )


async def run_agent(runner: InMemoryRunner, mcp_toolset: McpToolset) -> None:
    """Run the agent and handle cleanup."""
    try:
        result = await runner.run_debug("Get a tiny image", verbose=True)
        logger.info("Agent result: %s", result)
    finally:
        await mcp_toolset.close()
        logger.info("MCP connections closed.")


def is_notebook_environment() -> bool:
    """Check if running in a Jupyter notebook."""
    try:
        get_ipython()  # type: ignore[name-defined]
        return True
    except NameError:
        return False


def setup_notebook_async() -> None:
    """Configure async support for notebook environments."""
    logger.warning("MCP STDIO servers may have issues in Jupyter notebooks.")
    logger.warning("Recommendation: Run from terminal with 'python agent_mcp.py'")

    try:
        import nest_asyncio

        nest_asyncio.apply()
    except ImportError:
        logger.info("Installing nest_asyncio...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "nest_asyncio"])
        import nest_asyncio

        nest_asyncio.apply()


def main() -> None:
    """Main entry point."""
    try:
        config = load_config()

        mcp_toolset = create_mcp_toolset()
        logger.info("MCP toolset created.")

        agent = create_agent(config, mcp_toolset)
        runner = InMemoryRunner(agent=agent)
        logger.info("Agent and runner initialized.")

        if is_notebook_environment():
            setup_notebook_async()

        asyncio.run(run_agent(runner, mcp_toolset))

    except ValueError as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
