# -*- coding: utf-8 -*-
"""Image generation agent with approval flow.

Usage:
    - ADK Web: adk web (from parent directory)
    - Terminal: python -m image_generator.agent
"""

import asyncio
import sys
import uuid

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from google.adk.agents import LlmAgent
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.models import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from google.genai.types import FunctionResponse
from mcp import StdioServerParameters

from config import Config, create_retry_config, load_config, logger


# Constants
NUM_IMAGE_THRESHOLD = 1
MCP_TIMEOUT = 120.0
IMAGE_MCP_SERVER = "@singularity2045/image-generator-mcp-server"
APP_NAME = "image_generator_app"
USER_ID = "user"

# Output directory (relative to this file)
import os
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"


def clear_output_folder() -> None:
    """Clear all files in the output folder."""
    if OUTPUT_DIR.exists():
        for file in OUTPUT_DIR.iterdir():
            if file.is_file():
                file.unlink()
        logger.info("Output folder cleared: %s", OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def display_image(file_path: str) -> str:
    """Returns the file path where image is saved."""
    import os
    if os.path.exists(file_path):
        size_kb = os.path.getsize(file_path) / 1024
        return f"Image saved at: {file_path} ({size_kb:.1f} KB). Open with: open {file_path}"
    return f"Image file not found at: {file_path}"


def create_image_approval_tool() -> FunctionTool:
    """Create the image generation approval tool."""

    def request_image_generation(num_image: int, tool_context: ToolContext) -> dict:
        """
        Requests approval before generating multiple images.

        Args:
            num_image: Number of images to be generated.
            tool_context: ADK tool context for managing the confirmation flow.

        Returns:
            dict: Contains 'status' and 'message'.
        """
        if num_image <= NUM_IMAGE_THRESHOLD:
            return {"status": "SUCCESS", "message": "Image has been generated!"}

        if not tool_context.tool_confirmation:
            tool_context.request_confirmation(
                hint="Need approval to generate multiple images. Please respond Y/n",
                payload={"num_image": num_image},
            )
            return {"status": "PENDING", "message": "Waiting for approval!"}

        if tool_context.tool_confirmation.confirmed:
            return {"status": "SUCCESS", "message": "Image has been generated!"}

        return {"status": "REJECTED", "message": "Unable to generate image"}

    return FunctionTool(func=request_image_generation)


def create_mcp_toolset(config: Config) -> McpToolset:
    """Create MCP toolset for image generation."""
    return McpToolset(
        connection_params=StdioConnectionParams(
            timeout=MCP_TIMEOUT,
            server_params=StdioServerParameters(
                command="npx",
                args=[
                    "-y",
                    IMAGE_MCP_SERVER,
                    "--",
                    f"--openai-api-key={config.openai_api_key}",
                    "--port=0",
                ],
            ),
        ),
    )


def create_agent(
    config: Config, mcp_toolset: McpToolset, retry_config: types.HttpRetryOptions
) -> LlmAgent:
    """Create the image generator agent."""
    output_path = str(OUTPUT_DIR)
    return LlmAgent(
        model=Gemini(model=config.model_name, retry_options=retry_config),
        name="image_generator_agent",
        description="An agent that generates images with approval flow for multiple images.",
        instruction=f"""You are an expert at generating images.
When a user asks to generate an image, you must:
1. First call `request_image_generation` with the number of images requested.
2. If approved (status is SUCCESS), call the `generate_image` MCP tool with file path in this folder: {output_path}
   Use filenames like image_1.png, image_2.png, etc.
3. After the image is generated, call `display_image` with the same file path to show the result.""",
        tools=[
            create_image_approval_tool(),
            mcp_toolset,
            FunctionTool(func=display_image),
        ],
    )


def create_app(agent: LlmAgent) -> App:
    """Create the ADK App with resumability enabled."""
    return App(
        name=APP_NAME,
        root_agent=agent,
        resumability_config=ResumabilityConfig(is_resumable=True),
    )


# --- Workflow functions for terminal usage ---


def check_for_approval(events: list) -> dict | None:
    """Check events for an approval request."""
    for event in events:
        if event.content and event.content.parts:
            for part in event.content.parts:
                if (
                    part.function_call
                    and part.function_call.name == "adk_request_confirmation"
                ):
                    return {
                        "approval_id": part.function_call.id,
                        "invocation_id": event.invocation_id,
                    }
    return None


def construct_adk_response(approval_id: str, confirmed: bool = True) -> types.Content:
    """Construct ADK confirmation response."""
    adk_confirmation_response = FunctionResponse(
        id=approval_id,
        name="adk_request_confirmation",
        response={"confirmed": confirmed},
    )
    return types.Content(
        role="user",
        parts=[types.Part(function_response=adk_confirmation_response)],
    )


async def run_workflow(
    runner: Runner,
    session_service: InMemorySessionService,
    user_query: str,
    is_approved: bool = True,
) -> None:
    """Run the image generation workflow with approval handling."""
    session_id = str(uuid.uuid4())

    await session_service.create_session(
        app_name=APP_NAME,
        session_id=session_id,
        user_id=USER_ID,
    )

    events = []
    query = types.Content(parts=[types.Part(text=user_query)])

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=query,
    ):
        events.append(event)

    approval_info = check_for_approval(events)

    if approval_info:
        logger.info("Approval requested, responding with: %s", is_approved)
        async for event in runner.run_async(
            user_id=USER_ID,
            session_id=session_id,
            new_message=construct_adk_response(
                approval_info["approval_id"], is_approved
            ),
            invocation_id=approval_info["invocation_id"],
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    logger.info("Part: %s", part)
    else:
        for event in events:
            if event.content and event.content.parts:
                for part in event.content.parts:
                    logger.info("Agent response: %s", part)


_config = None
_retry_config = None
_mcp_toolset = None
root_agent = None

# --- ADK Web Discovery ---
# Export root_agent for adk web to find
try:
    clear_output_folder()  # Clear on every run
    _config = load_config(require_openai=True)
    _retry_config = create_retry_config(_config)
    _mcp_toolset = create_mcp_toolset(_config)
    root_agent = create_agent(_config, _mcp_toolset, _retry_config)
except Exception as e:
    logger.warning("Failed to initialize root_agent: %s", e)
    root_agent = None
    
async def main() -> None:
    """Main entry point for terminal usage."""
    try:
        clear_output_folder()  # Clear on every run
        config = _config or load_config(require_openai=True)
        retry_config = _retry_config or create_retry_config(config)

        mcp_toolset = _mcp_toolset or create_mcp_toolset(config)
        logger.info("MCP toolset created.")

        agent = root_agent or create_agent(config, mcp_toolset, retry_config)
        app = create_app(agent)
        logger.info("Agent and app initialized.")

        session_service = InMemorySessionService()
        runner = Runner(app=app, session_service=session_service)

        await run_workflow(
            runner=runner,
            session_service=session_service,
            user_query="Generate 4 different images of a fish",
            is_approved=True,
        )

    except ValueError as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
