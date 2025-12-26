
import asyncio
import sys

from typing import Any, Dict, List
import uuid
import sqlite3
from google.adk.agents import LlmAgent
from google.adk.apps import app
from google.api_core import retry
import pandas as pd

from google.adk import Agent, Runner
from google.adk.apps import App
from google.adk.apps.app import EventsCompactionConfig
from google.adk.models import Gemini
from google.adk.sessions import InMemorySessionService, Session
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from config import Config, DEFAULT_USER_ID as USER_ID, create_retry_config, load_config, logger
from utils import print_agent_response

def create_agent(config: Config, retry_config: types.HttpRetryOptions) -> Agent:
    return Agent(
        model=Gemini(model=config.model_name, retry_options=retry_config),
        name="session_demo_agent",
        instruction="Answer user queries"
    )

def create_app(agent: Agent) -> App:
    return App(
        name="session_demo_app",
        root_agent=agent,
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=3,
            overlap_size=1
        )
    )

async def create_session(
    app: App, session_service: DatabaseSessionService | InMemorySessionService, session_id: str
) -> Session:
    return await session_service.create_session(
        app_name=app.name,
        session_id=session_id,
        user_id=USER_ID
    )

async def run_session(
    app: App,
    session_service: DatabaseSessionService | InMemorySessionService,
    session_id: str,
    user_query: List[str] | str,
) -> None:
    runner = Runner(app=app, session_service=session_service)

    queries = [user_query] if isinstance(user_query, str) else user_query

    for query_text in queries:
        logger.info(f"User: {query_text}")
        message = types.Content(parts=[types.Part(text=query_text)])
        events = []
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=message
        ):
            events.append(event)
        print_agent_response(events)
        
def read_db()->None:
    # Connect to the database
    conn = sqlite3.connect("sessions.db")

    # Read the table directly into a readable dataframe
    df = pd.read_sql("SELECT * FROM events", conn)

    print(df)
    conn.close()

async def read_events(db_session_service: DatabaseSessionService, session_id: str)-> None:
    session = await db_session_service.get_session(
        app_name=app.__name__, session_id=session_id,
        user_id=USER_ID
    )
    
    if session:
        for event in session.events:
            if event.actions and event.actions.compaction:
                print(f"Author: {event.author}")

def set_user_details(tool_context:ToolContext,user_name:str,country:str)->Dict[str,Any]:
    """
    Function to save username & country to session state

    Args:
        tool_context (ToolContext)
        user_name (str): username
        country (str): country

    Returns:
        Dict[str,Any]: status message
    """
    
    tool_context.state["user:name"] = user_name
    tool_context.state["user:country"] = country
    
    return {"status":"SUCCESS"}

def get_user_details(tool_context:ToolContext)->Dict[str,Any]:
    """
    Function to get username & country from session state

    Args:
        tool_context (ToolContext)

    Returns:
        Dict[str,Any]: status message
    """
    username = tool_context.state.get("user:name", "Username not found")
    country = tool_context.state.get("user:country", "Country not found")
    
    return {"status":"SUCCESS","username":username,"country":country}

def create_session_agent(config:Config,retry_config:types.HttpRetryOptions) -> Agent:
    return LlmAgent(
        model=Gemini(model=config.model_name,retry_options=retry_config),
        name="session_demo_agent",
        instruction="""
        Answer user queries and save user details to session state
        1. To set user details use `set_user_details` tool
        2. To get user details use `get_user_details` tool
        """,
                tools=[set_user_details, get_user_details]
    )
    
async def main() -> None:
    """Main entry point for the image generation agent."""
    try:
        # Load config
        config = load_config()
        retry_config = create_retry_config(config)
    
        # Create agent
        agent = create_agent(config, retry_config)
        session_agent = create_session_agent(config, retry_config)
        logger.info("Agent initialized.")
        
        # Create app
        # app = create_app(agent)
        app = create_app(session_agent)
        logger.info("App initialized.")
        
        db_url = "sqlite+aiosqlite:///sessions.db"
        db_session_service = DatabaseSessionService(db_url=db_url)
        session_service = InMemorySessionService()

        session_id = "test-db-session-04"

        try:
            # await create_session(app, db_session_service, session_id)
            await create_session(app, session_service, session_id)
            logger.info("Session created.")
        except Exception:
            logger.info("Session already exists or could not be created, proceeding...")
        
        # Run session
        # await run_session(app,db_session_service,session_id,[
        #     "Hi! What is the long term effect of daily 2-3 hours of use of a typical TWS in 30-40% volume almost daily?"
        # ])        
        
        # # Run session
        # await run_session(app,db_session_service,session_id,[
        #     "Hi! What is the long term effect of daily 2-3 hours of use of a typical TWS in 30-40% volume almost daily?"
        # ])        
        
        # # Run session
        # await run_session(app,db_session_service,session_id,[
        #     "Hi! What is the long term effect of daily 2-3 hours of use of a typical TWS in 30-40% volume almost daily?"
        # ])        
        
        # # Run session
        # await run_session(app,db_session_service,session_id,[
        #     "Hi! What is the long term effect of daily 2-3 hours of use of a typical TWS in 30-40% volume almost daily?"
        # ])        
        
        # # Run session
        # await run_session(app,db_session_service,session_id,[
        #     "Hi! What is the long term effect of daily 2-3 hours of use of a typical TWS in 30-40% volume almost daily?"
        # ])        
        
        # # Run session
        # await run_session(app,db_session_service,session_id,[
        #     "Hi! What is the long term effect of daily 2-3 hours of use of a typical TWS in 30-40% volume almost daily?"
        # ])        
        
        await run_session(app,session_service,session_id,[
                # "hello, what is my name?",
                # "My name is Kounde and from France",
                "Am I from Spain or England?"
                ])

        # await read_events(db_session_service,session_id)
        
        session = await session_service.get_session(app_name=app.name,session_id=session_id,user_id=USER_ID)
        
        if session:
            print("session state > ", session.state)
        
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        sys.exit(1)

# Export root_agent for ADK web discovery
try:
    _config = load_config()
    _retry_config = create_retry_config(_config)
    root_agent = create_session_agent(_config, _retry_config)
except Exception as e:
    logger.warning("Failed to initialize root_agent: %s", e)
    root_agent = None
    
if __name__ == "__main__":
    asyncio.run(main())