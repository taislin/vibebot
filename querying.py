from manage_embedding import load_index, run_blocking
from loguru import logger
import os
from dotenv import load_dotenv
import asyncio

# --- Groq/LlamaIndex Configuration ---
from llama_index.core.settings import Settings
from llama_index.llms.groq import Groq
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer

# Configure logging
logger.remove()
logger.add("bot.log", rotation="1 MB", level="INFO")
logger.add(sink=lambda msg: print(msg, end=""), level="INFO")

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL", "gemma2-9b-it")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set.")

# Configure the LLM globally
logger.info("Configuring Groq LLM in querying.py")
Settings.llm = Groq(model=groq_model, api_key=groq_api_key)


def preprocess_query(input_text: str) -> str:
    if len(input_text.split()) < 3:
        return f"Explain {input_text} in the context of the game codebase."
    return input_text


async def data_querying(input_text: str, mode: str = "general", user_id: str = None):
    processed_query = preprocess_query(input_text)
    index = await load_index("data")
    if index is None:
        logger.error("Failed to load index in data_querying.")
        return "Error: Could not load the data index."
    context_prompt = {
        "general": "You are a helpful assistant for a game development codebase.",
        "docs": "Provide detailed documentation for the requested code element (e.g., function, class).",
        "search": "Find and return code snippets matching the query.",
        "debug": "Analyze the query as a code error and suggest fixes.",
        "generate": "Generate a code snippet in the style of the codebase.",
    }.get(mode, "general")
    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
    engine = CondensePlusContextChatEngine.from_defaults(
        index.as_retriever(),
        memory=memory,
        llm=Settings.llm,
        context_prompt=context_prompt,
    )
    logger.info(
        f"Querying index with Groq LLM ({Settings.llm.model}) for: {processed_query}"
    )
    try:
        response = await run_blocking(engine.chat, processed_query)
        if mode == "generate":
            response_text = f"```csharp\n{response.response}\n```"
        else:
            response_text = response.response
        logger.info(f"Response: {response_text}")
        return response_text
    except Exception as e:
        logger.error(f"Error during engine.query: {e}", exc_info=True)
        return f"Error during query processing: {e}"
