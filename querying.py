from manage_embedding import load_index, run_blocking
import logging
import sys
import os
from dotenv import load_dotenv
import asyncio

# --- Groq/LlamaIndex Configuration ---
from llama_index.core.settings import Settings  # <-- Import Settings
from llama_index.llms.groq import Groq  # <-- Import Groq LLM
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer

# Load environment variables (ensure GROQ_API_KEY is set)
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL", "gemma2-9b-it")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set.")

# Configure the LLM globally for LlamaIndex
# Choose your desired Groq model, e.g., "gemma-7b-it"
Settings.llm = Groq(model=groq_model, api_key=groq_api_key)
# Optional: Configure embedding model if you don't want the default (OpenAI)
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# --------------------------------------


logger.basicConfig(stream=sys.stdout, level=logger.INFO)
logger.getLogger().addHandler(logger.StreamHandler(stream=sys.stdout))

from loguru import logger

logger.add("bot.log", rotation="1 MB", level="INFO")


def preprocess_query(input_text: str) -> str:
    # Example: If query is vague, append codebase context
    if len(input_text.split()) < 3:
        return f"Explain {input_text} in the context of the game codebase."
    return input_text


async def data_querying(input_text: str, mode: str = "general", user_id: str = None):
    processed_query = preprocess_query(input_text)
    index = await load_index("data")
    if index is None:
        return "Error: Could not load the data index."
    context_prompt = {
        "general": "You are a helpful assistant for a game development codebase.",
        "docs": "Provide detailed documentation for the requested code element (e.g., function, class).",
        "search": "Find and return code snippets matching the query.",
        "debug": "Analyze the query as a code error and suggest fixes.",
        "generate": "Generate a code snippet in the style of the codebase.",
    }.get(mode, "general")
    # Use user-specific memory for context
    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
    engine = CondensePlusContextChatEngine.from_defaults(
        index.as_retriever(),
        memory=memory,
        llm=Settings.llm,
        context_prompt=context_prompt,
    )
    response = await run_blocking(engine.chat, processed_query)
    if mode == "generate":
        response_text = f"```csharp\n{response.response}\n```"  # Format as code block
    else:
        response_text = response.response
    return response_text
