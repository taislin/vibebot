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
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

# Configure logging
logger.remove()
logger.add("bot.log", rotation="1 MB", level="INFO")
logger.add(sink=lambda msg: print(msg, end=""), level="INFO")

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set.")

# Configure the LLM globally
logger.info("Configuring Groq LLM in querying.py")
Settings.llm = Groq(model=groq_model, api_key=groq_api_key)


def preprocess_query(input_text: str) -> str:
    if len(input_text.split()) < 3:
        return f"Explain {input_text} in the context of the game codebase."
    return input_text


async def data_querying(query: str, mode: str = "general", user_id: str = None):
    logger.info(f"Querying index with Groq LLM ({Settings.llm.model}) for: {query}")
    try:
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
        if not index.docstore.docs:
            logger.warning("Index is empty. Please run /updatedb or cli.py update.")
            return "No code found in the database. Please update the index using /updatedb."

        retriever = index.as_retriever(similarity_top_k=5)
        nodes = retriever.retrieve(query)
        if not nodes:
            logger.info(f"No relevant documents found for query: {query}")
            return f"No relevant code found for: {query}. Try rephrasing or updating the index."

        context = "\n".join([node.text for node in nodes])
        logger.info(f"Retrieved {len(nodes)} documents for query: {query}")

        prompt = f"Based on the following code context, answer the query: {query}\n\nContext:\n{context}\n\nAnswer:"
        response = await Settings.llm.acomplete(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error querying index: {e}", exc_info=True)
        return f"Error querying the database: {e}"
