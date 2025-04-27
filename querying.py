from manage_embedding import load_index, run_blocking
import logging
import sys
import os
from dotenv import load_dotenv
import asyncio

# --- Groq/LlamaIndex Configuration ---
from llama_index.core.settings import Settings  # <-- Import Settings
from llama_index.llms.groq import Groq  # <-- Import Groq LLM


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


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


async def data_querying(input_text: str):
    # load_index is now async and handles its own blocking calls
    index = await load_index("data")
    if index is None:
        logging.error("Failed to load index in data_querying.")
        return "Error: Could not load the data index. Please check logs."

    # Creating the engine is usually fast, no need to wrap typically
    engine = index.as_query_engine()

    logging.info(
        f"Querying index with Groq LLM ({Settings.llm.model}) (will run in thread)..."
    )
    try:
        # --- Wrap the blocking query call ---
        response = await run_blocking(engine.query, input_text)
        # ----------------------------------
        response_text = response.response
        logging.info(f"Groq Response: {response_text}")
        return response_text
    except Exception as e:
        logging.error(f"Error during engine.query: {e}", exc_info=True)
        return f"Error during query processing: {e}"
