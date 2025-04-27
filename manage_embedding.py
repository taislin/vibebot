from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.storage.storage_context import StorageContext
from dotenv import load_dotenv
import logging
import sys
import os
import asyncio

# --- Groq/LlamaIndex Configuration ---
from llama_index.core.settings import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ---------------------------------------------------------
from loguru import logger

logger.add("bot.log", rotation="1 MB", level="INFO")

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL", "gemma2-9b-it")
embed_model = os.getenv("EMBED_MODEL", "jinaai/jina-embeddings-v2-base-code")

# --- Ensure Embeddings are configured if needed ---
if not Settings.embed_model:
    logger.info("Configuring HuggingFace Embeddings in manage_embedding.py")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=embed_model,
    )
# -------------------------------------------------

# Only configure LLM if not already set and if the API key is available
if not isinstance(Settings.llm, Groq) and groq_api_key:
    logger.info("Configuring Groq LLM in manage_embedding.py")
    Settings.llm = Groq(model=groq_model, api_key=groq_api_key)
elif not groq_api_key and not isinstance(Settings.llm, Groq):
    logger.warning(
        "GROQ_API_KEY not found. LLM configuration might be missing or default."
    )
# --------------------------------------


logger.basicConfig(stream=sys.stdout, level=logger.INFO)
logger.getLogger().addHandler(logger.StreamHandler(stream=sys.stdout))


# Helper function to run blocking IO/CPU tasks
async def run_blocking(func, *args, **kwargs):
    # Runs the synchronous function 'func' in a separate thread
    return await asyncio.to_thread(func, *args, **kwargs)


async def load_index(directory_path: str = r"data"):
    persist_dir = "./storage"

    if not os.path.isdir(directory_path):
        logger.error(f"Data directory not found: {directory_path}")
        return None

    logger.info(f"Attempting to load documents from: {directory_path}")
    try:
        # --- Wrap blocking file reading ---
        reader = SimpleDirectoryReader(
            directory_path,
            filename_as_id=True,
            exclude=["*.exe", "*.bin", "*.dll", ".bat", ".sh"],
            recursive=True,
        )
        documents = await run_blocking(reader.load_data)
        # ----------------------------------
        logger.info(f"Loaded {len(documents)} documents/pages.")
        if not documents:
            logger.warning(
                f"No documents found in {directory_path}. Index creation skipped if storage empty."
            )

    except Exception as e:
        logger.error(
            f"Error loading documents from {directory_path}: {e}", exc_info=True
        )
        return None

    index = None

    try:
        logger.info(f"Attempting to load index from storage: {persist_dir}")
        # --- Wrap blocking storage loading ---
        storage_context = await run_blocking(
            StorageContext.from_defaults, persist_dir=persist_dir
        )
        index = await run_blocking(load_index_from_storage, storage_context)
        # -----------------------------------
        logger.info("Index loaded successfully from storage.")

    except FileNotFoundError:
        logger.info(
            f"Index not found in {persist_dir}. Attempting to create a new one..."
        )
        if not documents:
            logger.error("Cannot create index: No documents were loaded.")
            return None

        if not Settings.llm or not Settings.embed_model:
            logger.error(
                "LLM or Embed Model not configured in Settings. Cannot create index."
            )
            return None

        try:
            logger.info(f"Ensuring storage directory exists: {persist_dir}")
            # os.makedirs is generally fast, but can be wrapped if causing issues
            await run_blocking(os.makedirs, persist_dir, exist_ok=True)

            logger.info(
                "Creating new vector store index from documents (will run in thread)..."
            )
            # --- Wrap blocking index creation ---
            index = await run_blocking(
                VectorStoreIndex.from_documents,
                documents,
                show_progress=True,  # Note: progress bar might not display correctly from thread
            )
            # ------------------------------------

            logger.info(
                f"Persisting newly created index to {persist_dir} (will run in thread)..."
            )
            # --- Wrap blocking persistence ---
            # Accessing index.storage_context should be fine, persist() is the blocking part
            await run_blocking(index.storage_context.persist, persist_dir=persist_dir)
            # -------------------------------

            logger.info(f"New index created and persisted successfully.")

        except Exception as e:
            logger.error(f"Error creating or persisting new index: {e}", exc_info=True)
            return None

    except Exception as e:
        logger.error(f"Error loading index from storage: {e}", exc_info=True)
        return None

    if index is None:
        logger.error("Index object is None after load/create attempt.")

    return index


async def update_index(directory_path: str = r"data"):
    persist_dir = "./storage"
    if not os.path.isdir(directory_path):
        logger.error(f"Data directory not found: {directory_path}")
        return None

    # Get file metadata
    file_metadata = {}
    for root, _, files in os.walk(directory_path):
        for file in files:
            if not any(
                file.endswith(ext) for ext in [".exe", ".bin", ".dll", ".bat", ".sh"]
            ):
                file_path = os.path.join(root, file)
                file_metadata[file_path] = os.path.getmtime(file_path)

    try:
        storage_context = await run_blocking(
            StorageContext.from_defaults, persist_dir=persist_dir
        )
        index = await run_blocking(load_index_from_storage, storage_context)
        logger.info("Existing index loaded.")

        # Load only changed files
        reader = SimpleDirectoryReader(
            input_files=[
                f
                for f in file_metadata
                if not os.path.exists(f"{persist_dir}/{f}.meta")
                or file_metadata[f] > os.path.getmtime(f"{persist_dir}/{f}.meta")
            ],
            filename_as_id=True,
        )
        documents = await run_blocking(reader.load_data)
        if documents:
            refreshed_docs = await run_blocking(
                index.refresh_ref_docs,
                documents,
                update_kwargs={"delete_kwargs": {"delete_from_docstore": True}},
            )
            refreshed_count = sum(1 for refreshed in refreshed_docs if refreshed)
            logger.info(f"Refreshed {refreshed_count} documents.")
            await run_blocking(index.storage_context.persist, persist_dir=persist_dir)
            # Save metadata
            for file in file_metadata:
                with open(f"{persist_dir}/{file}.meta", "w") as f:
                    f.write(str(file_metadata[file]))
            return refreshed_docs
        else:
            logger.info("No changes detected.")
            return []
    except FileNotFoundError:
        logger.error("Index not found. Run load_index first.")
        return None
