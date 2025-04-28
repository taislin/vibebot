from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.storage.storage_context import StorageContext
from dotenv import load_dotenv
from loguru import logger
import sys
import os
import asyncio

# --- Groq/LlamaIndex Configuration ---
from llama_index.core.settings import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configure logging
logger.remove()
logger.add("bot.log", rotation="1 MB", level="INFO")
logger.add(sink=lambda msg: print(msg, end=""), level="INFO")

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
embed_model = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Configure LLM and embeddings immediately
if not groq_api_key:
    logger.error("GROQ_API_KEY not found. Cannot configure LLM.")
    raise ValueError("GROQ_API_KEY environment variable not set.")
logger.info("Configuring Groq LLM")
Settings.llm = Groq(model=groq_model, api_key=groq_api_key)

logger.info("Configuring HuggingFace Embeddings")
Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model)


# Helper function to run blocking IO/CPU tasks
async def run_blocking(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


async def load_index(directory_path: str = r"data"):
    persist_dir = "./storage"

    if not os.path.isdir(directory_path):
        logger.error(f"Data directory not found: {directory_path}")
        return None

    logger.info(f"Attempting to load documents from: {directory_path}")
    try:
        reader = SimpleDirectoryReader(
            directory_path,
            filename_as_id=True,
            exclude=["*.exe", "*.bin", "*.dll", "*.bat", "*.sh"],
            file_extractor={
                ".cs": "TextFileReader",
                ".py": "TextFileReader",
                ".yml": "TextFileReader",
            },
            recursive=True,
        )
        documents = await run_blocking(reader.load_data)
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
        storage_context = await run_blocking(
            StorageContext.from_defaults, persist_dir=persist_dir
        )
        index = await run_blocking(load_index_from_storage, storage_context)
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
            await run_blocking(os.makedirs, persist_dir, exist_ok=True)
            logger.info("Creating new vector store index from documents...")
            index = await run_blocking(
                VectorStoreIndex.from_documents,
                documents,
                show_progress=True,
                chunk_size=1024,
                chunk_overlap=200,
                transformations_kwargs={"batch_size": 32},  # Enable batching
            )
            logger.info(f"Persisting newly created index to {persist_dir}...")
            await run_blocking(index.storage_context.persist, persist_dir=persist_dir)
            logger.info("New index created and persisted successfully.")

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
                file.endswith(ext)
                for ext in [".exe", ".bin", ".dll", ".bat", ".sh", ".txt", ".md"]
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
            exclude=["*.exe", "*.bin", "*.dll", "*.bat", "*.sh", "*.txt", "*.md"],
            file_extractor={
                ".cs": "TextFileReader",
                ".py": "TextFileReader",
                ".yml": "TextFileReader",
            },
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
    except Exception as e:
        logger.error(f"Error during index update: {e}", exc_info=True)
        return None
