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


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL", "gemma2-9b-it")

# --- Ensure Embeddings are configured if needed ---
if not Settings.embed_model:
    logging.info("Configuring HuggingFace Embeddings in manage_embedding.py")
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# -------------------------------------------------

# Only configure LLM if not already set and if the API key is available
if not isinstance(Settings.llm, Groq) and groq_api_key:
    logging.info("Configuring Groq LLM in manage_embedding.py")
    Settings.llm = Groq(model=groq_model, api_key=groq_api_key)
elif not groq_api_key and not isinstance(Settings.llm, Groq):
    logging.warning(
        "GROQ_API_KEY not found. LLM configuration might be missing or default."
    )
# --------------------------------------


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# Helper function to run blocking IO/CPU tasks
async def run_blocking(func, *args, **kwargs):
    # Runs the synchronous function 'func' in a separate thread
    return await asyncio.to_thread(func, *args, **kwargs)


async def load_index(directory_path: str = r"data"):
    persist_dir = "./storage"

    if not os.path.isdir(directory_path):
        logging.error(f"Data directory not found: {directory_path}")
        return None

    logging.info(f"Attempting to load documents from: {directory_path}")
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
        logging.info(f"Loaded {len(documents)} documents/pages.")
        if not documents:
            logging.warning(
                f"No documents found in {directory_path}. Index creation skipped if storage empty."
            )

    except Exception as e:
        logging.error(
            f"Error loading documents from {directory_path}: {e}", exc_info=True
        )
        return None

    index = None

    try:
        logging.info(f"Attempting to load index from storage: {persist_dir}")
        # --- Wrap blocking storage loading ---
        storage_context = await run_blocking(
            StorageContext.from_defaults, persist_dir=persist_dir
        )
        index = await run_blocking(load_index_from_storage, storage_context)
        # -----------------------------------
        logging.info("Index loaded successfully from storage.")

    except FileNotFoundError:
        logging.info(
            f"Index not found in {persist_dir}. Attempting to create a new one..."
        )
        if not documents:
            logging.error("Cannot create index: No documents were loaded.")
            return None

        if not Settings.llm or not Settings.embed_model:
            logging.error(
                "LLM or Embed Model not configured in Settings. Cannot create index."
            )
            return None

        try:
            logging.info(f"Ensuring storage directory exists: {persist_dir}")
            # os.makedirs is generally fast, but can be wrapped if causing issues
            await run_blocking(os.makedirs, persist_dir, exist_ok=True)

            logging.info(
                "Creating new vector store index from documents (will run in thread)..."
            )
            # --- Wrap blocking index creation ---
            index = await run_blocking(
                VectorStoreIndex.from_documents,
                documents,
                show_progress=True,  # Note: progress bar might not display correctly from thread
            )
            # ------------------------------------

            logging.info(
                f"Persisting newly created index to {persist_dir} (will run in thread)..."
            )
            # --- Wrap blocking persistence ---
            # Accessing index.storage_context should be fine, persist() is the blocking part
            await run_blocking(index.storage_context.persist, persist_dir=persist_dir)
            # -------------------------------

            logging.info(f"New index created and persisted successfully.")

        except Exception as e:
            logging.error(f"Error creating or persisting new index: {e}", exc_info=True)
            return None

    except Exception as e:
        logging.error(f"Error loading index from storage: {e}", exc_info=True)
        return None

    if index is None:
        logging.error("Index object is None after load/create attempt.")

    return index


async def update_index(directory_path: str = r"data"):
    persist_dir = "./storage"

    if not os.path.isdir(directory_path):
        logging.error(f"Data directory not found for update: {directory_path}")
        return None

    logging.info(f"Attempting to load documents for update from: {directory_path}")
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
        logging.info(f"Loaded {len(documents)} documents/pages for potential update.")
        if not documents:
            logging.warning(f"No documents found in {directory_path} for update.")

    except Exception as e:
        logging.error(
            f"Error loading documents from {directory_path} for update: {e}",
            exc_info=True,
        )
        return None

    try:
        if not os.path.isdir(persist_dir):
            logging.error(
                f"Storage directory {persist_dir} not found. Cannot load index for update. Create index first."
            )
            return None

        logging.info(f"Loading existing index from {persist_dir} for update...")
        # --- Wrap blocking storage loading ---
        storage_context = await run_blocking(
            StorageContext.from_defaults, persist_dir=persist_dir
        )
        index = await run_blocking(load_index_from_storage, storage_context)
        # -----------------------------------
        logging.info("Existing index loaded successfully for update.")

        if not Settings.llm or not Settings.embed_model:
            logging.error(
                "LLM or Embed Model not configured in Settings. Cannot refresh index."
            )
            return None

        logging.info("Refreshing index documents (will run in thread)...")
        # --- Wrap blocking index refresh ---
        refreshed_docs = await run_blocking(
            index.refresh_ref_docs,
            documents,
            update_kwargs={"delete_kwargs": {"delete_from_docstore": True}},
        )
        # ---------------------------------

        refreshed_count = sum(1 for refreshed in refreshed_docs if refreshed)
        logging.info(
            f"Refreshed Docs Status (True means updated/inserted): {refreshed_docs}"
        )
        logging.info(f"Number of newly inserted/refreshed docs: {refreshed_count}")

        if any(refreshed_docs):
            logging.info(
                f"Persisting index changes to storage: {persist_dir} (will run in thread)..."
            )
            # --- Wrap blocking persistence ---
            await run_blocking(index.storage_context.persist, persist_dir=persist_dir)
            # -------------------------------
            logging.info("Index refreshed and changes persisted.")
        else:
            logging.info("No changes detected in documents. Index not persisted.")

        return refreshed_docs

    except FileNotFoundError:
        logging.error(
            f"Index files not found in {persist_dir}, although directory might exist. Cannot update. Recreate index if needed."
        )
        return None
    except Exception as e:
        logging.error(f"Error during index update: {e}", exc_info=True)
        return None
