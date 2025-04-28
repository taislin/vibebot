from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core import load_index_from_storage
from llama_index.core.storage.storage_context import StorageContext
from dotenv import load_dotenv
from loguru import logger
import sys
import os
import asyncio
import unicodedata

# Groq/LlamaIndex Configuration
from llama_index.core.settings import Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Logging
logger.remove()
logger.add("bot.log", rotation="1 MB", level="INFO")
logger.add(sink=lambda msg: print(msg, end=""), level="INFO")

# Environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
embed_model = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Configure LLM and embeddings
if not groq_api_key:
    logger.error("GROQ_API_KEY not found.")
    raise ValueError("GROQ_API_KEY not set.")
logger.info("Configuring Groq LLM")
Settings.llm = Groq(model=groq_model, api_key=groq_api_key)
logger.info("Configuring HuggingFace Embeddings")
Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model)


# Clean text
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(c for c in text if c.isprintable() or c.isspace())


# Custom file reader
class CustomTextFileReader:
    def __call__(self, file_path: str):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            cleaned_text = clean_text(text)
            return {"text": cleaned_text, "file_path": file_path}
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
            return None


# Run blocking tasks
async def run_blocking(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


async def load_index(directory_path: str = r"data"):
    persist_dir = "./storage"
    if not os.path.isdir(directory_path):
        logger.error(f"Data directory not found: {directory_path}")
        return None

    logger.info(f"Loading documents from: {directory_path}")
    try:
        reader = SimpleDirectoryReader(
            directory_path,
            filename_as_id=True,
            exclude=["*.exe", "*.bin", "*.dll", "*.bat", "*.sh", "*.txt", "*.md"],
            file_extractor={
                ".cs": CustomTextFileReader(),
                ".py": CustomTextFileReader(),
                ".yml": CustomTextFileReader(),
            },
            recursive=True,
        )
        documents = await run_blocking(reader.load_data)
        documents = [doc for doc in documents if doc is not None]
        logger.info(f"Loaded {len(documents)} documents/pages.")
        if not documents:
            logger.warning(f"No documents found in {directory_path}.")
    except Exception as e:
        logger.error(f"Error loading documents: {e}", exc_info=True)
        return None

    index = None
    try:
        logger.info(f"Loading index from: {persist_dir}")
        storage_context = await run_blocking(
            StorageContext.from_defaults, persist_dir=persist_dir
        )
        index = await run_blocking(load_index_from_storage, storage_context)
        logger.info("Index loaded.")
    except FileNotFoundError:
        logger.info(f"Index not found in {persist_dir}. Creating new one...")
        if not documents:
            logger.error("No documents loaded.")
            return None
        if not Settings.llm or not Settings.embed_model:
            logger.error("LLM or Embed Model not configured.")
            return None
        try:
            logger.info(f"Ensuring storage directory: {persist_dir}")
            await run_blocking(os.makedirs, persist_dir, exist_ok=True)
            logger.info("Creating new index...")
            index = await run_blocking(
                VectorStoreIndex.from_documents,
                documents,
                show_progress=True,
                chunk_size=2048,
                chunk_overlap=400,
                transformations_kwargs={"batch_size": 32},
            )
            logger.info(f"Persisting index to {persist_dir}...")
            await run_blocking(index.storage_context.persist, persist_dir=persist_dir)
            logger.info("Index created and persisted.")
        except Exception as e:
            logger.error(f"Error creating index: {e}", exc_info=True)
            return None
    except Exception as e:
        logger.error(f"Error loading index: {e}", exc_info=True)
        return None

    if index is None:
        logger.error("Index is None.")
    return index


async def update_index(directory_path: str = r"data"):
    persist_dir = "./storage"
    if not os.path.isdir(directory_path):
        logger.error(f"Data directory not found: {directory_path}")
        return None

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
                ".cs": CustomTextFileReader(),
                ".py": CustomTextFileReader(),
                ".yml": CustomTextFileReader(),
            },
        )
        documents = await run_blocking(reader.load_data)
        documents = [doc for doc in documents if doc is not None]
        if documents:
            refreshed_docs = await run_blocking(
                index.refresh_ref_docs,
                documents,
                update_kwargs={"delete_kwargs": {"delete_from_docstore": True}},
            )
            refreshed_count = sum(1 for refreshed in refreshed_docs if refreshed)
            logger.info(f"Refreshed {refreshed_count} documents.")
            await run_blocking(index.storage_context.persist, persist_dir=persist_dir)
            for file in file_metadata:
                with open(f"{persist_dir}/{f}.meta", "w") as f:
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
