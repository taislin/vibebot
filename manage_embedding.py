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
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document
import gc
from pathlib import Path
import json

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
Settings.llm = Groq(model=groq_model, api_key=groq_api_key)
Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model)


# Clean text
def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(c for c in text if c.isprintable() or c.isspace())


# Custom file reader
class CustomTextFileReader(BaseReader):
    def load_data(self, file_path: str, extra_info: dict = None):
        # Convert PosixPath to string if necessary
        file_path = str(file_path) if isinstance(file_path, Path) else file_path
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            cleaned_text = clean_text(text)
            metadata = {"file_path": file_path}
            if extra_info:
                metadata.update(extra_info)
            return [Document(text=cleaned_text, metadata=metadata)]
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
            return []


# Run blocking tasks
async def run_blocking(func, *args, **kwargs):
    result = await asyncio.to_thread(func, *args, **kwargs)
    gc.collect()
    return result


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
            exclude=[
                "*.exe",
                "*.bin",
                "*.dll",
                "*.bat",
                "*.sh",
                ".git/*",
            ],
            file_extractor={
                ".cs": CustomTextFileReader(),
                ".py": CustomTextFileReader(),
                ".yml": CustomTextFileReader(),
            },
            recursive=True,
        )
        # Ensure file paths are strings
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
                chunk_size=1024,  # Reduced
                chunk_overlap=200,  # Reduced
                transformations_kwargs={"batch_size": 32},  # Reduced
            )
            logger.info(f"Persisting index to {persist_dir}...")
            await run_blocking(index.storage_context.persist, persist_dir=persist_dir)
            logger.info("Index created and persisted.")
            # Save .meta files for all documents
            logger.info("Saving .meta files for initial index...")
            for doc in documents:
                file_path = doc.metadata.get("file_path")
                if file_path and os.path.exists(file_path):
                    meta_path = os.path.join(persist_dir, f"{file_path}.meta")
                    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
                    try:
                        with open(meta_path, "w") as f:
                            f.write(f"{os.path.getmtime(file_path):.6f}")
                    except Exception as e:
                        logger.error(f"Failed to save .meta for {file_path}: {e}")
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

    logger.info("Scanning files for metadata...")
    file_metadata = {}
    for root, _, files in os.walk(directory_path):
        if ".git" in root or persist_dir in os.path.abspath(root):
            continue  # Skip .git and storage directories
        for file in files:
            if not any(
                file.endswith(ext)
                for ext in [".exe", ".bin", ".dll", ".bat", ".sh", ".meta"]
            ) and file not in [".gitignore", ".gitattributes"]:
                file_path = os.path.join(root, file)
                # Check mtime stability
                mtime1 = os.path.getmtime(file_path)
                await asyncio.sleep(0.001)  # Brief delay to detect rapid changes
                mtime2 = os.path.getmtime(file_path)
                if abs(mtime1 - mtime2) > 0.01:
                    logger.warning(
                        f"Skipping {file_path}: mtime unstable ({mtime1} vs {mtime2})"
                    )
                    continue
                file_metadata[file_path] = mtime1
                logger.debug(f"Scanned file: {file_path}, mtime: {mtime1}")
    logger.info(f"Found {len(file_metadata)} files.")

    try:
        logger.info("Loading storage context...")
        vector_store_path = os.path.join(persist_dir, "default__vector_store.json")
        if os.path.exists(vector_store_path):
            try:
                with open(vector_store_path, "r") as f:
                    json.load(f)  # Raises JSONDecodeError if invalid
                logger.debug(
                    f"Validated default__vector_store.json: {vector_store_path}"
                )
            except json.JSONDecodeError as e:
                logger.error(
                    f"Invalid JSON in {vector_store_path}: {e}, removing file",
                    exc_info=True,
                )
                os.remove(vector_store_path)  # Remove corrupted file
        storage_context = await run_blocking(
            StorageContext.from_defaults, persist_dir=persist_dir
        )
        logger.info("Loading existing index...")
        try:
            index = await run_blocking(load_index_from_storage, storage_context)
            logger.info("Existing index loaded.")
        except Exception as e:
            logger.error(f"Failed to load index from storage: {e}", exc_info=True)
            raise
        logger.info("Checking for changed files...")
        changed_files = []
        for file_path in file_metadata:
            meta_path = os.path.join(persist_dir, f"{file_path}.meta")
            file_mtime = file_metadata[file_path]
            if not os.path.exists(meta_path):
                logger.debug(f"No .meta file for {file_path}, marking as changed")
                changed_files.append(file_path)
            else:
                try:
                    with open(meta_path, "r") as f:
                        meta_mtime_str = f.read().strip()
                    if not meta_mtime_str:  # Check for empty file
                        logger.warning(
                            f"Empty .meta file for {file_path}, marking as changed"
                        )
                        changed_files.append(file_path)
                        continue
                    try:
                        meta_mtime = float(meta_mtime_str)
                        if abs(file_mtime - meta_mtime) > 0.01:  # 10ms tolerance
                            logger.debug(
                                f"File {file_path} changed (file_mtime={file_mtime}, meta_mtime={meta_mtime})"
                            )
                            changed_files.append(file_path)
                        else:
                            logger.debug(
                                f"File {file_path} unchanged (file_mtime={file_mtime}, meta_mtime={meta_mtime})"
                            )
                    except ValueError as e:
                        logger.error(
                            f"Invalid .meta file content for {file_path}: {meta_mtime_str}, error: {e}, marking as changed"
                        )
                        changed_files.append(file_path)
                except IOError as e:
                    logger.error(
                        f"Error reading .meta file for {file_path}: {e}, marking as changed"
                    )
                    changed_files.append(file_path)
        logger.info(f"Found {len(changed_files)} changed files.")
        if changed_files:
            logger.debug(f"Changed files: {changed_files[:10]}")

        if changed_files:
            batch_size = 25
            for i in range(0, len(changed_files), batch_size):
                batch_files = changed_files[i : i + batch_size]
                logger.info(
                    f"Processing batch {i // batch_size + 1} with {len(batch_files)} files..."
                )

                logger.info("Loading changed documents...")
                reader = SimpleDirectoryReader(
                    input_files=[str(f) for f in batch_files],
                    filename_as_id=True,
                    exclude=[
                        "*.exe",
                        "*.bin",
                        "*.dll",
                        "*.bat",
                        "*.sh",
                        "*.meta",
                        ".git/*",
                    ],
                    file_extractor={
                        ".cs": CustomTextFileReader(),
                        ".py": CustomTextFileReader(),
                        ".yml": CustomTextFileReader(),
                    },
                )
                documents = await run_blocking(reader.load_data)
                documents = [doc for doc in documents if doc is not None]
                logger.info(f"Loaded {len(documents)} changed documents.")

                if documents:
                    logger.info("Refreshing documents...")
                    try:
                        refreshed_docs = await asyncio.wait_for(
                            run_blocking(
                                index.refresh_ref_docs,
                                documents,
                                delete_from_docstore=True,  # Pass directly instead of update_kwargs
                            ),
                            timeout=300,
                        )
                        refreshed_count = sum(
                            1 for refreshed in refreshed_docs if refreshed
                        )
                        logger.info(f"Refreshed {refreshed_count} documents.")
                    except asyncio.TimeoutError:
                        logger.error(
                            f"Document refresh timed out after 300 seconds for batch {i // batch_size + 1}"
                        )
                        return None
                    except Exception as e:
                        logger.error(
                            f"Error refreshing documents in batch {i // batch_size + 1}: {e}",
                            exc_info=True,
                        )
                        return None

                    logger.info("Persisting index...")
                    try:
                        await asyncio.wait_for(
                            run_blocking(
                                index.storage_context.persist, persist_dir=persist_dir
                            ),
                            timeout=300,
                        )
                        # Verify vector_store.json
                        vector_store_path = os.path.join(
                            persist_dir, "default__vector_store.json"
                        )
                        if os.path.exists(vector_store_path):
                            with open(vector_store_path, "r") as f:
                                json.load(f)  # Raises JSONDecodeError if invalid
                            logger.info(
                                "Index persisted and default__vector_store.json validated."
                            )
                        else:
                            logger.warning(
                                "default__vector_store.json not found after persisting."
                            )
                    except asyncio.TimeoutError:
                        logger.error(
                            f"Index persistence timed out after 300 seconds for batch {i // batch_size + 1}"
                        )
                        return None
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Invalid JSON in default__vector_store.json: {e}",
                            exc_info=True,
                        )
                        return None
                    logger.info("Updating .meta files for batch...")
                    for file in batch_files:
                        meta_path = os.path.join(persist_dir, f"{file}.meta")
                        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
                        try:
                            with open(meta_path, "w") as f:
                                f.write(f"{file_metadata[file]:.6f}")
                        except Exception as e:
                            logger.error(f"Failed to update .meta for {file}: {e}")
                    logger.info("Metadata updated for batch.")

                    # Clean up memory
                    del documents, refreshed_docs
                    gc.collect()
                    logger.info("Memory cleaned up.")
                else:
                    logger.info("No documents to refresh in batch.")

            return changed_files
        else:
            logger.info("No changes detected.")
            return []
    except FileNotFoundError:
        logger.error("Index not found. Creating new index...")
        index = await load_index(directory_path)
        return [] if index else None
    except Exception as e:
        logger.error(f"Error during index update: {e}", exc_info=True)
        return None
