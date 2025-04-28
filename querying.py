from llama_index.core.settings import Settings
from llama_index.llms.groq import Groq
from loguru import logger
import os
from dotenv import load_dotenv

# Logging
logger.remove()
logger.add("bot.log", rotation="1 MB", level="INFO")
logger.add(sink=lambda msg: print(msg, end=""), level="INFO")

# Environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Configure LLM
if not groq_api_key:
    logger.error("GROQ_API_KEY not found.")
    raise ValueError("GROQ_API_KEY not set.")
logger.info(f"Configuring Groq LLM in querying.py")
Settings.llm = Groq(model=groq_model, api_key=groq_api_key)


async def data_querying(index, query_text: str, mode: str = "general"):
    logger.info(
        f"Querying index with Groq LLM ({groq_model}) for: {query_text} (mode: {mode})"
    )
    try:
        # Configure query engine based on mode
        if mode == "general":
            query_engine = index.as_query_engine(
                llm=Settings.llm,
                similarity_top_k=5,
                response_mode="compact",
            )
        elif mode == "docs":
            query_engine = index.as_query_engine(
                llm=Settings.llm,
                similarity_top_k=10,
                response_mode="tree_summarize",
                output_cls=dict,  # Structured output
            )
        elif mode == "search":
            query_engine = index.as_query_engine(
                llm=Settings.llm,
                similarity_top_k=8,
                response_mode="no_text",
                return_source=True,
            )
        elif mode == "debug":
            query_engine = index.as_query_engine(
                llm=Settings.llm,
                similarity_top_k=5,
                response_mode="no_text",
                verbose=True,
                return_source=True,
            )
        elif mode == "generate":
            query_engine = index.as_query_engine(
                llm=Settings.llm,
                similarity_top_k=2,
                response_mode="compact",
                use_generation=True,
            )
        else:
            logger.warning(f"Unknown mode: {mode}, defaulting to general")
            query_engine = index.as_query_engine(
                llm=Settings.llm,
                similarity_top_k=5,
                response_mode="compact",
            )

        response = await query_engine.aquery(query_text)
        logger.info(
            f"Retrieved {len(response.source_nodes)} documents for query: {query_text}"
        )

        # Format response based on mode
        if mode == "docs":
            result = {
                "summary": str(response),
                "sources": [
                    node.metadata.get("file_path") for node in response.source_nodes
                ],
            }
            return result
        elif mode == "search" or mode == "debug":
            result = [
                {
                    "file_path": node.metadata.get("file_path"),
                    "text": (
                        node.text[:500] + "..." if len(node.text) > 500 else node.text
                    ),
                    "score": node.score,
                }
                for node in response.source_nodes
            ]
            return result
        else:
            return str(response)
    except Exception as e:
        logger.error(f"Error querying index: {e}", exc_info=True)
        return f"Error: {e}"
