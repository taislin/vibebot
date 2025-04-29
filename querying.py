from llama_index.core import VectorStoreIndex
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
from loguru import logger
from llama_index.llms.groq import Groq
from dotenv import load_dotenv
import os

# Logging
logger.remove()
logger.add("bot.log", rotation="1 MB", level="INFO")
logger.add(sink=lambda msg: print(msg, end=""), level="INFO")

# Initialize memory with FileChatMessageHistory
history_file = "data/conversation_history.jsonl"
os.makedirs(os.path.dirname(history_file), exist_ok=True)
chat_history = FileChatMessageHistory(file_path=history_file)
memory = ConversationBufferMemory(
    chat_memory=chat_history,
    return_messages=True,
)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Configure Groq LLM
if not groq_api_key:
    logger.error("GROQ_API_KEY not found.")
    raise ValueError("GROQ_API_KEY not set.")
llm = Groq(model=groq_model, api_key=groq_api_key)


async def data_querying(index: VectorStoreIndex, text: str, mode: str = "general"):
    logger.info(f"Processing query: {text} (mode: {mode})")
    try:
        # Save input to memory
        memory.save_context({"input": text}, {"output": ""})

        # Load conversation history
        history = memory.load_memory_variables({})["history"]
        context = "\n".join(
            [
                f"{'Human' if i % 2 == 0 else 'Assistant'}: {msg.content}"
                for i, msg in enumerate(history)
            ]
        )

        # Combine context with current query
        full_query = f"{context}\nHuman: {text}" if context else text

        # Create query engine
        query_engine = index.as_query_engine(llm=llm, similarity_top_k=3)

        # Query the index with context
        if mode == "general":
            response = await query_engine.aquery(full_query)
            response_text = str(response)
            # Save response to memory
            memory.save_context({"input": text}, {"output": response_text})
            return {
                "summary": response_text,
                "sources": [
                    node.metadata.get("file_path") for node in response.source_nodes
                ],
            }
        elif mode == "docs":
            query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)
            response = await query_engine.aquery(full_query)
            return [
                {
                    "file_path": node.metadata.get("file_path"),
                    "score": node.score,
                    "text": node.text,
                }
                for node in response.source_nodes
            ]
        elif mode == "search":
            query_engine = index.as_query_engine(llm=llm, similarity_top_k=10)
            response = await query_engine.aquery(full_query)
            return [
                {
                    "file_path": node.metadata.get("file_path"),
                    "score": node.score,
                    "text": node.text,
                }
                for node in response.source_nodes
            ]
        elif mode == "debug":
            query_engine = index.as_query_engine(llm=llm, similarity_top_k=3)
            response = await query_engine.aquery(full_query)
            return {
                "query": full_query,
                "response": str(response),
                "sources": [
                    node.metadata.get("file_path") for node in response.source_nodes
                ],
            }
        elif mode == "generate":
            response = await llm.complete(full_query)
            return response.text
        else:
            logger.error(f"Unknown mode: {mode}")
            return f"Error: Unknown mode {mode}"
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return f"Error: {e}"
