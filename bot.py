import os
import glob
import json
from git import Repo
from tree_sitter import Language, Parser
import tree_sitter_c_sharp as csharp
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from multiprocessing import Pool
from interactions import (
    Client,
    Intents,
    slash_command,
    SlashContext,
    SlashCommandOption,
    OptionType,
    listen,
    Embed,
)
from loguru import logger
import importlib
from groq import Groq

# --- Setup Logging ---
logger.remove()
logger.add("vibebot.log", rotation="1 MB", level="INFO")
logger.add(sink=lambda msg: print(msg, end=""), level="INFO")

# --- Load Environment Variables ---
load_dotenv()
device = "cuda" if os.getenv("USE_CUDA", "false").lower() == "true" else "cpu"
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
embed_model = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
repo_url = os.getenv("REPO_URL", "https://github.com/Civ13/Civ14.git")
discord_bot_token = os.getenv("DISCORD_BOT_TOKEN")
guild_id = os.getenv("GUILD_ID")

if not discord_bot_token:
    raise ValueError("DISCORD_BOT_TOKEN not found.")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found.")

logger.info(f"Using device: {device}")

# --- Initialize Discord Bot ---
intents = Intents.DEFAULT | Intents.GUILDS
bot = Client(intents=intents, token=discord_bot_token)

# --- Initialize C# Parser ---
CSHARP_LANGUAGE = Language(csharp.language())

# --- Chat History ---
chat_histories = {}  # Store per-user history


def get_session_history(session_id: str):
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()
    return chat_histories[session_id]


# --- Helper: Split Long Text ---
def split_text(text: str, max_length: int = 1024) -> list:
    if not isinstance(text, str):
        text = str(text)
    return [text[i : i + max_length] for i in range(0, len(text), max_length)]


# --- Code Parsing and Indexing ---
def chunk_file(file_path):
    try:
        parser = Parser(language=CSHARP_LANGUAGE)
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        tree = parser.parse(code.encode())
        chunks = [
            code[node.start_byte : node.end_byte]
            for node in tree.root_node.children
            if node.type
            in ["class_declaration", "method_declaration", "property_declaration"]
        ]
        if not chunks:
            step = 800
            overlap = 200
            chunks = [code[i : i + step] for i in range(0, len(code), step - overlap)]
        return [(file_path, chunk) for chunk in chunks]
    except Exception as e:
        logger.error(f"Error parsing {file_path}: {e}")
        return []


def update_index(repo, files, index, model):
    try:
        repo.git.pull()
        new_files = [
            f
            for pattern in ["**/*.cs", "**/*.csproj", "**/*.md", "**/*.yml", "**/*.py"]
            for f in glob.glob(f"data/{pattern}", recursive=True)
            if "/bin/" not in f
            and "/obj/" not in f
            and "/RobustToolbox/" not in f
            and f not in files
        ]
        if new_files:
            with Pool(processes=4) as pool:
                results = pool.map(chunk_file, new_files)
            code_chunks = [chunk for result in results for _, chunk in result]
            metadata = [
                {"file": file, "chunk": chunk}
                for result in results
                for file, chunk in result
            ]
            embeddings = model.embed_documents(code_chunks)
            batch_vectors = [
                (
                    str(i),
                    embeddings[i],
                    {"chunk": code_chunks[i], "file": metadata[i]["file"]},
                )
                for i in range(len(embeddings))
            ]
            index.upsert(vectors=batch_vectors)
            logger.info(f"Updated index with {len(new_files)} new files")
            return new_files
        return []
    except Exception as e:
        logger.error(f"Error updating index: {e}", exc_info=True)
        return None


# --- Bot Event: On Ready ---
@listen()
async def on_ready():
    logger.info(f"Logged in as {bot.user}")
    print("Bot ready")


# --- Slash Command: Query ---
@slash_command(
    name="query",
    description="Query the codebase",
    scopes=[int(guild_id)] if guild_id else None,
    options=[
        SlashCommandOption(
            name="input_text",
            description="Your query about the codebase",
            type=OptionType.STRING,
            required=True,
        ),
        SlashCommandOption(
            name="mode",
            description="Query mode: general, docs, search, debug",
            type=OptionType.STRING,
            required=False,
            choices=[
                {"name": "General", "value": "general"},
                {"name": "Documentation", "value": "docs"},
                {"name": "Code Search", "value": "search"},
                {"name": "Debugging", "value": "debug"},
            ],
        ),
    ],
)
async def query_cmd(ctx: SlashContext, input_text: str, mode: str = "general"):
    await ctx.defer()
    try:
        logger.info(f"Processing query: {input_text} (mode: {mode})")
        # Load vector store
        model = HuggingFaceEmbeddings(
            model_name=embed_model, model_kwargs={"device": device}
        )
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(pinecone_index_name)
        vector_store = PineconeVectorStore(
            index=index, embedding=model, text_key="chunk"
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 10})

        # Retrieve documents
        docs = retriever.invoke(input_text)
        context = "\n".join([doc.page_content for doc in docs])

        # Query Groq LLM with history
        client = Groq(api_key=groq_api_key)
        session_id = str(ctx.author.id)
        chain = RunnableWithMessageHistory(
            runnable=client.chat.completions,
            get_session_history=get_session_history,
            input_messages_key="messages",
            history_messages_key="history",
        )
        response = chain.invoke(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert assistant for analyzing a software codebase. Provide concise, accurate answers based on the provided context.",
                    },
                    {
                        "role": "user",
                        "content": f"Context: {context}\n\nQuestion: {input_text}",
                    },
                ],
                "model": groq_model,
            },
            config={"configurable": {"session_id": session_id}},
        )
        response_text = response.choices[0].message.content

        # Create embed
        embed = Embed(
            title="Query Response",
            description=f"**Input**: {input_text[:1000]}\n**Mode**: {mode}",
            color=0x00FF00,
        )
        response_chunks = split_text(response_text)
        for i, chunk in enumerate(response_chunks, 1):
            embed.add_field(
                name=f"Response (Part {i})" if len(response_chunks) > 1 else "Response",
                value=chunk,
                inline=False,
            )

        # Add conversation history
        history = get_session_history(session_id).messages
        if history:
            history_text = "\n".join(
                f"{'Human' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in history
            )
            history_chunks = split_text(history_text)
            for i, chunk in enumerate(history_chunks, 1):
                embed.add_field(
                    name=(
                        f"History (Part {i})" if len(history_chunks) > 1 else "History"
                    ),
                    value=chunk,
                    inline=False,
                )

        await ctx.send(embeds=embed)
    except Exception as e:
        logger.error(f"Error in query command: {e}", exc_info=True)
        await ctx.send(content=f"An error occurred: {e}", ephemeral=True)


# --- Slash Command: Update Database ---
@slash_command(
    name="updatedb",
    description="Update the codebase index",
    scopes=[int(guild_id)] if guild_id else None,
)
async def updatedb_cmd(ctx: SlashContext):
    await ctx.defer()
    try:
        local_path = "data"
        repo = Repo(local_path)
        files = [
            f
            for pattern in ["**/*.cs", "**/*.csproj", "**/*.md", "**/*.yml", "**/*.py"]
            for f in glob.glob(f"data/{pattern}", recursive=True)
            if "/bin/" not in f and "/obj/" not in f and "/RobustToolbox/" not in f
        ]
        model = HuggingFaceEmbeddings(
            model_name=embed_model, model_kwargs={"device": device}
        )
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(pinecone_index_name)
        update_results = update_index(repo, files, index, model)
        response = (
            f"Updated/Refreshed {len(update_results)} documents."
            if update_results is not None
            else "Error during update."
        )
        await ctx.send(content=response)
    except Exception as e:
        logger.error(f"Error in updatedb command: {e}", exc_info=True)
        await ctx.send(content=f"An error occurred: {e}", ephemeral=True)


# --- Slash Command: Pull Repository ---
@slash_command(
    name="pullrepo",
    description="Pull latest data from GitHub",
    scopes=[int(guild_id)] if guild_id else None,
)
async def pullrepo_cmd(ctx: SlashContext):
    await ctx.defer()
    try:
        import subprocess

        local_path = "data"
        subprocess.run(["git", "-C", local_path, "pull", "origin", "main"], check=True)
        repo = Repo(local_path)
        files = [
            f
            for pattern in ["**/*.cs", "**/*.csproj", "**/*.md", "**/*.yml", "**/*.py"]
            for f in glob.glob(f"data/{pattern}", recursive=True)
            if "/bin/" not in f and "/obj/" not in f and "/RobustToolbox/" not in f
        ]
        model = HuggingFaceEmbeddings(
            model_name=embed_model, model_kwargs={"device": device}
        )
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(pinecone_index_name)
        update_results = update_index(repo, files, index, model)
        num_updated = len(update_results) if update_results else 0
        await ctx.send(
            content=f"Repository updated. Refreshed {num_updated} documents."
        )
    except Exception as e:
        logger.error(f"Error in pullrepo command: {e}", exc_info=True)
        await ctx.send(content=f"An error occurred: {e}", ephemeral=True)


# --- Slash Command: Clear Memory ---
@slash_command(
    name="clear_memory",
    description="Clear conversation memory",
    scopes=[int(guild_id)] if guild_id else None,
)
async def clear_memory_cmd(ctx: SlashContext):
    await ctx.defer()
    try:
        session_id = str(ctx.author.id)
        if session_id in chat_histories:
            chat_histories[session_id].clear()
            logger.info(f"Cleared memory for user {session_id}")
        await ctx.send("Conversation memory cleared.", ephemeral=True)
    except Exception as e:
        logger.error(f"Error in clear_memory command: {e}", exc_info=True)
        await ctx.send(content=f"An error occurred: {e}", ephemeral=True)


# --- Slash Command: Index Status ---
@slash_command(
    name="indexstatus",
    description="Check index status",
    scopes=[int(guild_id)] if guild_id else None,
)
async def indexstatus_cmd(ctx: SlashContext):
    await ctx.defer()
    try:
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(pinecone_index_name)
        stats = index.describe_index_stats()
        doc_count = stats.get("total_vector_count", 0)
        last_updated = os.path.getmtime("data") if os.path.exists("data") else "Unknown"
        await ctx.send(
            content=f"Index Status:\n- Documents: {doc_count}\n- Last Updated: {last_updated}"
        )
    except Exception as e:
        logger.error(f"Error in indexstatus command: {e}", exc_info=True)
        await ctx.send(content=f"An error occurred: {e}", ephemeral=True)


# --- Main Function ---
def main():
    # Clone repo (sparse checkout)
    local_path = "data"
    sparse_checkout_file = "sparse_checkout.txt"
    if not os.path.exists(local_path):
        repo = Repo.init(local_path)
        repo.git.config("core.sparseCheckout", "true")
        os.makedirs(f"{local_path}/.git/info", exist_ok=True)
        with open(sparse_checkout_file, "r") as src, open(
            f"{local_path}/.git/info/sparse-checkout", "w"
        ) as dst:
            dst.write(src.read())
        repo.git.remote("add", "origin", repo_url)
        repo.git.fetch("origin")
        repo.git.checkout("main")

    # Index files
    include_patterns = ["**/*.cs", "**/*.csproj", "**/*.md", "**/*.yml", "**/*.py"]
    files = [
        f
        for pattern in include_patterns
        for f in glob.glob(f"data/{pattern}", recursive=True)
        if "/bin/" not in f and "/obj/" not in f and "/RobustToolbox/" not in f
    ]
    logger.info(f"Total files: {len(files)}")

    with Pool(processes=4) as pool:
        results = pool.map(chunk_file, files)

    code_chunks = [chunk for result in results for _, chunk in result]
    metadata = [
        {"file": file, "chunk": chunk} for result in results for file, chunk in result
    ]
    logger.info(f"Total chunks: {len(code_chunks)}")

    with open("metadata.json", "w") as f:
        json.dump(metadata, f)

    model = HuggingFaceEmbeddings(
        model_name=embed_model, model_kwargs={"device": device}
    )
    embeddings = model.embed_documents(code_chunks)

    pc = Pinecone(api_key=pinecone_api_key)
    if pinecone_index_name not in pc.list_indexes().names():
        pc.create_index(
            name=pinecone_index_name,
            dimension=len(embeddings[0]),
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    index = pc.Index(pinecone_index_name)

    batch_size = 100
    for i in range(0, len(embeddings), batch_size):
        batch_vectors = [
            (
                str(j),
                embeddings[j],
                {"chunk": code_chunks[j], "file": metadata[j]["file"]},
            )
            for j in range(i, min(i + batch_size, len(embeddings)))
        ]
        index.upsert(vectors=batch_vectors)
        logger.info(f"Upserted batch {i // batch_size + 1}")

    print("Starting bot...")
    bot.start()


if __name__ == "__main__":
    main()
