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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from multiprocessing import Pool
import uuid
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import Runnable

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


class SafeRetriever(Runnable):
    def __init__(self, retriever, namespace):
        self.retriever = retriever
        self.namespace = namespace

    def invoke(self, input, config=None):
        try:
            docs = self.retriever.invoke(input, config=config)
            # Filter out documents with missing metadata
            valid_docs = [
                doc
                for doc in docs
                if hasattr(doc, "metadata") and doc.metadata is not None
            ]
            if not valid_docs:
                logger.warning(
                    f"No valid documents retrieved for namespace: {self.namespace}"
                )
                return [Document(page_content="", metadata={"file": "unknown"})]
            return valid_docs
        except Exception as e:
            logger.error(f"Error in retriever for namespace {self.namespace}: {e}")
            return [Document(page_content="", metadata={"file": "unknown"})]


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
            step = 500  # Smaller chunks for metadata
            overlap = 100
            chunks = [code[i : i + step] for i in range(0, len(code), step - overlap)]
        return [(file_path, chunk) for chunk in chunks]
    except Exception as e:
        logger.error(f"Error parsing {file_path}: {e}")
        return []


def update_index(repo, files, index, model):
    try:
        repo.git.pull("origin", "master")  # Use correct branch
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
                {"file": file, "chunk": chunk[:500], "full_chunk": chunk}
                for result in results
                for file, chunk in result
            ]
            embeddings = model.embed_documents(code_chunks)
            batch_vectors = [
                (
                    str(i),
                    embeddings[i],
                    {"chunk": code_chunks[i][:500], "file": metadata[i]["file"]},
                )
                for i in range(len(embeddings))
            ]
            index.upsert(vectors=batch_vectors)
            logger.info(f"Updated index with {len(new_files)} new files")
            # Update metadata.json
            metadata_file = "metadata.json"
            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    existing_metadata = json.load(f)
            else:
                existing_metadata = []
            existing_metadata.extend(metadata)
            with open(metadata_file, "w") as f:
                json.dump(existing_metadata, f)
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
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(pinecone_index_name)

        # Load metadata for BM25
        with open("metadata.json", "r") as f:
            raw_docs = json.load(f)

        bm25_docs = [
            Document(page_content=doc["chunk"], metadata={"source": doc["file"]})
            for doc in raw_docs
        ]
        bm25_retriever = BM25Retriever.from_documents(bm25_docs)
        bm25_retriever.k = 2

        embedding = HuggingFaceEmbeddings(model_name=embed_model)
        dense_retrievers = []
        for ns in ["learned", "code", "qa_history", ""]:
            vs = PineconeVectorStore(
                index=index, embedding=embedding, text_key="chunk", namespace=ns
            )
            # Wrap the Pinecone retriever in SafeRetriever
            safe_retriever = SafeRetriever(
                vs.as_retriever(search_kwargs={"k": 2}), namespace=ns
            )
            dense_retrievers.append(safe_retriever)

        # Combine dense + sparse retrievers
        retriever = EnsembleRetriever(
            retrievers=dense_retrievers + [bm25_retriever],
            weights=[1.0] * (len(dense_retrievers) + 1),
        )

        # Retrieve documents
        docs = retriever.invoke(input_text)
        context = "\n".join(
            [doc.page_content[:500] for doc in docs if doc.page_content]
        )[:2000]
        logger.info(f"Context size: {len(context)} characters")

        # Initialize ChatGroq LLM with explicit base URL
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=groq_model,
            base_url="https://api.groq.com",
        )

        # Create a prompt template
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are Ungacode bot, a helpful and slightly witty assistant trained on a space station 14 forked codebase called Civ14. 
Speak informally, like a programmer explaining things to another programmer. Use humor where appropriate, but never be sarcastic or rude.
You can also attempt to match the tone of the user interacting with you.
""",
                ),
                MessagesPlaceholder(variable_name="history"),
                ("human", "Context: {context}\n\nQuestion: {input_text}"),
            ]
        )

        # Create a chain
        chain = prompt | llm

        # Wrap with history
        chain_with_history = RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=get_session_history,
            input_messages_key="input_text",
            history_messages_key="history",
        )

        # Limit history
        session_id = str(ctx.author.id)
        history = get_session_history(session_id)
        if len(history.messages) > 10:
            history.messages = history.messages[-10:]
        logger.info(f"History size: {len(history.messages)} messages")

        # Query with history
        response = chain_with_history.invoke(
            {"context": context, "input_text": input_text},
            config={"configurable": {"session_id": session_id}},
        )
        response_text = response.content  # Updated to use .content instead of .choices
        qa_id = str(uuid.uuid4())
        qa_text = f"Q: {input_text}\\nA: {response_text}"
        vector = embedding.embed_documents([qa_text])[0]
        index.upsert(
            vectors=[
                (
                    qa_id,
                    vector,
                    {"chunk": qa_text[:500], "file": f"qa_history/{qa_id}.txt"},
                )
            ],
            namespace="qa_history",
        )
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

        await ctx.send(embeds=embed)
    except Exception as e:
        logger.error(f"Error in query command: {str(e)}", exc_info=True)
        await ctx.send(content=f"An error occurred: {str(e)}", ephemeral=True)


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
        subprocess.run(
            ["git", "-C", local_path, "pull", "origin", "master"], check=True
        )  # Use master
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


# --- Slash Command: Learn ---
@slash_command(
    name="learn",
    description="Teach the bot new knowledge",
    scopes=[int(guild_id)] if guild_id else None,
    options=[
        SlashCommandOption(
            name="knowledge",
            description="The knowledge to learn (e.g., 'Your name is Vibebot')",
            type=OptionType.STRING,
            required=True,
        ),
    ],
)
async def learn_cmd(ctx: SlashContext, knowledge: str):
    await ctx.defer()
    try:
        logger.info(f"Learning new knowledge: {knowledge}")
        model = HuggingFaceEmbeddings(
            model_name=embed_model, model_kwargs={"device": device}
        )
        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(pinecone_index_name)
        knowledge_id = str(uuid.uuid4())
        embedding = model.embed_documents([knowledge])[0]
        metadata = {
            "chunk": knowledge[:500],
            "file": f"learned_knowledge/{knowledge_id}.txt",
        }
        index.upsert(
            vectors=[(knowledge_id, embedding, metadata)],
            namespace="learned",
        )
        metadata_entry = {
            "file": metadata["file"],
            "chunk": metadata["chunk"],
            "full_chunk": knowledge,
        }
        metadata_file = "metadata.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                existing_metadata = json.load(f)
        else:
            existing_metadata = []
        existing_metadata.append(metadata_entry)
        with open(metadata_file, "w") as f:
            json.dump(existing_metadata, f)
        await ctx.send(f"Learned new knowledge: '{knowledge}'")
        logger.info(f"Successfully learned knowledge with ID: {knowledge_id}")
    except Exception as e:
        logger.error(f"Error in learn command: {e}", exc_info=True)
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
        repo.git.checkout("master")
        repo.git.branch("--set-upstream-to=origin/master", "master")
    else:
        repo = Repo(local_path)

    # Initialize Pinecone and embedding model
    model = HuggingFaceEmbeddings(
        model_name=embed_model, model_kwargs={"device": device}
    )
    pc = Pinecone(api_key=pinecone_api_key)

    # Check if index exists
    if pinecone_index_name not in pc.list_indexes().names():
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
            {"file": file, "chunk": chunk[:500], "full_chunk": chunk}
            for result in results
            for file, chunk in result
        ]
        logger.info(f"Total chunks: {len(code_chunks)}")
        with open("metadata.json", "w") as f:
            json.dump(metadata, f)
        embeddings = model.embed_documents(code_chunks)
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
                    {"chunk": code_chunks[j][:500], "file": metadata[j]["file"]},
                )
                for j in range(i, min(i + batch_size, len(embeddings)))
            ]
            index.upsert(vectors=batch_vectors, namespace="code")
            logger.info(f"Upserted batch {i // batch_size + 1}")
    else:
        include_patterns = ["**/*.cs", "**/*.csproj", "**/*.md", "**/*.yml", "**/*.py"]
        files = [
            f
            for pattern in include_patterns
            for f in glob.glob(f"data/{pattern}", recursive=True)
            if "/bin/" not in f and "/obj/" not in f and "/RobustToolbox/" not in f
        ]
        index = pc.Index(pinecone_index_name)
        updated_files = update_index(repo, files, index, model)
        if updated_files is not None:
            logger.info(f"Updated {len(updated_files)} files")
        else:
            logger.info("No updates needed or error occurred")

    print("Starting bot...")
    bot.start()


if __name__ == "__main__":
    main()
