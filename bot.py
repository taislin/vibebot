import os
from dotenv import load_dotenv
from interactions import (
    Client,
    Intents,
    slash_command,
    SlashContext,
    listen,
    slash_option,
    OptionType,
)
from interactions import Embed
import importlib
import glob

# --- LlamaIndex Configuration ---
from llama_index.llms.groq import Groq
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- Load environment variables ---
load_dotenv()

from loguru import logger

logger.add("bot.log", rotation="1 MB", level="INFO")

# --- Configure Settings BEFORE anything else ---
groq_api_key = os.getenv("GROQ_API_KEY")
discord_bot_token = os.getenv("DISCORD_BOT_TOKEN")
groq_model = os.getenv("GROQ_MODEL", "gemma2-9b-it")
guild_id = os.getenv("GUILD_ID")
embed_model = os.getenv("EMBED_MODEL", "jinaai/jina-embeddings-v2-base-code")

if not discord_bot_token:
    raise ValueError("DISCORD_BOT_TOKEN not found.")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found.")

print("Configuring Groq LLM...")
Settings.llm = Groq(model=groq_model, api_key=groq_api_key)

print("Configuring HuggingFace Embeddings...")
Settings.embed_model = HuggingFaceEmbedding(
    model_name=embed_model,
)

# --- Now safely import your app modules ---
print("Importing application modules...")
from querying import data_querying
from manage_embedding import update_index, run_blocking

# --- Use interactions.py Intents ---
intents = Intents.DEFAULT

# --- Use interactions.py Client ---
bot = Client(intents=intents)


# --- Helper: Save a new memory dynamically ---
async def save_memory(fact_text: str):
    from llama_index.core import Document, load_index_from_storage
    from llama_index.core.storage.storage_context import StorageContext

    persist_dir = "./storage"

    # Load storage context directly
    storage_context = await run_blocking(
        StorageContext.from_defaults, persist_dir=persist_dir
    )
    index = await run_blocking(load_index_from_storage, storage_context)

    # Insert the new document
    new_doc = Document(text=fact_text)
    index.insert(new_doc)

    # Persist changes
    await run_blocking(index.storage_context.persist, persist_dir=persist_dir)

    return f'I\'ve learned: "{fact_text}"'


# --- Helper: Load plugins dynamically ---
def load_plugins():
    for plugin_file in glob.glob("plugins/*.py"):
        module_name = plugin_file.replace("/", ".").replace("\\", ".")[:-3]
        module = importlib.import_module(module_name)
        module.register_plugin(bot)


load_plugins()


# --- Listen for Bot Ready Event ---
@listen()
async def on_ready():
    print(f"Logged in as {bot.user.username}")
    print("Ready")


# --- Query Slash Command ---
@slash_command(name="query", description="Enter your query", scopes=[guild_id])
@slash_option(
    name="input_text",
    description="Input text or learning fact",
    required=True,
    opt_type=OptionType.STRING,
)
@slash_option(
    name="mode",
    description="Query mode: general, docs, search, debug",
    required=False,
    opt_type=OptionType.STRING,
    choices=[
        {"name": "General", "value": "general"},
        {"name": "Documentation", "value": "docs"},
        {"name": "Code Search", "value": "search"},
        {"name": "Debugging", "value": "debug"},
    ],
)
async def get_response(ctx: SlashContext, input_text: str, mode: str = "general"):
    await ctx.defer()
    try:
        if input_text.lower().startswith("learn:"):
            fact = input_text[len("learn:") :].strip()
            response = await save_memory(fact)
        else:
            response = await data_querying(
                input_text, mode=mode, user_id=str(ctx.author.id)
            )
        embed = Embed(
            title="Query Response",
            description=f"**Input**: {input_text}\n**Mode**: {mode}",
            color=0x00FF00,
        )
        embed.add_field(name="Response", value=response, inline=False)
        await ctx.send(embeds=embed)
    except Exception as e:
        await ctx.send(f"An error occurred: {e}", ephemeral=True)


# --- Update DB Slash Command ---
@slash_command(
    name="updatedb",
    description="Update your information database",
    scopes=[guild_id],
)
async def updated_database(ctx: SlashContext):
    await ctx.defer()
    print("Received updatedb command")
    try:
        update_results = await update_index()
        if update_results is not None:
            num_updated = sum(1 for refreshed in update_results if refreshed)
            response = f"Updated/Refreshed {num_updated} documents/nodes."
        else:
            response = "Index not found or error during update. Please check logs."
        await ctx.send(response)
    except Exception as e:
        print(f"Error during DB update: {e}")
        await ctx.send(
            f"An error occurred while updating the database: {e}", ephemeral=True
        )


@slash_command(
    name="listfunctions",
    description="List functions in the codebase",
    scopes=[guild_id],
)
async def list_functions(ctx: SlashContext):
    await ctx.defer()
    try:
        from code_parser import list_functions

        functions = await list_functions("data")
        response = "\n".join(functions) if functions else "No functions found."
        await ctx.send(f"**Functions in Codebase**:\n{response}")
    except Exception as e:
        await ctx.send(f"An error occurred: {e}", ephemeral=True)


@slash_command(
    name="indexstatus", description="Check index status", scopes=[MY_GUILD_ID]
)
async def index_status(ctx: SlashContext):
    await ctx.defer()
    try:
        index = await load_index("data")
        if index is None:
            await ctx.send("Index not found.", ephemeral=True)
            return
        doc_count = len(index.docstore.docs)
        last_updated = (
            os.path.getmtime("./storage") if os.path.exists("./storage") else "Unknown"
        )
        await ctx.send(
            f"Index Status:\n- Documents: {doc_count}\n- Last Updated: {last_updated}"
        )
    except Exception as e:
        await ctx.send(f"An error occurred: {e}", ephemeral=True)


@slash_command(
    name="pullrepo",
    description="Pull latest codebase from GitHub",
    scopes=[MY_GUILD_ID],
)
async def pull_repo(ctx: SlashContext):
    await ctx.defer()
    try:
        import subprocess

        subprocess.run(["git", "-C", "data", "pull", "origin", "master"], check=True)
        update_results = await update_index()
        await ctx.send(
            f"Repository updated. Refreshed {sum(update_results)} documents."
        )
    except Exception as e:
        await ctx.send(f"An error occurred: {e}", ephemeral=True)


# --- Start Bot ---
print("Starting bot...")
bot.start(discord_bot_token)
