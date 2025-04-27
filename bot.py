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

# --- LlamaIndex Configuration ---
from llama_index.llms.groq import Groq
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# --- Load environment variables ---
load_dotenv()

# --- Configure Settings BEFORE anything else ---
groq_api_key = os.getenv("GROQ_API_KEY")
discord_bot_token = os.getenv("DISCORD_BOT_TOKEN")
groq_model = os.getenv("GROQ_MODEL", "gemma2-9b-it")

if not discord_bot_token:
    raise ValueError("DISCORD_BOT_TOKEN not found.")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found.")

print("Configuring Groq LLM...")
Settings.llm = Groq(model=groq_model, api_key=groq_api_key)

print("Configuring HuggingFace Embeddings...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# ✅ Now LlamaIndex is configured correctly!

# --- Now safely import your app modules ---
print("Importing application modules...")
from querying import data_querying
from manage_embedding import update_index, run_blocking

# --- Use interactions.py Intents ---
intents = Intents.DEFAULT

# --- Use interactions.py Client ---
bot = Client(intents=intents)

MY_GUILD_ID = 468979034571931648


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


# --- Listen for Bot Ready Event ---
@listen()
async def on_ready():
    print(f"Logged in as {bot.user.username}")
    print("Ready")


# --- Query Slash Command ---
@slash_command(name="query", description="Enter your query", scopes=[MY_GUILD_ID])
@slash_option(
    name="input_text",
    description="Input text or learning fact",
    required=True,
    opt_type=OptionType.STRING,
)
async def get_response(ctx: SlashContext, input_text: str):
    await ctx.defer()
    print(f"Received query: {input_text}")
    try:
        if input_text.lower().startswith("learn:"):
            fact = input_text[len("learn:") :].strip()
            response = await save_memory(fact)
        else:
            response = await data_querying(input_text)

        response_message = f"**Input Query**: {input_text}\n\n{response}"
        await ctx.send(response_message)

    except Exception as e:
        print(f"Error during query processing: {e}")
        await ctx.send(f"An error occurred: {e}", ephemeral=True)


# --- Update DB Slash Command ---
@slash_command(
    name="updatedb",
    description="Update your information database",
    scopes=[MY_GUILD_ID],
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


# --- Start Bot ---
print("Starting bot...")
bot.start(discord_bot_token)
