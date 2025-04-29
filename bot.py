import os
from dotenv import load_dotenv
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
import importlib
import glob
from loguru import logger
from manage_embedding import update_index, load_index
from querying import data_querying, memory

# Logging
logger.remove()
logger.add("bot.log", rotation="1 MB", level="INFO")
logger.add(sink=lambda msg: print(msg, end=""), level="INFO")

# Load environment variables
load_dotenv()
discord_bot_token = os.getenv("DISCORD_BOT_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")
guild_id = os.getenv("GUILD_ID")

if not discord_bot_token:
    raise ValueError("DISCORD_BOT_TOKEN not found.")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found.")

# Initialize bot
intents = Intents.DEFAULT | Intents.GUILDS
bot = Client(intents=intents, token=discord_bot_token)


# Helper: Split long text into chunks
def split_text(text: str, max_length: int = 1024) -> list:
    if not isinstance(text, str):
        text = str(text)
    return [text[i : i + max_length] for i in range(0, len(text), max_length)]


# Helper: Load plugins dynamically
def load_plugins():
    for plugin_file in glob.glob("plugins/*.py"):
        try:
            module_name = plugin_file.replace("/", ".").replace("\\", ".")[:-3]
            module = importlib.import_module(module_name)
            module.register_plugin(bot)
            logger.info(f"Loaded plugin: {module_name}")
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_file}: {e}", exc_info=True)


load_plugins()


# Listen for Bot Ready Event
@listen()
async def on_ready():
    print(f"Logged in as {bot.user}")
    print("Ready")


# Query Slash Command
@slash_command(
    name="query",
    description="Enter your query",
    scopes=[int(guild_id)] if guild_id else None,
    options=[
        SlashCommandOption(
            name="input_text",
            description="Input text or learning fact",
            type=OptionType.STRING,
            required=True,
        ),
        SlashCommandOption(
            name="mode",
            description="Query mode: general, docs, search, debug, generate",
            type=OptionType.STRING,
            required=False,
            choices=[
                {"name": "General", "value": "general"},
                {"name": "Documentation", "value": "docs"},
                {"name": "Code Search", "value": "search"},
                {"name": "Debugging", "value": "debug"},
                {"name": "Generate", "value": "generate"},
            ],
        ),
    ],
)
async def get_response(ctx: SlashContext, input_text: str, mode: str = "general"):
    await ctx.defer()
    try:
        logger.info(f"Processing query: {input_text} (mode: {mode})")
        index = await load_index()
        if index is None:
            await ctx.send("Error: Index is not loaded.", ephemeral=True)
            return

        response = await data_querying(index, input_text, mode=mode)
        embed = Embed(
            title="Query Response",
            description=f"**Input**: {input_text[:1000]}\n**Mode**: {mode}",
            color=0x00FF00,
        )

        if isinstance(response, dict):
            response_text = response.get("summary", "")
            sources = ", ".join(response.get("sources", []))
            response_chunks = split_text(f"{response_text}\nSources: {sources}")
        elif isinstance(response, list):
            response_text = "\n".join(
                f"File: {item.get('file_path')}\nScore: {item.get('score')}\nText: {item.get('text')}\n{'-' * 50}"
                for item in response
            )
            response_chunks = split_text(response_text)
        else:
            response_chunks = split_text(str(response))

        for i, chunk in enumerate(response_chunks, 1):
            embed.add_field(
                name=f"Response (Part {i})" if len(response_chunks) > 1 else "Response",
                value=chunk,
                inline=False,
            )

        # Add conversation history
        history = memory.load_memory_variables({})["history"]
        if history:
            history_text = "\n".join(
                f"{'Human' if i % 2 == 0 else 'Assistant'}: {msg.content}"
                for i, msg in enumerate(history)
            )
            history_chunks = split_text(history_text, max_length=1024)
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


# Clear Memory Slash Command
@slash_command(
    name="clear_memory",
    description="Clear conversation memory",
    scopes=[int(guild_id)] if guild_id else None,
)
async def clear_memory_cmd(ctx: SlashContext):
    await ctx.defer()
    try:
        memory.clear()
        await ctx.send("Conversation memory cleared.", ephemeral=True)
    except Exception as e:
        logger.error(f"Error in clear_memory command: {e}", exc_info=True)
        await ctx.send(content=f"An error occurred: {e}", ephemeral=True)


# Update DB Slash Command
@slash_command(
    name="updatedb",
    description="Update your information database",
    scopes=[int(guild_id)] if guild_id else None,
)
async def updated_database(ctx: SlashContext):
    await ctx.defer()
    try:
        update_results = await update_index()
        if update_results is not None:
            response = f"Updated/Refreshed {len(update_results)} documents/nodes."
        else:
            response = "Index not found or error during update. Please check logs."
        await ctx.send(content=response)
    except Exception as e:
        logger.error(f"Error in updatedb command: {e}", exc_info=True)
        await ctx.send(content=f"An error occurred: {e}", ephemeral=True)


# List Functions Slash Command
@slash_command(
    name="listfunctions",
    description="List functions in the codebase",
    scopes=[int(guild_id)] if guild_id else None,
)
async def list_functions(ctx: SlashContext):
    await ctx.defer()
    try:
        from code_parser import list_functions

        functions = await list_functions("data")
        response = "\n".join(functions) if functions else "No functions found."
        response_chunks = split_text(response, max_length=1024)
        embed = Embed(title="Functions in Codebase", color=0x00FF00)
        for i, chunk in enumerate(response_chunks, 1):
            embed.add_field(
                name=(
                    f"Functions (Part {i})" if len(response_chunks) > 1 else "Functions"
                ),
                value=chunk,
                inline=False,
            )
        await ctx.send(embeds=embed)
    except Exception as e:
        logger.error(f"Error in listfunctions command: {e}", exc_info=True)
        await ctx.send(content=f"An error occurred: {e}", ephemeral=True)


# Index Status Slash Command
@slash_command(
    name="indexstatus",
    description="Check index status",
    scopes=[int(guild_id)] if guild_id else None,
)
async def index_status(ctx: SlashContext):
    await ctx.defer()
    try:
        index = await load_index()
        if index is None:
            await ctx.send(content="Index not found.", ephemeral=True)
            return
        doc_count = len(index.docstore.docs)
        last_updated = (
            os.path.getmtime("./storage") if os.path.exists("./storage") else "Unknown"
        )
        await ctx.send(
            content=f"Index Status:\n- Documents: {doc_count}\n- Last Updated: {last_updated}"
        )
    except Exception as e:
        logger.error(f"Error in indexstatus command: {e}", exc_info=True)
        await ctx.send(content=f"An error occurred: {e}", ephemeral=True)


# Pull Repo Slash Command
@slash_command(
    name="pullrepo",
    description="Pull latest data from GitHub",
    scopes=[int(guild_id)] if guild_id else None,
)
async def pull_repo(ctx: SlashContext):
    await ctx.defer()
    try:
        import subprocess

        subprocess.run(["git", "-C", "data", "pull", "origin", "main"], check=True)
        update_results = await update_index()
        num_updated = len(update_results) if update_results else 0
        await ctx.send(
            content=f"Repository updated. Refreshed {num_updated} documents."
        )
    except Exception as e:
        logger.error(f"Error in pullrepo command: {e}", exc_info=True)
        await ctx.send(content=f"An error occurred: {e}", ephemeral=True)


@slash_command(name="ingesturl", description="Ingest content from a URL")
async def ingest_url(ctx: SlashContext, url: str):
    await ctx.defer()
    try:
        import requests
        from bs4 import BeautifulSoup

        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(strip=True)
        # Save to data/learned_data/
        os.makedirs("data/learned_data", exist_ok=True)
        file_path = f"data/learned_data/url_{url.replace('/', '_')}.txt"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        await update_index()
        await ctx.send(f"Ingested content from {url} and updated index.")
    except Exception as e:
        logger.error(f"Error ingesting URL {url}: {e}")
        await ctx.send(f"Error: {e}")


@slash_command(name="ingestrepo", description="Ingest files from a GitHub repo")
async def ingest_repo(ctx: SlashContext, repo_url: str):
    await ctx.defer()
    try:
        import requests

        repo = repo_url.replace("https://github.com/", "")
        api_url = f"https://api.github.com/repos/{repo}/contents"
        headers = {"Accept": "application/vnd.github.v3+json"}
        response = requests.get(api_url, headers=headers)
        files = response.json()
        os.makedirs("data/github", exist_ok=True)
        for file in files:
            if file["type"] == "file" and file["name"].endswith((".py", ".cs", ".md")):
                file_content = requests.get(file["download_url"]).text
                file_path = f"data/github/{file['name']}"
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(file_content)
        await update_index()
        await ctx.send(f"Ingested files from {repo_url}.")
    except Exception as e:
        logger.error(f"Error ingesting repo {repo_url}: {e}")
        await ctx.send(f"Error: {e}")


# Start Bot
print("Starting bot...")
bot.start()
