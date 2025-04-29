import click
import asyncio
import os
from manage_embedding import load_index, update_index
from querying import data_querying, memory
from loguru import logger

# Logging
logger.remove()
logger.add("bot.log", rotation="1 MB", level="INFO")
logger.add(sink=lambda msg: print(msg, end=""), level="INFO")


async def query(text: str, mode: str):
    logger.info(f"Running query: {text} (mode: {mode})")

    # Handle learn: keyword
    if text.startswith("learn:"):
        learn_content = text.replace("learn:", "", 1).strip()
        if not learn_content:
            logger.error("No content provided after 'learn:'")
            print("Error: Please provide content to learn after 'learn:'.")
            return
        learned_file = os.path.join("data", "learned_data", "learned_info.txt")
        os.makedirs(os.path.dirname(learned_file), exist_ok=True)
        try:
            with open(learned_file, "a", encoding="utf-8") as f:
                f.write(f"{learn_content}\n")
            logger.info("Successfully learned new information")
            print("Successfully learned new information!")
            await update_index()
        except Exception as e:
            logger.error(f"Error saving learned data: {e}")
            print(f"Error saving learned data: {e}")
        return

    index = await load_index()
    if index is None:
        logger.error("Index is not loaded.")
        print("Error: Index is not loaded.")
        return
    try:
        response = await data_querying(index, text, mode=mode)
        print(f"Query: {text}")
        print(f"Mode: {mode}")
        if isinstance(response, dict):
            print("Response:")
            print(f"Summary: {response.get('summary')}")
            print("Sources:", ", ".join(response.get("sources", [])))
        elif isinstance(response, list):
            print("Response:")
            for item in response:
                print(f"File: {item.get('file_path')}")
                print(f"Score: {item.get('score')}")
                print(f"Text: {item.get('text')}")
                print("-" * 50)
        else:
            print(f"Response: {response}")
    except Exception as e:
        logger.error(f"Error during query: {e}", exc_info=True)
        print(f"Error: {e}")


async def update_index_async():  # Renamed async function slightly for clarity
    logger.info("Running update_index...")
    try:
        changed_files = await update_index()
        if changed_files is None:
            logger.error("Error during update: update_index returned None")
            print("Error: update_index failed")
        elif changed_files:
            print(f"Updated index with {len(changed_files)} changed files.")
        else:
            print("No changes detected.")
    except Exception as e:
        logger.error(f"Error during update: {e}", exc_info=True)
        print(f"Error: {e}")


async def check_index_status_async():  # Renamed async function slightly for clarity
    logger.info("Checking index status...")
    try:
        index = await load_index()
        if index is None:
            logger.error("Index is not loaded.")
            print("Index is not loaded.")
        else:
            print("Index is loaded and ready.")
    except Exception as e:
        logger.error(f"Error checking index status: {e}", exc_info=True)
        print(f"Error: {e}")


async def clear_memory():
    logger.info("Clearing conversation memory...")
    try:
        memory.clear()
        logger.info("Conversation memory cleared.")
        print("Conversation memory cleared.")
    except Exception as e:
        logger.error(f"Error clearing memory: {e}", exc_info=True)
        print(f"Error: {e}")


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# Use 'query' as the command name explicitly if that's what you want to type
# Otherwise, the default would be 'query-cmd'
@cli.command("query")
@click.argument("text")
@click.option(
    "--mode",
    default="general",
    type=click.Choice(["general", "docs", "search", "debug", "generate"]),
    help="Query mode",
)
def query_cmd(text, mode):
    """Queries the indexed data or learns new information."""
    asyncio.run(query(text, mode))


# Explicitly name the command 'update'
@cli.command("update")
def update_cmd():  # Renamed function to avoid potential clashes
    """Updates the vector index with changes in the data directory."""
    asyncio.run(update_index_async())


# Explicitly name the command 'index-status'
@cli.command("index-status")
def index_status_cmd():  # Renamed function to avoid potential clashes
    """Checks if the index is loaded."""
    asyncio.run(check_index_status_async())


@cli.command("clear-memory")
def clear_memory():
    asyncio.run(clear_memory())


if __name__ == "__main__":
    cli()
