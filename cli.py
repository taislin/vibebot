import click
import asyncio
from manage_embedding import load_index, update_index
from querying import data_querying
from loguru import logger

# Logging
logger.remove()
logger.add("bot.log", rotation="1 MB", level="INFO")
logger.add(sink=lambda msg: print(msg, end=""), level="INFO")


async def query(text: str, mode: str):
    logger.info(f"Running query: {text} (mode: {mode})")
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


async def update():
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


async def index_status():
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


@click.group()
def cli():
    pass


@cli.command()
@click.argument("text")
@click.option(
    "--mode",
    default="general",
    type=click.Choice(["general", "docs", "search", "debug", "generate"]),
    help="Query mode",
)
def query_cmd(text, mode):
    asyncio.run(query(text, mode))


@cli.command()
def update():
    asyncio.run(update())


@cli.command()
def index_status():
    asyncio.run(index_status())


if __name__ == "__main__":
    cli()
