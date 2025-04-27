import asyncio
import click
import os
import subprocess
from loguru import logger
from manage_embedding import load_index, update_index
from querying import data_querying
from code_parser import list_functions

# Configure logging
logger.remove()
logger.add("bot.log", rotation="1 MB", level="INFO")
logger.add(sink=lambda msg: print(msg, end=""), level="INFO")


@click.group()
def cli():
    """Command-line interface for LLM-RAG-Bot to query and manage a game development codebase."""
    pass


@cli.command()
@click.argument("query_text")
@click.option(
    "--mode",
    default="general",
    type=click.Choice(["general", "docs", "search", "debug", "generate"]),
    help="Query mode: general, docs, search, debug, generate",
)
def query(query_text, mode):
    """Query the codebase with a given text and mode."""
    logger.info(f"Running query: {query_text} (mode: {mode})")
    try:
        response = asyncio.run(data_querying(query_text, mode=mode))
        click.echo(f"Query: {query_text}\nMode: {mode}\nResponse: {response}")
    except Exception as e:
        logger.error(f"Error during query: {e}", exc_info=True)
        click.echo(f"Error: {e}")


@cli.command()
def update():
    """Update the codebase index."""
    logger.info("Running index update")
    try:
        update_results = asyncio.run(update_index())
        if update_results is not None:
            num_updated = sum(1 for refreshed in update_results if refreshed)
            click.echo(f"Updated/Refreshed {num_updated} documents.")
        else:
            click.echo("Index not found or error during update. Check logs.")
    except Exception as e:
        logger.error(f"Error during update: {e}", exc_info=True)
        click.echo(f"Error: {e}")


@cli.command()
def list_functions():
    """List functions in the codebase."""
    logger.info("Listing functions in codebase")
    try:
        functions = asyncio.run(list_functions("data"))
        response = "\n".join(functions) if functions else "No functions found."
        click.echo(f"Functions in Codebase:\n{response}")
    except Exception as e:
        logger.error(f"Error listing functions: {e}", exc_info=True)
        click.echo(f"Error: {e}")


@cli.command()
def index_status():
    """Check the status of the index."""
    logger.info("Checking index status")
    try:
        index = asyncio.run(load_index("data"))
        if index is None:
            click.echo("Index not found.")
            return
        doc_count = len(index.docstore.docs)
        last_updated = (
            os.path.getmtime("./storage") if os.path.exists("./storage") else "Unknown"
        )
        click.echo(
            f"Index Status:\n- Documents: {doc_count}\n- Last Updated: {last_updated}"
        )
    except Exception as e:
        logger.error(f"Error checking index status: {e}", exc_info=True)
        click.echo(f"Error: {e}")


@cli.command()
def pull_repo():
    """Pull the latest codebase from GitHub."""
    logger.info("Pulling latest codebase from GitHub")
    try:
        subprocess.run(["git", "-C", "data", "pull", "origin", "master"], check=True)
        update_results = asyncio.run(update_index())
        num_updated = (
            sum(1 for refreshed in update_results if refreshed) if update_results else 0
        )
        click.echo(f"Repository updated. Refreshed {num_updated} documents.")
    except Exception as e:
        logger.error(f"Error pulling repository: {e}", exc_info=True)
        click.echo(f"Error: {e}")


if __name__ == "__main__":
    cli()
