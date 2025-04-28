from interactions import (
    Client,
    slash_command,
    SlashContext,
    SlashCommandOption,
    OptionType,
    Embed,
)
from loguru import logger
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.settings import Settings

logger.add("bot.log", rotation="1 MB", level="INFO")


def split_text(text: str, max_length: int = 1024) -> list:
    if not isinstance(text, str):
        text = str(text)
    return [text[i : i + max_length] for i in range(0, len(text), max_length)]


def register_plugin(bot: Client):
    @slash_command(
        name="analyze_code",
        description="Analyze code in the codebase",
        scopes=[int(bot.guilds[0].id)] if bot.guilds else None,
        options=[
            SlashCommandOption(
                name="query",
                description="Code or query to analyze (e.g., 'find async methods in GameManager.cs')",
                type=OptionType.STRING,
                required=True,
            )
        ],
    )
    async def analyze_code(ctx: SlashContext, query: str):
        await ctx.defer()
        logger.info(f"Analyzing code with query: {query}")
        try:
            storage_context = StorageContext.from_defaults(persist_dir="./storage")
            index = load_index_from_storage(storage_context)
            if not index.docstore.docs:
                logger.warning("Index is empty. Please run /updatedb.")
                await ctx.send(
                    content="No code found in the database. Please update the index using /updatedb.",
                    ephemeral=True,
                )
                return

            retriever = index.as_retriever(similarity_top_k=3)
            nodes = retriever.retrieve(query)
            if not nodes:
                logger.info(f"No relevant documents found for query: {query}")
                await ctx.send(
                    content=f"No relevant code found for: {query}. Try rephrasing or updating the index.",
                    ephemeral=True,
                )
                return

            context = "\n".join([node.text for node in nodes])
            logger.info(f"Retrieved {len(nodes)} documents for query: {query}")

            prompt = f"Analyze the following code context and provide a detailed analysis for the query: {query}\n\nContext:\n{context}\n\nAnalysis:"
            response = await Settings.llm.acomplete(prompt)
            response_text = response.text

            embed = Embed(
                title="Code Analysis",
                description=f"**Query**: {query[:1000]}",
                color=0x00FF00,
            )
            response_chunks = split_text(response_text, max_length=1024)
            for i, chunk in enumerate(response_chunks, 1):
                embed.add_field(
                    name=(
                        f"Analysis (Part {i})"
                        if len(response_chunks) > 1
                        else "Analysis"
                    ),
                    value=chunk,
                    inline=False,
                )
            await ctx.send(embeds=embed)
        except Exception as e:
            logger.error(f"Error in analyze_code: {e}", exc_info=True)
            await ctx.send(content=f"An error occurred: {e}", ephemeral=True)

    logger.info("Registered analyze_code command")
