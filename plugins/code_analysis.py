from interactions import Client, slash_command, SlashContext, SlashCommandOption, OptionType
from loguru import logger

logger.add("bot.log", rotation="1 MB", level="INFO")

def register_plugin(bot: Client):
    @slash_command(
        name="analyze_code",
        description="Analyze code in the codebase",
        scopes=[int(bot.guilds[0].id)] if bot.guilds else None,
        options=[
            SlashCommandOption(
                name="query",
                description="Code or query to analyze",
                type=OptionType.STRING,
                required=True
            )
        ]
    )
    async def analyze_code(ctx: SlashContext, query: str):
        logger.info(f"Analyzing code with query: {query}")
        await ctx.send(content=f"Analyzing code with query: {query}")

    logger.info("Registered analyze_code command")