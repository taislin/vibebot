# Vibebot

![vibebot](vibebot.gif)

A Discord bot that uses retrieval-augmented generation (RAG) to query and analyze a codebase. The bot indexes code files from a Git repository, stores embeddings in Pinecone, and answers queries using Groq's language model. It supports commands to query the codebase, update the index, learn new knowledge, and manage conversation history.

## Features

-   Indexes code files (`.cs`, `.py`, `.md`, `.yml`, `.csproj`) from a Git repository.
-   Uses HuggingFace embeddings and Pinecone for efficient code retrieval.
-   Answers queries with context-aware responses via Groq's LLM.
-   Supports Discord slash commands for querying, updating, and learning.
-   Maintains per-user conversation history.
-   Dynamically loads plugins for extensibility.

## Prerequisites

-   Python 3.8+
-   Git
-   Pinecone account and API key
-   Groq account and API key
-   Discord bot token and guild ID
-   A GitHub repository

## Setup

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/taislin/vibebot.git
    cd vibebot
    ```

2. **Install Dependencies**:

    ```bash
    pip install interactions loguru langchain langchain-huggingface langchain-groq langchain-pinecone pinecone-client groq gitpython tree-sitter tree-sitter-c-sharp
    ```

3. **Create `.env` File**:
   Create a `.env` file in the project root with the following:

    ```
    DISCORD_BOT_TOKEN=your_discord_bot_token
    GUILD_ID=your_guild_id
    PINECONE_API_KEY=your_pinecone_key
    PINECONE_INDEX_NAME=your_index_name
    GROQ_API_KEY=your_groq_key
    GROQ_MODEL=llama-3.1-8b-instant
    EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
    REPO_URL=your_repo_url
    USE_CUDA=false
    ```

4. **Fine tune `sparse_checkout.txt`**:
   Change the `sparse_checkout.txt` file in the project root to define which files to retrieve from the repository, for example:

    ```
    *.cs
    *.csproj
    *.md
    *.yml
    *.py
    ```

5. **Run the Bot**:
    ```bash
    python discordbot.py
    ```
    The bot will clone the repository, index files, and start on Discord.

## Bot Commands

All commands are Discord slash commands, available in the specified guild (`GUILD_ID`).

-   **/query `<input_text>` `[mode]`**

    -   Queries the codebase.
    -   `input_text`: Your question (e.g., "What is the codebase's name?").
    -   `mode` (optional): `general`, `docs`, `search`, `debug`.
    -   Example: `/query What scripts do you have? general`

-   **/learn `<knowledge>`**

    -   Teaches the bot new knowledge, stored in the index.
    -   `knowledge`: Text to learn (e.g., "Your name is Vibebot").
    -   Example: `/learn Your name is Vibebot`

-   **/updatedb**

    -   Updates the Pinecone index with new or changed files.
    -   Example: `/updatedb`

-   **/pullrepo**

    -   Pulls the latest repository data and updates the index.
    -   Example: `/pullrepo`

-   **/clear_memory**

    -   Clears the user's conversation history.
    -   Example: `/clear_memory`

-   **/indexstatus**
    -   Shows the number of indexed documents and last update time.
    -   Example: `/indexstatus`

## Usage

1. Invite the bot to your Discord server using the bot token.
2. Use slash commands in the specified guild to interact with the bot.
3. Query the codebase (e.g., `/query What is the codebase's name?`) to get context-aware answers.
4. Teach the bot new facts with `/learn` (e.g., `/learn The codebase is Civ14`).
5. Update the index with `/updatedb` or `/pullrepo` when the codebase changes.

## Project Structure

-   `discordbot.py`: Main script for indexing and running the Discord bot.
-   `vibebot.py`: Command line interface for the bot. Use `python vibebot.py query "hello"` to send a query. You will need Pinecone to be set up first.
-   `vibebot_update.py`: Generates and stores data in Pinecone only.

-   `data/`: Cloned repository with code files.
-   `metadata.json`: Stores metadata for indexed chunks.
-   `vibebot.log`: Log file for debugging.
-   `sparse_checkout.txt`: Defines files to include in sparse checkout.

## Troubleshooting

-   **Git Errors**: Ensure `REPO_URL` is correct and accessible.
-   **Pinecone Errors**: Verify `PINECONE_API_KEY` and `PINECONE_INDEX_NAME`. Delete and recreate the index if metadata issues occur.
-   **Bot Not Responding**: Check `DISCORD_BOT_TOKEN` and `GUILD_ID`. Ensure the bot is invited to the server.
-   View `vibebot.log` for detailed error messages.

## License

MIT License. See `LICENSE` for details.
