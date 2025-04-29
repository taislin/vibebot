# Vibebot

![vibebot](vibebot.gif)

## Overview

Vibebot is a Discord bot that leverages **[Retrieval-Augmented Generation (RAG)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)** to enhance responses from a large language model by referencing an external knowledge base. Built as a fork of [LLM-RAG-Bot](https://github.com/nur-zaman/LLM-RAG-Bot), it uses [Groq](https://groq.com/) for language model inference instead of OpenAI. The bot interacts with users via Discord slash commands and provides a command-line interface (CLI) and web interface for querying the knowledge base.

-   [Vibebot](#vibebot)
    -   [Overview](#overview)
    -   [Getting Started](#getting-started)
        -   [Prerequisites](#prerequisites)
    -   [Bot Usage](#bot-usage)
        -   [`/query`](#query)
        -   [`/updatedb`](#updatedb)
        -   [`/listfunctions`](#listfunctions)
        -   [`/indexstatus`](#indexstatus)
        -   [`/pullrepo`](#pullrepo)
        -   [`/ingesturl`](#ingesturl)
        -   [`/ingestrepo`](#ingestrepo)
        -   [`/clear_memory`](#clear_memory)
    -   [CLI](#cli)
        -   [Commands](#commands)
        -   [Query Modes](#query-modes)
    -   [Webserver](#webserver)
        -   [Setup](#setup)
        -   [Features](#features)
    -   [Notes](#notes)

## Getting Started

### Prerequisites

-   **Python 3.11** or higher
-   A Discord bot token
-   A Groq API key
-   A `data/` directory containing files for the knowledge base
-   Git (for `/pullrepo` command)

1. **Clone the repository**:

    ```bash
    git clone https://github.com/taislin/vibebot.git
    cd vibebot
    ```

2. **Install dependencies**:

    ```bash
    pip install discord-py-interactions interactions.py llama-index llama-index-embeddings-huggingface llama-index-llms-groq numpy python-dotenv PyYAML torch sentence-transformers pynacl setuptools tree-sitter tree-sitter-python tree-sitter-c-sharp sentence_transformers loguru requests langchain langchain_community beautifulsoup4
    ```

3. **Set up environment variables**:

    - Rename `.env.example` to `.env` in the project root.
    - Add your tokens and API keys:
        ```env
        DISCORD_BOT_TOKEN=your_discord_bot_token
        GROQ_API_KEY=your_groq_api_key
        GROQ_MODEL=llama-3.1-8b-instant
        ```

4. **Prepare the knowledge base**:

    - Place your files in the `data/` directory (create it if it doesn’t exist).
    - The bot will index these files to build the knowledge base.

5. **Launch the Discord bot:**

    ```bash
    python bot.py
    ```

    - You’ll see "Ready" in the console when the bot connects to Discord.

## Bot Usage

Vibebot supports the following Discord slash commands, available in the specified guild (if `GUILD_ID` is set) or globally.

### `/query`

-   **Description**: Query the knowledge base or learn new information.
-   **Options**:
    -   `input_text` (required): The query text. Prefix with "learn:" to append to `data/learned_data/learned_info.txt` and update the index.
    -   `mode` (optional): Query mode (`general`, `docs`, `search`, `debug`, `generate`). See [Query Modes](#query-modes).
-   **Examples**:
    -   `/query show me which animal entities are present in the code`
    -   `/query learn: new animal entity data`
    -   `/query input_text:Find all references to 'discord' in the code mode:search`

### `/updatedb`

-   **Description**: Updates the knowledge base by re-indexing files in `data/`.
-   **Example**: `/updatedb`

### `/listfunctions`

-   **Description**: Lists functions found in `.py` and `.cs` files in the `data/` directory.
-   **Example**: `/listfunctions`

### `/indexstatus`

-   **Description**: Displays the current state of the knowledge base index, including document count and last update time.
-   **Example**: `/indexstatus`

### `/pullrepo`

-   **Description**: Pulls the latest data from the GitHub repository in `data/` (must be a Git repository) and updates the knowledge base.
-   **Example**: `/pullrepo`

### `/ingesturl`

-   **Description**: Ingests content from a specified URL, saves it to `data/learned_data/`, and updates the index.
-   **Options**:
    -   `url` (required): The URL to scrape and ingest.
-   **Example**: `/ingesturl https://example.com`

### `/ingestrepo`

-   **Description**: Ingests `.py`, `.cs`, and `.md` files from a GitHub repository, saves them to `data/github/`, and updates the index.
-   **Options**:
    -   `repo_url` (required): The GitHub repository URL (e.g., `https://github.com/user/repo`).
-   **Example**: `/ingestrepo https://github.com/taislin/vibebot`

### `/clear_memory`

-   **Description**: Clears the conversation memory stored in `data/conversation_history.jsonl`.
-   **Example**: `/clear_memory`

## CLI

The command-line interface (`cli.py`) allows direct interaction with the knowledge base outside Discord.

### Commands

-   **Query the knowledge base**:

    ```bash
    python cli.py query "your query here" --mode <mode>
    ```

    Example:

    ```bash
    python cli.py query "show me which animal entities are present in the code" --mode docs
    ```

    Learning example:

    ```bash
    python cli.py query "learn: new animal entity data" --mode general
    ```

-   **Update the knowledge base**:

    ```bash
    python cli.py update
    ```

    Re-indexes files in `data/` to refresh the knowledge base.

-   **Check index status**:

    ```bash
    python cli.py index-status
    ```

    Displays the current state of the index.

-   **Clear conversation memory**:

    ```bash
    python cli.py clear-memory
    ```

    Clears the conversation history in `data/conversation_history.jsonl`.

### Query Modes

Vibebot supports multiple query modes to customize how queries are processed and responses are formatted. These modes are available in the CLI (`--mode`), Discord bot (`/query mode:`), and web interface (mode dropdown).

-   **general**:

    -   **Purpose**: Provides a balanced, conversational response based on the knowledge base.
    -   **Behavior**: Retrieves up to 3 relevant documents and generates a concise response using the Groq LLM.
    -   **Use Case**: General questions about the codebase or project.
    -   **Example**: `python cli.py query "What is the purpose of Vibebot?" --mode general`
    -   **Output**: A summarized answer (e.g., "Vibebot is a Discord bot that uses RAG to answer queries...").

-   **docs**:

    -   **Purpose**: Focuses on summarizing content from the knowledge base with source references.
    -   **Behavior**: Retrieves up to 5 documents and returns a structured response with file paths, scores, and text snippets.
    -   **Use Case**: Detailed queries requiring information from multiple files.
    -   **Example**: `python cli.py query "Show me which animal entities are present in the code" --mode docs`
    -   **Output**: A list of results (e.g., `File: data/file1.cs, Score: 0.85, Text: ...`).

-   **search**:

    -   **Purpose**: Acts like a search engine, returning raw document snippets with relevance scores.
    -   **Behavior**: Retrieves up to 10 documents, returns snippets with file paths and similarity scores, without LLM-generated text.
    -   **Use Case**: Finding specific code or text matches in the knowledge base.
    -   **Example**: `python cli.py query "Find all references to 'discord' in the code" --mode search`
    -   **Output**: A list of snippets (e.g., `File: data/bot.py, Score: 0.95, Text: import discord...`).

-   **debug**:

    -   **Purpose**: Provides diagnostic information for troubleshooting queries.
    -   **Behavior**: Retrieves up to 3 documents, returns raw snippets with scores, and logs detailed query processing information in `bot.log`.
    -   **Use Case**: Debugging issues with the knowledge base or query results.
    -   **Example**: `python cli.py query "Why is the index failing?" --mode debug`
    -   **Output**: Query details and sources, with additional logs in `bot.log`.

-   **generate**:
    -   **Purpose**: Emphasizes creative or generative responses with minimal reliance on the knowledge base.
    -   **Behavior**: Uses the Groq LLM to generate a response, optionally informed by up to 2 documents.
    -   **Use Case**: Creative tasks or questions requiring less strict adherence to the knowledge base.
    -   **Example**: `python cli.py query "Write a poem about Vibebot" --mode generate`
    -   **Output**: A creative response (e.g., a poem about Vibebot).

## Webserver

Vibebot includes a web interface for querying the knowledge base via a browser.

### Setup

1. Ensure `index.html` and `server.py` are in the project root.
2. Run the web server:

    ```bash
    python server.py
    ```

3. Access the interface at `http://localhost:8000`.

### Features

-   **Query Input**: Enter queries in a text field (e.g., "show me which animal entities are present in the code") and select a query mode from a dropdown (`general`, `docs`, `search`, `debug`, `generate`).
-   **Scrollable Chat History**: View all queries and responses in a dark-themed, scrollable container.
-   **Clear Memory**: Reset conversation history via a button.
-   **Responsive Design**: Built with Tailwind CSS for a clean, dark-mode interface.

## Notes

-   Ensure sufficient system resources for indexing large datasets (recommended: >4GB RAM or GPU for faster embeddings).
-   Logs are stored in `bot.log` for debugging. Enable `exc_info=True` in error logs for detailed stack traces.
-   For production, secure the web server with authentication and HTTPS.
-   The `data/` directory should exclude binary files (e.g., `.exe`, `.dll`) to avoid indexing errors.
