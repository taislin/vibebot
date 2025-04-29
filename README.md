# Vibebot

![vibebot](vibebot.gif)

## Overview

Vibebot is a Discord bot that leverages **[Retrieval-Augmented Generation (RAG)](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)** to enhance responses from a large language model by referencing an external knowledge base. Built as a fork of [LLM-RAG-Bot](https://github.com/nur-zaman/LLM-RAG-Bot), it uses [Groq](https://groq.com/) for language model inference instead of OpenAI. The bot interacts with users via Discord slash commands and provides a command-line interface (CLI) and web interface for querying the knowledge base.

- [Vibebot](#vibebot)
  - [Overview](#overview)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
  - [Bot Usage](#bot-usage)
    - [`/query`](#query)
    - [`/updatedb`](#updatedb)
    - [`/listfunctions`](#listfunctions)
    - [`/indexstatus`](#indexstatus)
    - [`/pullrepo`](#pullrepo)
  - [CLI](#cli)
    - [Commands](#commands)
    - [Query Modes](#query-modes)
  - [Webserver](#webserver)
    - [Setup](#setup)
    - [Features](#features)
  - [Notes](#notes)

## Getting Started

### Prerequisites

-   Python 3.10 or higher
-   A Discord bot token
-   A Groq API key
-   A `data/` directory containing files for the knowledge base

1. **Clone the repository**:

    ```bash
    git clone https://github.com/taislin/vibebot.git
    cd vibebot
    ```

2. **Install dependencies**:

    ```bash
    pip install discord-py-interactions interactions.py llama-index llama-index-embeddings-huggingface llama-index-llms-groq numpy python-dotenv PyYAML torch sentence-transformers pynacl setuptools tree-sitter tree-sitter-python tree-sitter-c-sharp sentence_transformers loguru requests langchain langchain_community
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

Vibebot supports the following Discord slash commands:

### `/query`

-   **Description**: Query the knowledge base.
-   **Options**:
    -   `input_text` (required): The query text. Prefix with "learn:" to train the model on the input.
    -   `mode` (optional): Query mode (`general`, `docs`, `search`, `debug`). See [Query Modes](#query-modes) for details.
    -   Example: `/query show me which animal entities are present in the code`
    -   Example (learning): `/query learn: new animal entity data`
    -   Example with mode: `/query input_text:Find all references to 'discord' in the code mode:search`

### `/updatedb`

-   **Description**: Updates the knowledge base by re-indexing files in `data/`.

### `/listfunctions`

-   **Description**: Lists functions found in `.py` and `.cs` files in the `data/` directory.

### `/indexstatus`

-   **Description**: Displays the current state of the knowledge base index, including document count and last update time.

### `/pullrepo`

-   **Description**: Pulls the latest data from the GitHub repository in `data/` and updates the knowledge base.

## CLI

The command-line interface (`cli.py`) allows direct interaction with the knowledge base outside Discord.

### Commands

-   **Query the knowledge base**:

    ```bash
    python cli.py query-cmd "your query here" --mode docs
    ```

    Example:

    ```bash
    python cli.py query-cmd "show me which animal entities are present in the code" --mode docs
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

### Query Modes

Vibebot supports multiple query modes to customize how queries are processed and responses are formatted. These modes are available in the CLI (`--mode`), Discord bot (`/query mode:`), and web interface (mode dropdown).

-   **general**:

    -   **Purpose**: Provides a balanced, conversational response based on the knowledge base.
    -   **Behavior**: Retrieves up to 5 relevant documents and generates a concise response using the Groq LLM.
    -   **Use Case**: General questions about the codebase or project.
    -   **Example**: `python cli.py query "What is the purpose of Vibebot?" --mode general`
    -   **Output**: A summarized answer (e.g., "Vibebot is a Discord bot that uses RAG to answer queries...").

-   **docs**:

    -   **Purpose**: Focuses on summarizing content from the knowledge base with source references.
    -   **Behavior**: Retrieves up to 10 documents, uses `tree_summarize` to aggregate content, and returns a structured response with a summary and list of source file paths.
    -   **Use Case**: Detailed queries requiring information from multiple files.
    -   **Example**: `python cli.py query-cmd "Show me which animal entities are present in the code" --mode docs`
    -   **Output**: A summary (e.g., "Animal entities include...") and sources (e.g., `data/file1.cs, data/file2.py`).

-   **search**:

    -   **Purpose**: Acts like a search engine, returning raw document snippets with relevance scores.
    -   **Behavior**: Retrieves up to 8 documents, returns snippets (up to 500 characters each) with file paths and similarity scores, without LLM-generated text.
    -   **Use Case**: Finding specific code or text matches in the knowledge base.
    -   **Example**: `python cli.py -cmd "Find all references to 'discord' in the code" --mode search`
    -   **Output**: A list of snippets (e.g., `File: data/bot.py, Score: 0.95, Text: import discord...`).

-   **debug**:

    -   **Purpose**: Provides diagnostic information for troubleshooting queries.
    -   **Behavior**: Retrieves up to 5 documents, returns raw snippets with scores, and enables verbose logging for detailed query processing information (check `bot.log`).
    -   **Use Case**: Debugging issues with the knowledge base or query results.
    -   **Example**: `python cli.py query-cmd "Why is the index failing?" --mode debug`
    -   **Output**: Similar to `search` but with additional logs in `bot.log`.

-   **generate**:
    -   **Purpose**: Emphasizes creative or generative responses with minimal reliance on the knowledge base.
    -   **Behavior**: Retrieves only 2 documents and prioritizes the Groq LLM’s generative capabilities for a creative or elaborated response.
    -   **Use Case**: Creative tasks or questions requiring less strict adherence to the knowledge base.
    -   **Example**: `python cli.py query-cmd "Write a poem about Vibebot" --mode generate`
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

-   **Query Input**: Enter queries in a text field (e.g., "show me which animal entities are present in the code") and select a query mode from a dropdown (general, docs, search, debug, generate).
-   **Scrollable Chat History**: View all queries and responses in a dark-themed, scrollable container.
-   **Responsive Design**: Built with Tailwind CSS for a clean, dark-mode interface.

## Notes

-   Ensure sufficient system resources for indexing large datasets (recommended: >4GB RAM or GPU for faster embeddings).
-   Logs are stored in `bot.log` for debugging.
-   For production, secure the web server with authentication and HTTPS.
-   The `generate` mode is not available in the Discord bot’s `/query` command due to its option choices but can be used via CLI or web interface.
