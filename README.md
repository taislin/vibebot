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
    -   [CLI](#cli)
        -   [Commands](#commands)
    -   [Webserver](#webserver)
        -   [Setup](#setup)
        -   [Features](#features)
    -   [Notes](#notes)

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
    pip install discord-py-interactions interactions.py llama-index llama-index-embeddings-huggingface llama-index-llms-groq numpy python-dotenv PyYAML torch sentence-transformers pynacl setuptools tree-sitter tree-sitter-python tree-sitter-c-sharp sentence_transformers loguru requests
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
    -   Example: `/query show me which animal entities are present in the code`
    -   Example (learning): `/query learn: new animal entity data`

### `/updatedb`

-   **Description**: Updates the knowledge base by re-indexing files in `data/`.

## CLI

The command-line interface (`cli.py`) allows direct interaction with the knowledge base outside Discord.

### Commands

-   **Query the knowledge base**:

    ```bash
    python cli.py query "your query here" --mode docs
    ```

    Example:

    ```bash
    python cli.py query "show me which animal entities are present in the code" --mode docs
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

-   **Query Input**: Enter queries in a text field (e.g., "show me which animal entities are present in the code").
-   **Scrollable Chat History**: View all queries and responses in a dark-themed, scrollable container.
-   **Responsive Design**: Built with Tailwind CSS for a clean, dark-mode interface.

## Notes

-   Ensure sufficient system resources for indexing large datasets (recommended: >4GB RAM or GPU for faster embeddings).
-   Logs are stored in `bot.log` for debugging.
-   For production, secure the web server with authentication and HTTPS.
