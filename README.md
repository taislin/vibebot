# Vibebot

![vibebot](vibebot.gif)

This is a Discord bot utilizes the **[Retrieval-Augmented Generation](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)** (RAG) approach to enhance the output of a large language model by referencing an authoritative knowledge base outside of its training data sources. The bot is designed to interact with users through Discord slash commands, allowing them to query an underlying knowledge base for relevant information.

This was created as a fork of the [LLM-RAG-Bot](https://github.com/nur-zaman/LLM-RAG-Bot) project that uses [Groq](https://groq.com/) instead of OpenAI.

## Prerequisites

Before running the bot, make sure you have the following dependencies installed:

-   Python 3.10 or higher
-   The packages in the requirements.txt. Run `pip install -r requirements.txt` to install them.

## Getting Started

1. Clone the repository to your local machine:

```bash
git clone https://github.com/taislin/vibebot.git
cd LLM-RAG-Bot
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Rename the `.env.example` file to `.env` in the project root directory and add your tokens and api keys:

```env
DISCORD_BOT_TOKEN=12345
GROQ_API_KEY=12345
GROQ_MODEL=gemma2-9b-it
```

4. Add the files your model will be based on to the `storage/` folder. Create it if it does not exist.

## Running the Bot

```bash
python bot.py
```

This will launch the bot, and you should see "Ready" in the console once it has successfully connected to Discord.

## Bot Usage

The bot responds to a single slash command:

### `/query`

-   **Description:** Enter your query
-   **Options:**
    -   `input_text` (required): The input text for the query. If you add "learn:" before your input, the model will learn from it!

### `/updatedb`

-   **Description:** Updates your information database.
