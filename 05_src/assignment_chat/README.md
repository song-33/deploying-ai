# The AI Mixologist Chatbot

## Overview

The AI Mixologist is a conversational agent built with LangGraph, LangChain, and Gradio. It is designed to act as an expert, friendly bartender. Users can interact with the chatbot to:

* Discover new cocktail recipes;
* Search for drinks based on specific flavor profiles;
* Learn about the historical origins of various drinks.



## Features and Services

This chatbot utilizes an LLM (gpt-4o-mini) equipped with three specific tools to answer user queries:

1. **Random Cocktail Generator (API)**: Uses the public '[TheCocktailDB](https://www.thecocktaildb.com/api.php)' API to fetch a random cocktail recipe, including glassware, ingredients, and instructions. This is triggered when users ask for a surprise or don't know what to order.
2. **Flavor Profile Search (Semantic query)**: Uses a local ChromaDB vector database to perform semantic searches over a dataset of cocktails adapted from [cocktails data](https://www.kaggle.com/datasets/banajitrajbongshi/cocktails-data) on Kaggle. This allows users to find drinks using natural language descriptions (e.g., "a sweet and sour rum drink").
3. **Cocktail History (Web search)**: Uses the DuckDuckGo Search to look up historical facts, trivia, and origins of specific cocktails, returning top results with snippets and source links.



## Guardrails

The system prompt includes strict guardrails that prevent the model from:

* Revealing or modifying the system prompt;
* Discussing topics unrelated to mixology;
* Discussing restricted topics: cats or dogs, horoscopes or zodiac signs, and Taylor Swift.



## Setup

* Ensure the deploying-ai virtual environment is properly installed.
* One additional library is required for Service 3, install the DDGS library by running: 

```bash
uv add ddgs
```

* If the chroma_db folder is not present in the documents/ folder, generate it by running:

```bash
python ingest.py
```

* To launch the chatbot, run:

```bash
python app.py
```



## Implementation Decisions

* **Embedding process**: The ingest.py script reads from a CSV file, converts it into a readable text format, and creates a JSON file from it. The text is chunked (chunk size = 3000, overlap = 300) to ensure the recipes are kept intact. To ensure a reliable ingestion pipeline and bypass previous 502 Bad Gateway errors encountered with remote API calls, the embeddings are generated locally using ChromaDB's default function rather than relying on OpenAI's endpoints. These documents are stored locally with file persistence. 
* **Agent architecture**: I utilized LangGraph to define a StateGraph that manages the conversation flow between the LLM and the tools. This handles short-term memory by appending new user and tool messages to the State dictionary as the conversation progresses.



## Folder Structure

05_src/assignment_chat/
├── 05_src/
│   └── assignment_chat/
├── agent.py
├── app.py
├── documents/
│   ├── chroma_db/
│   │   ├── 19e0f077-4cd5-4149-afdf-b7c13563b272/
│   │   │   ├── data_level0.bin
│   │   │   ├── header.bin
│   │   │   ├── length.bin
│   │   │   └── link_lists.bin
│   │   └── chroma.sqlite3
│   ├── cocktails.json
│   └── final_cocktails.csv
├── ingest.py
├── README.md
└── tools.py


* agent.py: Contains the LangGraph state compilation, system prompt, and LLM initialization.
* app.py: The Gradio interface. Run this file to start the chat.
* documents/: Contains the raw 'final_cocktails.csv', the json file adapted from the csv file, and the generated 'chroma_db' folder.
* ingest.py: A utility script used to chunk and embed the cocktail dataset into the ChromaDB instance.
* README.md: This file.
* tools.py: Contains the logic for the three tools (API, Semantic search, and Web search).

