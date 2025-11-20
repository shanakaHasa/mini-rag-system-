# mini-rag-system-
Build a lightweight Retrieval-Augmented Generation (RAG) system that can answer questions about movie plots from a small subset of the Wikipedia Movie Plots dataset.

"""
RAG CLI utility (rag_cli.py)

Features:
- Build a Chroma vectorstore from a CSV of movie plots (or any CSV with Title & Plot columns)
- Persist embeddings (OpenAI text-embedding-3-large) and texts into a Persistent Chroma DB
- Query loop in the terminal: retrieves top-k chunks and asks the LLM to answer using only retrieved contexts

Usage examples:
# Build the index (creates/overwrites the chroma DB at --db-path)
python rag_cli.py --build --csv ../data/wiki_movie_plots_deduped.csv --db-path ./vectorstore/chroma_db_openai

# Run interactive chat against the existing DB
python rag_cli.py --chat --db-path ./vectorstore/chroma_db_openai

Requirements:
pip install openai chromadb numpy python-dotenv pandas

Make sure you have an environment variable OPENAI_API_KEY set (or a .env file with OPENAI_API_KEY=...)

This file is intentionally self-contained and conservative with memory (processes first 500 rows by default)
"""