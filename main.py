import os
import argparse
import json
import sys
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import chromadb
from chromadb.config import Settings

# load env
load_dotenv()
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("ERROR: OPENAI_API_KEY not set. Put it in your environment or .env file.")
    sys.exit(1)


def get_embedding(text, model="text-embedding-3-large"):
    # uses the same call style you used earlier
    resp = openai.embeddings.create(model=model, input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)

def answer_with_existing_db(db_path, query, top_k=5, collection_name="movie_plots"):
    collection = load_collection(db_path, collection_name=collection_name)
    docs, metadatas = retrieve_chunks(collection, query, top_k=top_k)
    if not docs:
        return {"answer": "No relevant information was found in the knowledge base.", "contexts": [], "reasoning": "No chunks returned."}
    return generate_answer_with_context(query, docs, metadatas)



# -------- load collection & retrieval --------

def load_collection(db_path, collection_name="movie_plots"):
    try:
        client = chromadb.PersistentClient(path=db_path)
    except Exception:
        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=db_path))

    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        collection = client.get_or_create_collection(name=collection_name)
    return collection


def retrieve_chunks(collection, query, top_k=5, model="text-embedding-3-large"):
    query_embedding = get_embedding(query, model=model)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    return docs, metadatas


# -------- LLM answer generation --------

def generate_answer_with_context(query, docs, metadatas, system_message=None):
    combined_context = "\n\n".join([f"Title: {m.get('title','')}\nText: {d}" for d, m in zip(docs, metadatas)])

    prompt = f"""
            You are a RAG assistant. Answer the user's question using ONLY the provided movie plot chunks.
            Return ONLY valid JSON in this exact format:
            {{
            "answer": "...",
            "contexts": ["..."],
            "reasoning": "..."
            }}
            If no chunk is relevant:
            - answer: "No relevant information was found in the knowledge base."
            - contexts: []
            - reasoning: "No chunk matches the user query."

            User Query: {query}

            Movie Plot Chunks:
            {combined_context}
            """

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    resp = openai.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0)
    text = resp.choices[0].message.content
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "JSON parse failed", "raw": text}


# -------- CLI / Interactive Loop --------

def interactive_chat(collection):
    print("RAG CLI — type your question, or 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.strip().lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        docs, metadatas = retrieve_chunks(collection, query)
        if not docs:
            print("No documents retrieved from the vectorstore.")
            continue

        answer = generate_answer_with_context(query, docs, metadatas)
        print("\nAssistant (RAG):")
        print(json.dumps(answer, indent=2, ensure_ascii=False))


# -------- main --------

def main():
    parser = argparse.ArgumentParser(description="RAG CLI utility")

    parser.add_argument("--chat", action="store_true", help="Start interactive chat against existing Chroma DB")
    parser.add_argument("--db-path", type=str, default="vectorstore/chroma_db_openai", help="Path where the chroma DB will be persisted")
    parser.add_argument("--collection", type=str, default="movie_plots", help="Chroma collection name")
    args = parser.parse_args()


    if args.chat:
        collection = load_collection(args.db_path, collection_name=args.collection)
        interactive_chat(collection)



if __name__ == "__main__":
    main()
