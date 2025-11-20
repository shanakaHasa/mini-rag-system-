# mini-rag-system

A minimal Retrieval-Augmented Generation (RAG) system demonstrating both OpenAI and Hugging Face embeddings, hybrid semantic search, and context-based QA with a vector database (ChromaDB).

---

## Project Structure

- `main.py`: Command-line entry point that lets you query an existing ChromaDB knowledge base (see instructions below).
- `requirements.txt`: All Python dependencies needed for this project.
- `data/wiki_movie_plots_deduped.csv`: Example dataset (movie plots) used for vector indexing and QA.
- `vectorstore/`: Vector database storage for experiment and main workflow.
- `experiments/`: Contains Jupyter notebooks demonstrating OpenAI and Hugging Face embedding setups.

---

## Experiment Notebooks

### `experiments/openai_chroma.ipynb`
Step-by-step workflow to:
- Preprocess and chunk text from the movie plots dataset.
- Generate embeddings for each chunk using the OpenAI API.
- Store and query those embeddings with ChromaDB (vector database).
- Run simple retrieval and answer generation using OpenAI models.

### `experiments/hybrod_search_chroma_hf.ipynb`
Workflow to:
- Preprocess and chunk text as above.
- Generate embeddings with a local Hugging Face `sentence-transformers` model.
- Store and search with ChromaDB, using both vector and metadata search.
- Compose context windows and generate structured answers.

---

## Main CLI Utility (main.py)
- Query the vector store with a natural language question.
- Retrieves the most relevant chunks from the movie database.
- Uses an LLM to generate a JSON-formatted answer with supporting context.
- Run the chatbot in your terminal interactively.

---

## Getting Started

### 1. Install dependencies
```sh
pip install -r requirements.txt
```

### 2. Set your OpenAI API key
Create a `.env` file in the project root with:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Start the command-line QA chat tool
Run in your terminal:
```sh
python main.py --chat
```
(Optional: use `--db-path` and `--collection` to override defaults for your vector database and collection name.)


### 4. Run the notebooks(optional)
Open `experiments/openai_chroma.ipynb` or `experiments/hybrod_search_chroma_hf.ipynb` with Jupyter/VS Code. Execute the cells to see end-to-end RAG flows (embedding, storing, retrieval, answering).
---

## Notes
- The `data/wiki_movie_plots_deduped.csv` data file must exist for all workflows.
- The `.env` file and OpenAI key are required for OpenAI-powered embedding/QA.
- Extend the project by adding scripts, API endpoints, or tests as your use case grows.
