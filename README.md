# Retrieval-Augmented Generation over PDFs

A retrieval-augmented generation (RAG) pipeline built step by step in a notebook. It loads PDF and text documents, chunks them, embeds the chunks into a persistent vector store, retrieves the most relevant chunks for a query and passes them as context to an LLM.

## Pipeline

| Stage | Implementation |
|---|---|
| Loading | LangChain `TextLoader` and `PyPDFLoader` over every PDF in `data/pdf/` |
| Chunking | `RecursiveCharacterTextSplitter`, 500-character chunks with 50-character overlap |
| Embedding | `sentence-transformers` `all-MiniLM-L6-v2` (384-dimensional), wrapped in an `EmbeddingManager` class |
| Vector store | ChromaDB `PersistentClient` (`data/vector_store/`), wrapped in a `VectorStoreManager` class with per-chunk metadata |
| Retrieval | `RAGRetriever`: embeds the query, returns the top-k chunks with distance and score |
| Generation | Groq-hosted LLM (`openai/gpt-oss-120b`) via `langchain-groq`, prompted with the retrieved context |

## Files

- `RAG_pipeline.ipynb`: the full pipeline, from loading to the LLM answer.
- `RAG_pipeline2.ipynb`: a second pass through ingestion and retrieval, without the generation step.
- `data/Python.txt`: a small text document used to test the loader.

## Running

```bash
pip install -r requirements.txt
```

1. Put the PDFs you want to query in `data/pdf/`. The original run used research papers such as *Attention Is All You Need*. These are not redistributed here.
2. Create a free API key at [console.groq.com](https://console.groq.com) and paste it into the `API_Key_GROQ` cell. Do not commit it; `.env` is git-ignored if you prefer loading it from the environment.
3. Run the notebook top to bottom. The vector store is created in `data/vector_store/` on first run.

## Known limitation

The Chroma collection uses Chroma's default L2 distance, so `similarity_score = 1 - distance` is not a cosine similarity. Creating the collection with `metadata={"hnsw:space": "cosine"}` would make the score meaningful.
