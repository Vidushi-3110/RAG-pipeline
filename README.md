# RAG Pipeline: PDF Question Answering System

A Retrieval-Augmented Generation (RAG) pipeline that answers natural language questions using the content of your own PDF documents — instead of relying on an LLM's pretrained knowledge, it retrieves relevant context from your files and generates grounded answers from that context.

## How it works

```
PDF documents → chunking → embeddings → vector store (ChromaDB) → semantic retrieval → LLM-generated answer
```

1. **Ingestion** — PDFs are loaded and split into overlapping text chunks using `RecursiveCharacterTextSplitter`, so retrieval stays precise without losing context across boundaries.
2. **Embedding** — Each chunk is converted into a dense vector using `SentenceTransformers` (`all-MiniLM-L6-v2`), capturing its semantic meaning.
3. **Vector Store** — Embeddings are persisted in a **ChromaDB** collection for fast similarity search.
4. **Retrieval** — A user query is embedded the same way, and the top-k most similar chunks are retrieved via cosine similarity.
5. **Generation** — Retrieved chunks are passed as context into a prompt, and an LLM (via **Groq**) generates a final, grounded answer with cited sources.

## Tech Stack

| Component | Tool |
|---|---|
| Orchestration | LangChain |
| PDF Loading | PyPDFLoader |
| Chunking | RecursiveCharacterTextSplitter |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Vector Database | ChromaDB |
| LLM Inference | Groq (`llama-3.1-8b-instant`) |

## Project Structure

```
RAG_pipeline.ipynb   # Full pipeline: ingestion, embedding, vector store, retrieval, generation
data/pdfs/            # Source PDF documents (add your own)
data/vector_store/    # Persisted ChromaDB collection (auto-generated)
```

## Running it

1. Open the notebook in Google Colab (or locally with Jupyter).
2. Install dependencies (first cell handles this).
3. Add your PDF files to `data/pdfs/`.
4. Get a free API key from [Groq Console](https://console.groq.com/keys).
5. Run all cells top to bottom — you'll be prompted to enter your Groq key securely at runtime (it is never stored in the notebook).
6. Ask a question at the bottom cell and get a context-grounded answer with source references.

## Example

**Query:** `"What is encoder decoder"`

The pipeline retrieves the most relevant chunks from the ingested PDFs and generates a natural-language answer, along with the source documents used to construct it.

## Key Design Choices

- **Chunk size (500) with overlap (50):** balances retrieval precision against context loss at chunk boundaries.
- **Cosine similarity:** standard choice for comparing text embeddings, robust to differences in text length.
- **Groq for inference:** fast, free-tier LLM API — easily swappable for OpenAI, Gemini, or a local model.

## Possible Extensions

- Add re-ranking of retrieved chunks before generation
- Support multi-turn conversational queries with chat history
- Add evaluation metrics (retrieval precision@k, answer faithfulness)
- Swap ChromaDB for FAISS/Pinecone for larger-scale deployments
