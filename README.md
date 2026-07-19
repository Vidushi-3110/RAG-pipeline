# RAG Pipeline: PDF Question Answering System

A Retrieval-Augmented Generation (RAG) pipeline built from scratch that lets you ask
questions about the content of your own PDF documents and get grounded, context-aware
answers from an LLM.

## How It Works

```
PDFs → Documents → Chunks → Embeddings → Vector Store (ChromaDB) → Retrieval → LLM Answer
```

1. **Load** — every PDF in `data/pdfs/` is parsed page-by-page into LangChain `Document` objects.
2. **Split** — pages are broken into smaller, overlapping text chunks so each embedding stays focused on one idea.
3. **Embed** — each chunk is converted into a numeric vector using a `sentence-transformers` model (`all-MiniLM-L6-v2`).
4. **Store** — chunks and their embeddings are saved in a persistent **ChromaDB** collection.
5. **Retrieve** — a user's question is embedded the same way, and ChromaDB returns the most semantically similar chunks.
6. **Generate** — the retrieved chunks are injected into a prompt as context, and an LLM (via **Groq**) generates the final answer.

## Tech Stack

| Component        | Library                     |
|-------------------|------------------------------|
| PDF loading        | `langchain-community` (PyPDFLoader) |
| Text splitting      | `langchain-text-splitters`        |
| Embeddings          | `sentence-transformers`           |
| Vector database     | `chromadb`                        |
| LLM inference        | `langchain-groq`                  |

## Project Structure

```
├── RAG_pipeline_scratch.ipynb   # main notebook — run top to bottom
├── data/
│   ├── pdfs/                    # put your source PDFs here
│   └── vector_store/            # ChromaDB persists embeddings here (auto-created)
└── README.md
```

## Setup & Usage

1. **Open the notebook** in Google Colab or Jupyter.
2. **Install dependencies** — the first code cell runs:
   ```bash
   pip install langchain langchain-core langchain-community langchain-groq pypdf pymupdf sentence-transformers chromadb -q
   ```
3. **Add your PDF(s)** into `data/pdfs/` (create the folder first if needed — the notebook does this automatically). Make sure the PDF sits directly inside `data/pdfs/`, not in `data/`.
4. **Get a free Groq API key** from [console.groq.com/keys](https://console.groq.com/keys) — needed for the answer-generation step. Copy the key somewhere safe when it's created, since Groq only shows it once.
5. **Run all cells top to bottom** (Runtime → Run all in Colab). When prompted, paste your Groq API key.
6. Ask a question by calling:
   ```python
   result = rag_pipeline.answer("your question here")
   print(result["answer"])
   ```

## Key Classes

- **`EmbeddingManager`** — loads the sentence-transformer model and turns text into embedding vectors.
- **`VectorStoreManager`** — wraps a persistent ChromaDB collection for storing and querying embeddings.
- **`RAGRetriever`** — embeds a query and retrieves the top-k most relevant chunks from the vector store.
- **`RAGPipeline`** — combines retrieval with LLM generation to produce a final, context-grounded answer.

## Troubleshooting

- **`ModuleNotFoundError`** — the install cell didn't finish running, or the kernel needs a restart after installing new packages. Re-run the `pip install` cell, then restart the runtime/kernel and run all cells again in order.
- **`PDFs loaded: 0`** — your PDF isn't inside `data/pdfs/`. Check the file path in the Colab/Jupyter file browser; the PDF must be directly inside the `pdfs` subfolder.
- **Empty embeddings / `ValueError` in ChromaDB's `add()`** — this means no PDF pages were loaded. Re-run `load_all_pdfs()` and `split_into_chunks()` after fixing the PDF location, before re-running the ingestion cell.
- **Stuck on "Enter your Groq API key"** — this isn't an error; paste your key into the box and press Enter. Typing stays hidden for security.

## Notes

- This project was built for learning purposes to understand each stage of a RAG pipeline hands-on.
- Swap in OpenAI, Gemini, or a local model in place of Groq by replacing the `ChatGroq` call inside `RAGPipeline`.
