# PDF-Based RAG Question Answering System

A Streamlit application that accepts PDFs, extracts text page-by-page, retrieves semantically relevant passages with FAISS, and produces grounded answers through Groq or Mistral.

## Features

- Multiple PDF upload with PyMuPDF validation and empty-PDF handling.
- Page-aware chunking with adjustable size/overlap.
- `all-MiniLM-L6-v2` Sentence Transformer embeddings and FAISS cosine-similarity search.
- Configurable top-k retrieval and visible document/page source passages.
- Groq or Mistral generation, with a prompt that refuses to answer outside retrieved context.
- Streamlit session chat history and clear-index control.

## Architecture

`PDF upload -> PyMuPDF page extraction -> recursive chunks + metadata -> SentenceTransformer embeddings -> FAISS -> top-k contexts -> Groq/Mistral -> answer + citations`

Each chunk stores `document`, `page`, `chunk`, and `text`. The UI displays the retained document name and page for every retrieved source. The model prompt explicitly prohibits fabricated answers and citations.

## Setup

```bash
git clone <your-repository-url>
cd pdf-rag-assistant
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
```

Put your own key in `.env`. For Groq, keep `LLM_PROVIDER=groq` and set `GROQ_API_KEY`. For Mistral, set `LLM_PROVIDER=mistral`, `MISTRAL_API_KEY`, and a compatible `MODEL_NAME`.

```bash
streamlit run app.py
```

## Testing checklist

1. Start the app and process all three PDFs in `sample_pdfs/`.
2. Ask: `What is semantic search?` Expected: a cited answer from `rag_overview.pdf`.
3. Ask: `Which metric compares retrieved passages?` Expected: a cited answer from `evaluation_notes.pdf`.
4. Ask: `Who won the 2022 World Cup?` Expected: the unavailable-answer response, not a fabricated answer.
5. Upload a renamed text file ending in `.pdf`; it should be skipped with an error.
6. Change chunk size from 1000/150 to 600/100 and compare the source passages. This satisfies the two-configuration experiment.

## Demo guide (3-5 minutes)

1. Introduce the problem: an LLM alone cannot reliably answer from private PDFs.
2. Upload the sample PDFs and show page-aware processing.
3. Explain chunking, embeddings, and FAISS retrieval.
4. Ask an in-document question; expand the cited passages.
5. Ask an out-of-document question to demonstrate grounding.
6. Change chunk settings, explain the trade-off, and show the `.env` configuration.

## Deployment

Push this folder to GitHub, then create a Streamlit Community Cloud app pointed at `app.py`. In the deployment secrets, add the same variables from `.env` (never commit the key). Alternatively run it locally with the command above.

## Notes for evaluation

The application makes no hardcoded domain answers. FAISS returns the evidence, and the LLM receives only retrieved excerpts plus the question. A production version could add a relevance-score threshold and persistent vector storage.
