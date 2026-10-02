from dataclasses import dataclass
from typing import Any
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src.config import AppSettings
from src.pdf_loader import extract_pages

SYSTEM_PROMPT = """You answer questions only from the supplied PDF excerpts. Be concise and factual.
If the excerpts do not contain the answer, say: 'I could not find that in the uploaded documents.'
Do not invent facts, sources, or page numbers. Cite claims inline as [Document, p. N]."""


@dataclass
class IngestResult:
    documents: int = 0
    chunks: int = 0
    skipped: list[str] | None = None


@dataclass
class Answer:
    text: str
    sources: list[dict[str, Any]]


class RAGService:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.embedder = SentenceTransformer(settings.embedding_model)
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[dict[str, Any]] = []

    def clear(self) -> None:
        self.index, self.chunks = None, []

    def ingest(self, uploads, chunk_size: int, chunk_overlap: int) -> IngestResult:
        self.clear()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        skipped: list[str] = []
        document_count = 0
        for upload in uploads:
            try:
                pages = extract_pages(upload.name, upload.getvalue())
            except ValueError as exc:
                skipped.append(str(exc))
                continue
            document_count += 1
            for page in pages:
                for position, text in enumerate(splitter.split_text(page.text)):
                    self.chunks.append({"document": page.document, "page": page.page,
                                        "chunk": position, "text": text})
        if self.chunks:
            vectors = self._embed([item["text"] for item in self.chunks])
            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)
        return IngestResult(document_count, len(self.chunks), skipped)

    def answer(self, question: str, history: list[dict], top_k: int) -> Answer:
        if self.index is None:
            return Answer("Please upload and process at least one readable PDF first.", [])
        sources = self._retrieve(question, top_k)
        context = "\n\n".join(f"[{s['document']}, p. {s['page']}]\n{s['text']}" for s in sources)
        recent_history = "\n".join(f"{m['role'].title()}: {m['content']}" for m in history[-6:])
        prompt = f"{SYSTEM_PROMPT}\n\nConversation:\n{recent_history}\n\nPDF excerpts:\n{context}\n\nQuestion: {question}"
        try:
            text = self._generate(prompt)
        except ValueError as exc:
            return Answer(str(exc), self._source_cards(sources))
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code in (401, 403):
                message = "Your LLM API key was rejected. Check the selected provider and replace its key in `.env`, then restart Streamlit."
            elif status_code == 429:
                message = "Groq rate limit or account quota reached. Wait briefly, check your Groq account limits, then try again."
            elif status_code == 404:
                available = self._available_groq_models()
                suffix = (" Models available to this key: " + ", ".join(available[:12]) + ".") if available else ""
                message = f"The configured model '{self.settings.model_name}' is unavailable to this Groq account. Choose an available model in `.env` and restart Streamlit.{suffix}"
            else:
                detail = str(exc).replace("\n", " ")[:300]
                message = f"The LLM provider could not complete this request ({detail}). Check your internet connection, API key, model name, and provider status, then try again."
            return Answer(message, self._source_cards(sources))
        return Answer(text, self._source_cards(sources))

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")

    def _retrieve(self, question: str, top_k: int) -> list[dict]:
        _, ids = self.index.search(self._embed([question]), min(top_k, len(self.chunks)))
        return [self.chunks[index] for index in ids[0] if index >= 0]

    @staticmethod
    def _source_cards(sources: list[dict]) -> list[dict]:
        return [{"document": s["document"], "page": s["page"],
                 "excerpt": s["text"][:260].replace("\n", " ") + ("..." if len(s["text"]) > 260 else "")}
                for s in sources]

    def _generate(self, prompt: str) -> str:
        if self.settings.llm_provider == "groq":
            if not self.settings.groq_api_key:
                raise ValueError("GROQ_API_KEY is missing. Add it to `.env`, then restart Streamlit.")
            from groq import Groq
            response = Groq(api_key=self.settings.groq_api_key).chat.completions.create(
                model=self.settings.model_name, messages=[{"role": "user", "content": prompt}], temperature=0.1)
            return response.choices[0].message.content
        if self.settings.llm_provider == "mistral":
            if not self.settings.mistral_api_key:
                raise ValueError("MISTRAL_API_KEY is missing. Add it to `.env`, then restart Streamlit.")
            from mistralai import Mistral
            response = Mistral(api_key=self.settings.mistral_api_key).chat.complete(
                model=self.settings.model_name, messages=[{"role": "user", "content": prompt}])
            return response.choices[0].message.content
        raise ValueError("LLM_PROVIDER must be either 'groq' or 'mistral'.")

    def _available_groq_models(self) -> list[str]:
        """Return model IDs visible to the current Groq key without exposing the key."""
        if self.settings.llm_provider != "groq" or not self.settings.groq_api_key:
            return []
        try:
            from groq import Groq
            response = Groq(api_key=self.settings.groq_api_key).models.list()
            return sorted(model.id for model in response.data)
        except Exception:
            return []
