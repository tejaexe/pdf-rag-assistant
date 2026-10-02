from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppSettings:
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq").lower()
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    mistral_api_key: str | None = os.getenv("MISTRAL_API_KEY")
    model_name: str = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "4"))
