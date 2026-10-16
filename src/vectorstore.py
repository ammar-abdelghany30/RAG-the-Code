# src/vectorstore.py
from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, TOP_K

_embeddings = None
_store = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once, then reuse it."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def get_vectorstore() -> Chroma:
    """Open the persistent Chroma collection (creates it on first use)."""
    global _store
    if _store is None:
        _store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _store


def add_documents(docs: List[Document]) -> None:
    """Add documents, using task_id as the Chroma ID."""
    ids = [d.metadata["task_id"] for d in docs]
    get_vectorstore().add_documents(docs, ids=ids)


def retrieve(query: str, k: int = TOP_K) -> List[Tuple[Document, float]]:
    """Top-k most similar documents, each with its cosine distance (lower = closer)."""
    return get_vectorstore().similarity_search_with_score(query, k=k)


def count() -> int:
    return get_vectorstore()._collection.count()