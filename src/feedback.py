# src/feedback.py
import hashlib

from langchain_core.documents import Document

from src.vectorstore import add_documents


def store_solution(request: str, solution: str, language: str = "unknown") -> str:
    """Embed and store a user-taught solution in Chroma. Returns its ID."""
    # Hash-based ID: sending the same request+solution twice upserts instead of duplicating
    digest = hashlib.sha1((request + solution).encode("utf-8")).hexdigest()[:10]
    task_id = f"USER/{digest}"

    doc = Document(
        page_content=f"# Task: {request}\n{solution}",
        metadata={
            "task_id": task_id,
            "source": "user_feedback",
            "language": language,
            "test": "",  # keeps the same metadata keys as the benchmark documents
        },
    )
    add_documents([doc])
    return task_id