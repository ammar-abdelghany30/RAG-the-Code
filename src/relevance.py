# src/relevance.py
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.config import RELEVANCE_MAX_DISTANCE
from src.LLM import get_llm

SYSTEM = """You are a strict relevance judge for a coding assistant.
You get a user's request and reference snippets retrieved from a knowledge base.
Decide whether the snippets contain knowledge that directly helps write the requested code.

Answer yes only if at least one snippet solves the same problem, or a very close
variant that could be adapted with small changes.

Answer no if the snippets only share keywords or a topic with the request
(for example, the request is about X and a snippet solves a different task that
merely mentions X), or if they solve a different problem.

Reply with ONE word: yes or no. No punctuation, no explanation."""

_prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM), ("human", "User request:\n{query}\n\nRetrieved snippets:\n{context}")]
)


def format_context(results: List[Tuple[Document, float]], max_chars: int = 800) -> str:
    parts = []
    for i, (doc, _dist) in enumerate(results, start=1):
        parts.append(f"[{i}] {doc.metadata['task_id']}\n{doc.page_content[:max_chars]}")
    return "\n\n".join(parts)


def check_relevance(query: str, results: List[Tuple[Document, float]]) -> bool:
    """True = Relevant, False = Not Relevant."""
    if not results:
        return False

    # Cheap gate: skip the LLM call when even the best match is very far away.
    best_distance = results[0][1]
    if RELEVANCE_MAX_DISTANCE is not None and best_distance > RELEVANCE_MAX_DISTANCE:
        return False

    chain = _prompt | get_llm(temperature=0.0) | StrOutputParser()
    context = format_context(results)

    raw = ""
    for _ in range(2):  # one retry
        raw = chain.invoke({"query": query, "context": context}).strip().lower()
        word = raw.split()[0].strip(".,:;!\"'`") if raw else ""
        if word in ("yes", "no"):
            return word == "yes"

    print(f"[relevance warning] unrecognized output {raw!r}, treating as Not Relevant")
    return False


if __name__ == "__main__":
    from src.vectorstore import retrieve

    tests = [
        ("greatest common divisor of two integers", True),
        ("implement Kruskal's minimum spanning tree algorithm", False),
        ("find the factorial of a number", None),  # informational: no expected answer
    ]
    for query, expected in tests:
        verdict = check_relevance(query, retrieve(query))
        label = "Relevant" if verdict else "Not Relevant"
        flag = "" if expected is None else ("  [OK]" if verdict == expected else "  [UNEXPECTED]")
        print(f"{label:13} <- {query}{flag}")