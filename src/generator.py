# src/generator.py
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.LLM import get_llm
from src.relevance import format_context

EXPLAIN_SYSTEM = """You are a patient programming tutor.
- If the message contains code, work on that code directly: explain it step by step and point
  out any bugs. If the user asks to fix, optimize, refactor or convert it, show the improved
  code in a fenced code block and say what changed and why.
- If there is no code, explain the concept clearly with a short example.
- Start in plain language, then add technical detail.
- Only write new code when it is a fix, change or conversion of the user's existing code."""

GENERATE_SYSTEM = """You are an expert programmer. Write code for the user's request.
You are given reference snippets from a trusted knowledge base. Use them as guidance for the
approach, but write a complete, self-contained solution.

Requirements:
- Complete, runnable source code in ONE fenced code block (state the language).
- Helpful comments.
- Sensible error handling and input validation where appropriate.
- Follow the language's best practices.
- After the code, add a short explanation (2 to 4 sentences)."""

_explain_prompt = ChatPromptTemplate.from_messages(
    [("system", EXPLAIN_SYSTEM),
     ("human", "Conversation context:\n{memory}\n\nUser message:\n{query}")]
)
_generate_prompt = ChatPromptTemplate.from_messages(
    [("system", GENERATE_SYSTEM),
     ("human", "Conversation context:\n{memory}\n\nReference snippets:\n{context}\n\nRequest:\n{query}")]
)


def explain(query: str, memory: str = "") -> str:
    chain = _explain_prompt | get_llm(temperature=0.2) | StrOutputParser()
    return chain.invoke({"query": query, "memory": memory or "None"})


def generate(query: str, results: List[Tuple[Document, float]], memory: str = "") -> str:
    chain = _generate_prompt | get_llm(temperature=0.2) | StrOutputParser()
    return chain.invoke(
        {"query": query, "context": format_context(results), "memory": memory or "None"}
    )


if __name__ == "__main__":
    from src.relevance import check_relevance
    from src.vectorstore import retrieve

    print("=== EXPLAIN (fix request) ===")
    print(explain("Fix this code: def f(x) return x"))

    q = "write a function to check whether a number is prime"
    results = retrieve(q)
    print("\n=== GENERATE ===  relevant:", check_relevance(q, results))
    print(generate(q, results))