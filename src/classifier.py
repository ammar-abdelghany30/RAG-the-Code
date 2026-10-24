# src/classifier.py
from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.LLM import get_llm

Intent = Literal["explain", "generate"]

SYSTEM = """You classify a user's programming request into exactly one label.

explain  - the user gives existing code (or refers to it) and wants it explained, analyzed,
           debugged, fixed, optimized, refactored, or converted to another language.
generate - the user wants NEW code written from a description, with no code to work on.

Examples:
"Explain this code." -> explain
"What does this function do?" -> explain
"Why is this loop incorrect?" -> explain
"Fix this code: def f(x) return x" -> explain
"Optimize this function: def f(n): return sum([i for i in range(n)])" -> explain
"Convert this Python function to Java: def f(x): return x" -> explain
"Write a function that sorts a list." -> generate
"Build a Flask API." -> generate
"Create a CNN model." -> generate

Reply with ONE word: explain or generate. No punctuation, no extra text."""

_prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", "{query}")])


def classify_intent(query: str) -> Intent:
    chain = _prompt | get_llm(temperature=0.0) | StrOutputParser()
    raw = ""
    for _ in range(2):  
        raw = chain.invoke({"query": query}).strip().lower()
        word = raw.split()[0].strip(".,:;!\"'`") if raw else ""
        if word in ("explain", "generate"):
            return word
    print(f"[classifier warning] unrecognized output {raw!r}, defaulting to generate")
    return "generate"


if __name__ == "__main__":
    tests = [
        "Explain this code line by line: for i in range(3): print(i)",
        "What does this function do? def f(x): return x * 2",
        "Why is this loop incorrect? while i < 10: print(i)",
        "Write a function to check if a number is prime",
        "Build a Flask API with one GET endpoint",
        "How does quicksort work?",
        "Optimize this function: <code>",
        "Convert this Python function to Java: def f(x): return x * 2",
        "Fix this code: def f(x) return x",  # ambiguous on purpose
    ]
    for q in tests:
        print(f"{classify_intent(q):9} <- {q}")