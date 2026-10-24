# src/router.py
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.LLM import get_llm
from src.classifier import classify_intent
from src.feedback import store_solution
from src.generator import explain, generate
from src.relevance import check_relevance
from src.vectorstore import retrieve

NOT_FOUND_MESSAGE = (
    "I couldn't find relevant knowledge for your request. Could you provide the "
    "correct solution or a reference so I can learn it for future interactions?"
)
LEARNED_MESSAGE = "Thanks! I've saved your solution and will use it for similar requests from now on."

JUDGE_SYSTEM = """A coding assistant could not find knowledge for a user's request, so it asked the user
to provide a solution or reference. You get the original request and the user's next message.

Answer yes if the message provides a solution, implementation, code, or a reference for the original request.
Answer no if it is a new or different question, or unrelated to the original request.

Reply with ONE word: yes or no."""

_judge_prompt = ChatPromptTemplate.from_messages(
    [("system", JUDGE_SYSTEM),
     ("human", "Original request:\n{request}\n\nUser's next message:\n{message}")]
)


@dataclass
class SessionState:
    awaiting_solution: bool = False
    pending_request: Optional[str] = None


@dataclass
class RouterResult:
    route: str  # "explain" | "generate" | "needs_solution" | "learned"
    answer: str
    sources: List[str] = field(default_factory=list)
    needs_solution: bool = False


def provides_solution(request: str, message: str) -> bool:
    chain = _judge_prompt | get_llm(temperature=0.0) | StrOutputParser()
    raw = ""
    for _ in range(2):  # one retry
        raw = chain.invoke({"request": request, "message": message}).strip().lower()
        word = raw.split()[0].strip(".,:;!\"'`") if raw else ""
        if word in ("yes", "no"):
            return word == "yes"
    print(f"[router warning] unrecognized judge output {raw!r}, treating as a new question")
    return False


def handle_message(message: str, state: SessionState, memory: str = "") -> RouterResult:
    # 1. Are we waiting for the user to teach us something?
    if state.awaiting_solution:
        pending = state.pending_request
        state.awaiting_solution = False
        state.pending_request = None
        if provides_solution(pending, message):
            doc_id = store_solution(pending, message)
            return RouterResult("learned", LEARNED_MESSAGE, [doc_id])
        # otherwise: it's a new question, fall through to normal routing

    # 2. Normal routing
    if classify_intent(message) == "explain":
        return RouterResult("explain", explain(message, memory))

    results = retrieve(message)
    sources = [d.metadata["task_id"] for d, _ in results]

    if check_relevance(message, results):
        return RouterResult("generate", generate(message, results, memory), sources)

    state.awaiting_solution = True
    state.pending_request = message
    return RouterResult("needs_solution", NOT_FOUND_MESSAGE, sources, needs_solution=True)


if __name__ == "__main__":
    # Single-line chat only (pasting multi-line code into input() sends one line at a time)
    state = SessionState()
    while True:
        text = input("you> ").strip()
        if text in ("", "exit"):
            break
        r = handle_message(text, state)
        print(f"[{r.route}] {r.answer}\n")