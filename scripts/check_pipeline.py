# scripts/check_pipeline.py
import sys

from src.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME
from src.classifier import classify_intent
from src.LLM import get_llm
from src.vectorstore import count, retrieve

EXPECTED_COUNT = 1138
failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failures
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures += 1


# ---------- 1. Config ----------
print("1. CONFIG")
check("API key loaded", bool(LLM_API_KEY))
check("model name loaded", bool(LLM_MODEL_NAME), f"({LLM_MODEL_NAME})")
check("base URL loaded", bool(LLM_BASE_URL), f"({LLM_BASE_URL})")

# ---------- 2. LLM connectivity ----------
print("\n2. LLM CONNECTIVITY")
try:
    reply = get_llm().invoke("Reply with exactly one word: OK").content
    check("OpenRouter call succeeded", bool(reply.strip()), f"(reply: {reply.strip()[:30]!r})")
except Exception as e:
    check("OpenRouter call succeeded", False, f"({type(e).__name__}: {str(e)[:120]})")

# ---------- 3. Vector store + retrieval ----------
print("\n3. VECTOR STORE")
check("document count", count() == EXPECTED_COUNT, f"(found {count()}, expected {EXPECTED_COUNT})")

probes = {
    "check if any two numbers in a list are closer than a threshold": {"HumanEval/0"},
    "greatest common divisor of two integers": {"HumanEval/13"},
    "check whether a string is a palindrome": {"HumanEval/10", "HumanEval/48"},
    "filter a list of strings that contain a given substring": {"HumanEval/7"},
}
for query, expected in probes.items():
    got = [d.metadata["task_id"] for d, _ in retrieve(query, k=5)]
    check(f"retrieves {sorted(expected)}", bool(expected & set(got)), f"<- {query!r}")

# ---------- 4. Classifier ----------
print("\n4. CLASSIFIER")
labeled = [
    ("Explain this code line by line: for i in range(3): print(i)", "explain"),
    ("What does this function do? def f(x): return x * 2", "explain"),
    ("Why is this loop incorrect? while i < 10: print(i)", "explain"),
    ("How does quicksort work?", "explain"),
    ("Write a function to check if a number is prime", "generate"),
    ("Build a Flask API with one GET endpoint", "generate"),
    ("Create a CNN model for MNIST", "generate"),
    ("Generate a sorting algorithm", "generate"),
]
for query, expected in labeled:
    check(f"-> {expected}", classify_intent(query) == expected, f"<- {query!r}")

print("  Ambiguous (info only, no pass/fail):")
for query in [
    "Fix this code: def f(x) return x",
    "Convert this Python function to Java: def f(x): return x",
    "Optimize this function: def f(n): return sum([i for i in range(n)])",
    "Add error handling to this code: def f(x): return 1 / x",
]:
    print(f"     {classify_intent(query):9} <- {query!r}")

# ---------- 5. Distance calibration (for the relevance step) ----------
print("\n5. DISTANCE CALIBRATION (best distance per query, lower = closer)")
likely_covered = [
    "write a function to find the gcd of two numbers",
    "check if a string is a palindrome",
    "filter a list of strings by a substring",
    "write a function to check whether a number is prime",
    "sort a list of numbers in ascending order",
    "find the factorial of a number",
]
not_covered = [
    "implement Kruskal's minimum spanning tree algorithm",
    "implement Dijkstra's shortest path algorithm",
    "breadth first search on a graph",
    "build a Flask REST API with a GET endpoint",
    "train a CNN in PyTorch on MNIST",
    "scrape a website using requests and BeautifulSoup",
    "connect to a PostgreSQL database",
    "create a tkinter GUI with a button",
]


def best_distance(query: str):
    doc, dist = retrieve(query, k=1)[0]
    return dist, doc.metadata["task_id"]


covered = []
print("  Likely covered:")
for q in likely_covered:
    d, tid = best_distance(q)
    covered.append(d)
    print(f"     {d:.3f}  {tid:14} <- {q}")

uncovered = []
print("  Not covered:")
for q in not_covered:
    d, tid = best_distance(q)
    uncovered.append(d)
    print(f"     {d:.3f}  {tid:14} <- {q}")

print(f"\n  Covered   : min {min(covered):.3f}  max {max(covered):.3f}")
print(f"  Uncovered : min {min(uncovered):.3f}  max {max(uncovered):.3f}")
if max(covered) < min(uncovered):
    print(f"  Clean gap between {max(covered):.3f} and {min(uncovered):.3f}")
else:
    print("  Ranges OVERLAP: a distance cutoff alone is not enough, the LLM evaluator must decide")

# ---------- Verdict ----------
print("\n" + ("ALL HARD CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED"))
sys.exit(1 if failures else 0)