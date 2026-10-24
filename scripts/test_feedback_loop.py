# scripts/test_feedback_loop.py
from src.router import SessionState, handle_message
from src.vectorstore import count, delete_documents

REQUEST = "implement Kruskal's minimum spanning tree algorithm"
SOLUTION = """Here is my implementation of Kruskal's algorithm using union-find:

def kruskal(n, edges):
    # edges are (weight, u, v) tuples; nodes are 0..n-1
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    mst, total = [], 0
    for w, u, v in sorted(edges):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            mst.append((u, v, w))
            total += w
    return mst, total
"""

failures = 0


def check(name, ok, detail=""):
    global failures
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    failures += 0 if ok else 1


before = count()
stored_id = None
try:
    state = SessionState()

    print("1. Ask for something the knowledge base lacks")
    r1 = handle_message(REQUEST, state)
    check("route is needs_solution", r1.route == "needs_solution", f"(got {r1.route})")
    check("waiting flag is set", state.awaiting_solution)

    print("2. Teach it")
    r2 = handle_message(SOLUTION, state)
    check("route is learned", r2.route == "learned", f"(got {r2.route})")
    check("waiting flag cleared", not state.awaiting_solution)
    stored_id = r2.sources[0] if r2.sources else None
    check("document count grew by 1", count() == before + 1, f"({before} -> {count()})")

    print("3. Ask again in a fresh session")
    r3 = handle_message(REQUEST, SessionState())
    check("route is generate", r3.route == "generate", f"(got {r3.route})")
    check("taught solution was retrieved", stored_id in r3.sources, f"(sources {r3.sources})")
finally:
    if stored_id:
        delete_documents([stored_id])
    check("cleanup restored the count", count() == before, f"({count()} vs {before})")

print("\nALL PASSED" if failures == 0 else f"\n{failures} CHECK(S) FAILED")