from collections import Counter

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.loader import load_all

REQUIRED_KEYS = {"task_id", "source", "language", "test"}
SIMPLE_TYPES = (str, int, float, bool)  # Chroma metadata limitation


def check_structure(docs):
    assert all(isinstance(d, Document) for d in docs), "non-Document found"
    assert all(d.page_content.strip() for d in docs), "empty page_content"

    ids = [d.metadata["task_id"] for d in docs]
    dupes = [i for i, c in Counter(ids).items() if c > 1]
    assert not dupes, f"duplicate IDs: {dupes[:5]}"

    for d in docs:
        missing = REQUIRED_KEYS - d.metadata.keys()
        assert not missing, f"{d.metadata['task_id']} missing {missing}"
        bad = [k for k, v in d.metadata.items() if not isinstance(v, SIMPLE_TYPES)]
        assert not bad, f"{d.metadata['task_id']} non-simple metadata: {bad}"

    counts = Counter(d.metadata["source"] for d in docs)
    assert counts["humaneval"] == 164, f"HumanEval count is {counts['humaneval']}"
    print("STRUCTURE OK:", dict(counts), "total", len(docs))


def keyword_scan(docs):
    print("\nKEYWORD SCAN (does the data contain it at all?)")
    for term in ["kruskal", "spanning tree", "factorial", "fibonacci", "binary search", "bfs", "dijkstra"]:
        hits = [d.metadata["task_id"] for d in docs if term in d.page_content.lower()]
        print(f"  {term:15} {len(hits):3} hits  {hits[:4]}")


def retrieval_smoke_test(docs):
    print("\nRETRIEVAL SMOKE TEST (embedding all docs, may take a minute)")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = Chroma.from_documents(
        docs,
        embeddings,
        ids=[d.metadata["task_id"] for d in docs],
        collection_name="validate_tmp",  # in-memory, nothing saved to disk
    )

    probes = {
        "check if any two numbers in a list are closer than a threshold": {"HumanEval/0"},
        "greatest common divisor of two integers": {"HumanEval/13"},
        "check whether a string is a palindrome": {"HumanEval/10", "HumanEval/48"},
    }
    for query, expected in probes.items():
        got = [d.metadata["task_id"] for d in store.similarity_search(query, k=5)]
        status = "PASS" if expected & set(got) else "MISS"
        print(f"  [{status}] {query!r} -> {got}")

    print("\nOPEN QUESTIONS (no expected answer, just look):")
    for query in ["kruskal minimum spanning tree", "factorial of a number"]:
        print(f"  {query!r}")
        for d in store.similarity_search(query, k=3):
            first_line = d.page_content.strip().splitlines()[0][:70]
            print(f"     {d.metadata['task_id']:14} {first_line}")


if __name__ == "__main__":
    docs = load_all()
    check_structure(docs)
    keyword_scan(docs)
    retrieval_smoke_test(docs)