# scripts/ingest.py
from src.loader import load_all
from src.vectorstore import add_documents, count, retrieve

BATCH_SIZE = 256


def main():
    docs = load_all()
    print(f"Loaded {len(docs)} documents")

    for i in range(0, len(docs), BATCH_SIZE):
        add_documents(docs[i : i + BATCH_SIZE])
        print(f"  ingested {min(i + BATCH_SIZE, len(docs))}/{len(docs)}")

    print(f"Collection now holds {count()} documents")

    # Sanity check: distances for a query we know is in the data vs. one that isn't
    for query in ["greatest common divisor of two integers", "kruskal minimum spanning tree"]:
        print(f"\n{query!r}")
        for doc, dist in retrieve(query, k=3):
            print(f"  {dist:.3f}  {doc.metadata['task_id']}")


if __name__ == "__main__":
    main()