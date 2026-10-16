# src/loader.py
from typing import List
from datasets import load_dataset
from langchain_core.documents import Document

HUMANEVAL_NAME = "openai/openai_humaneval"


def load_humaneval() -> List[Document]:
    """Turn each HumanEval row into one LangChain Document."""
    ds = load_dataset(HUMANEVAL_NAME, split="test")

    docs: List[Document] = []
    for row in ds:
        # prompt = signature + docstring, canonical_solution = the body,
        # so joined together they form one complete working function
        content = row["prompt"] + row["canonical_solution"]

        docs.append(
            Document(
                page_content=content,
                metadata={
                    "task_id": row["task_id"],      
                    "entry_point": row["entry_point"],
                    "test": row["test"],           
                    "source": "humaneval",
                    "language": "python",
                },
            )
        )
    return docs

MBPP_NAME = "google-research-datasets/mbpp"


def load_mbpp(config: str = "full") -> List[Document]:
    """Turn each MBPP row (all splits) into one LangChain Document."""
    splits = load_dataset(MBPP_NAME, config)  # DatasetDict: train/test/...

    docs: List[Document] = []
    for split_name, ds in splits.items():
        for row in ds:
            text = row.get("text") or row.get("prompt")  # full vs sanitized
            content = f"# Task: {text}\n{row['code']}"

            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "task_id": f"MBPP/{row['task_id']}",
                        "test": "\n".join(row["test_list"]),
                        "source": "mbpp",
                        "language": "python",
                        "split": split_name,
                    },
                )
            )
    return docs


def load_all() -> List[Document]:
    return load_humaneval() + load_mbpp()

if __name__ == "__main__":
    docs = load_all()
    print(len(docs), "documents")
    print(20*"-")
    print(docs[7].metadata["task_id"])
    print(20*"-")
    print(docs[7].page_content)