from src.config import LLM_MODEL_NAME
from src.classifier import _prompt
from src.LLM import get_llm

print("model:", LLM_MODEL_NAME)
chain = _prompt | get_llm(temperature=0.0)
for q in [
    "What does this function do? def f(x): return x * 2",
    "Why is this loop incorrect? while i < 10: print(i)",
]:
    for _ in range(6):
        r = chain.invoke({"query": q})
        print(repr(r.content), "<-", r.response_metadata.get("model_name"))