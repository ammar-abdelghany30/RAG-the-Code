# scripts/test_code_runner.py
from src.config import LLM_API_KEY  # importing config loads .env into os.environ
from src.code_runner import extract_code, run_code
from src.loader import load_humaneval

FENCE = "`" * 3
failures = 0


def check(name, ok, detail=""):
    global failures
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    failures += 0 if ok else 1


assert LLM_API_KEY, "the leak test needs an API key in .env"

r = run_code('print("hello")')
check("captures output", r.ok and r.stdout.strip() == "hello", repr(r.stdout))

r = run_code("print(1 / 0)")
check("captures runtime errors", not r.ok and "ZeroDivisionError" in r.stderr)

r = run_code("while True:\n    pass", timeout=2)
check("stops an infinite loop", r.timed_out)

r = run_code("input()")
check("input() fails fast", not r.ok and "EOFError" in r.stderr)

r = run_code("import os\nprint(os.environ.get('LLM_API_KEY'))")
check("API key is not visible to the code", r.stdout.strip() == "None")

md = f"Here you go:\n{FENCE}python\nprint(2 + 2)\n{FENCE}\nDone."
check("extract_code finds the block", extract_code(md).strip() == "print(2 + 2)")
r = run_code(extract_code(md))
check("extracted code runs", r.ok and r.stdout.strip() == "4")

row = load_humaneval()[0]
program = (
    row.page_content + "\n\n" + row.metadata["test"]
    + f"\n\ncheck({row.metadata['entry_point']})\n"
)
r = run_code(program)
check("HumanEval/0 solution passes its own stored test", r.ok, r.stderr[-150:])

print("\nALL PASSED" if failures == 0 else f"\n{failures} CHECK(S) FAILED")