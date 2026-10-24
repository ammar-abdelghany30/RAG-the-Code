# src/code_runner.py
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

TIMEOUT_SECONDS = 10
MAX_OUTPUT_CHARS = 10_000


@dataclass
class RunResult:
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def _to_text(value) -> str:
    # On timeout, Python may hand back bytes (or None) even with text=True
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if len(value) > MAX_OUTPUT_CHARS:
        value = value[:MAX_OUTPUT_CHARS] + "\n[output truncated]"
    return value


def _minimal_env() -> dict:
    # Do NOT pass our environment: it contains the LLM API key loaded from .env
    keep = ("SYSTEMROOT", "PATH", "TEMP", "TMP")
    return {k: os.environ[k] for k in keep if k in os.environ}


def run_code(code: str, timeout: int = TIMEOUT_SECONDS) -> RunResult:
    """Run a Python source string in a separate process and capture the result."""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "main.py"
        script.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],  # -I = isolated mode
                cwd=tmp,
                env=_minimal_env(),
                stdin=subprocess.DEVNULL,  # input() fails fast instead of hanging
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            return RunResult(
                _to_text(e.stdout),
                _to_text(e.stderr) + f"\nStopped: exceeded {timeout}s time limit.",
                -1,
                True,
            )
        return RunResult(_to_text(proc.stdout), _to_text(proc.stderr), proc.returncode, False)


def extract_code(text: str) -> str:
    """Return the first fenced code block from an LLM answer, or the text itself if none."""
    match = re.search(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.DOTALL)
    return match.group(1) if match else text