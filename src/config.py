# src/config.py
from pathlib import Path
import os
from dotenv import load_dotenv

# --- Paths (anchored to the repo root, so they work from any working directory) ---
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

# --- Vector store ---
COLLECTION_NAME = "code_knowledge"

# --- Embeddings ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Retrieval ---
TOP_K = 5

load_dotenv(BASE_DIR / ".env")

# --- LLM (OpenRouter, OpenAI-compatible) ---
LLM_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

MEMORY_DB_PATH = BASE_DIR / "data" / "memory.db"

RELEVANCE_MAX_DISTANCE = None