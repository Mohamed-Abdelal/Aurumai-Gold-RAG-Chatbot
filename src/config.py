"""Centralized configuration for AurumAI — Gold Investment Recommendation Assistant."""

from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

INDEX_DIR = ROOT_DIR / "index"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npz"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

EVAL_DIR = ROOT_DIR / "eval"
EVAL_QUESTIONS_PATH = EVAL_DIR / "questions.json"
EVAL_RESULTS_PATH = EVAL_DIR / "results.json"
EVAL_REPORT_PATH = EVAL_DIR / "evaluation_report.md"

# Word-based chunking hyperparameters
CHUNK_SIZE_WORDS = 220      # Target words per chunk
CHUNK_OVERLAP_WORDS = 40    # Sliding word overlap between consecutive chunks
MIN_CHUNK_WORDS = 25        # Drop tiny trailing fragments

# Embedding model (compact, fast on CPU/Streamlit Cloud, 384-dim)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Pure-NumPy retrieval defaults
DEFAULT_TOP_K = 4
DEFAULT_MIN_SIMILARITY = 0.25

# Groq LLM defaults (active free-tier chat models)
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
AVAILABLE_GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_COMPLETION_TOKENS = 1024
DEFAULT_REASONING_EFFORT = "low"
