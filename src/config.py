"""Configuration settings for AurumAI — Gold Investment RAG Chatbot."""

from pathlib import Path

# Project directories and files
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

INDEX_DIR = ROOT_DIR / "index"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npz"
CHUNKS_PATH = INDEX_DIR / "chunks.json"

EVAL_DIR = ROOT_DIR / "eval"
EVAL_QUESTIONS_PATH = EVAL_DIR / "questions.json"
EVAL_RESULTS_PATH = EVAL_DIR / "results.json"
EVAL_REPORT_PATH = EVAL_DIR / "evaluation_report.md"

# Word-based chunking settings
CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40
MIN_CHUNK_WORDS = 25

# Embedding model (384-dimensional dense vectors)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Pure-NumPy retrieval settings
DEFAULT_TOP_K = 4
DEFAULT_MIN_SIMILARITY = 0.25

# Groq LLM settings
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_COMPLETION_TOKENS = 1024
DEFAULT_REASONING_EFFORT = "low"
