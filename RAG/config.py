from pathlib import Path
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

RAG_DIR = Path(__file__).resolve().paren
ROOT_DIR = RAG_DIR.paren
DATA_DIR = RAG_DIR / "data"
VECTORSTORE_DIR = RAG_DIR / "vectorstore" / "chroma"

EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
CHROMA_PATH = os.environ.get("RAG_VECTORSTORE_PATH", str(VECTORSTORE_DIR))
CHROMA_COLLECTION = os.environ.get("RAG_COLLECTION", "atlas_demo_verified_knowledge")
VERIFIED_KNOWLEDGE_PATH = os.environ.get(
    "RAG_EVIDENCE_PATH",
    str(DATA_DIR / "atlas_demo_chatbot_grounded_answers_verified.csv"),
)
INDEX_BUILD_REPORT_PATH = os.environ.get(
    "RAG_INDEX_REPORT_PATH",
    str(RAG_DIR / "vectorstore" / "index_build_report.json"),
)

RETRIEVER_TYPE = "hybrid_rrf"
TOP_K = 5
CANDIDATE_K = 20
RRF_K = 60
MIN_LEXICAL_OVERLAP = 1
CONTEXT_TOP_K = 3
CONTEXT_MAX_CHARS = 6000
QUESTION_TYPE_MATCH_BOOST = 0.035
QUESTION_TYPE_MISMATCH_PENALTY = 0.012
WEAK_RECORD_PENALTY = 0.08
COMPARISON_MATCH_BOOST = 0.05

LLM_PROVIDER = "ollama"
LLM_MODEL = "phi4-mini"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", os.environ.get("CHATBOT_OLLAMA_HOST", "http://127.0.0.1:11434"))
LLM_TEMPERATURE = 0.0
LLM_TIMEOUT_SECONDS = 120
LLM_MAX_CITATION_RETRIES = 0
DETERMINISTIC_SINGLE_EVIDENCE_FAST_PATH = True
