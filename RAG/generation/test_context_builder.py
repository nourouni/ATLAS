import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAG_DIR = ROOT / "rag"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

try:
    from rag.generation.context_builder import build_contex
except ModuleNotFoundError:
    from generation.context_builder import build_contex


QUERIES = [
    "How does Cash Management work?",
    "What is Carte CIB?",
    "What does Attijari Real Time do?",
    "What is the difference between Gold Nationale and Gold Internationale?",
]


def main():
    for query in QUERIES:
        print("=" * 80)
        print(f"Query: {query}")
        context = build_context(query)
        print(json.dumps(context, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
