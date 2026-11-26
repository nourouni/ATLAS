import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RAG_DIR = ROOT / "RAG"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

try:
    from RAG import config
    from RAG.retrieval.common import bge_query, clean
except ModuleNotFoundError:
    from rag import config
    from rag.retrieval.common import bge_query, clean


class DenseRetriever:
    def __init__(self, model_name=None, chroma_path=None, collection_name=None):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or config.EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
        self.client = chromadb.PersistentClient(path=str(chroma_path or config.CHROMA_PATH))
        self.collection = self.client.get_collection(collection_name or config.CHROMA_COLLECTION)

    def _embed_query(self, query):
        embedding = self.model.encode(
            [bge_query(query, self.model_name)],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm:
            embedding = embedding / norm
        return embedding.tolist()

    def retrieve(self, query: str, top_k: int = 5):
        result = self.collection.query(
            query_embeddings=[self._embed_query(query)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        rows = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for document_id, document, metadata, distance in zip(ids, docs, metadatas, distances):
            rows.append({
                "document_id": document_id,
                "document": document,
                "product": clean((metadata or {}).get("product")),
                "verification_status": clean((metadata or {}).get("verification_status")),
                "score": 1.0 - float(distance),
                "metadata": metadata or {},
            })
        return rows


def retrieve(query: str, top_k: int = 5):
    return DenseRetriever().retrieve(query, top_k)
