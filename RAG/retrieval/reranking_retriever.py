try:
    from rag.retrieval.dense_retriever import DenseRetriever
except ModuleNotFoundError:
    from retrieval.dense_retriever import DenseRetriever


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


class RerankingRetriever:
    def __init__(self, dense_retriever=None, reranker_model=RERANKER_MODEL, candidate_k=20):
        from sentence_transformers import CrossEncoder

        self.dense_retriever = dense_retriever or DenseRetriever()
        self.reranker = CrossEncoder(reranker_model)
        self.candidate_k = candidate_k
        self.last_dense_latency_ms = None
        self.last_rerank_latency_ms = None

    def retrieve(self, query: str, top_k: int = 5):
        import time

        start = time.perf_counter()
        candidates = self.dense_retriever.retrieve(query, top_k=self.candidate_k)
        self.last_dense_latency_ms = (time.perf_counter() - start) * 1000

        pairs = [(query, candidate["document"]) for candidate in candidates]
        start = time.perf_counter()
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        self.last_rerank_latency_ms = (time.perf_counter() - start) * 1000

        reranked = []
        for candidate, score in zip(candidates, scores):
            updated = dict(candidate)
            updated["score"] = float(score)
            reranked.append(updated)
        return sorted(reranked, key=lambda item: item["score"], reverse=True)[:top_k]


def retrieve(query: str, top_k: int = 5):
    return RerankingRetriever().retrieve(query, top_k)
