try:
    from RAG.retrieval.bm25_retriever import BM25Retriever
    from RAG.retrieval.dense_retriever import DenseRetriever
except ModuleNotFoundError:
    from rag.retrieval.bm25_retriever import BM25Retriever
    from rag.retrieval.dense_retriever import DenseRetriever


RRF_K = 60


class HybridRetriever:
    def __init__(self, dense_retriever=None, bm25_retriever=None, rrf_k=RRF_K, candidate_k=20):
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def retrieve(self, query: str, top_k: int = 5):
        dense_results = self.dense_retriever.retrieve(query, top_k=self.candidate_k)
        bm25_results = self.bm25_retriever.retrieve(query, top_k=self.candidate_k)

        fused = {}
        for results in (dense_results, bm25_results):
            for rank, result in enumerate(results, start=1):
                document_id = result["document_id"]
                if document_id not in fused:
                    fused[document_id] = dict(result)
                    fused[document_id]["score"] = 0.0
                fused[document_id]["score"] += 1.0 / (self.rrf_k + rank)

        return sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:top_k]


def retrieve(query: str, top_k: int = 5):
    return HybridRetriever().retrieve(query, top_k)
