from rank_bm25 import BM25Okapi

try:
    from RAG.retrieval.common import build_retrieval_text, load_verified_documents, make_result, tokenize
except ModuleNotFoundError:
    from rag.retrieval.common import build_retrieval_text, load_verified_documents, make_result, tokenize


class BM25Retriever:
    def __init__(self, documents=None):
        self.documents = documents or load_verified_documents()
        self.texts = [build_retrieval_text(document) for document in self.documents]
        self.tokenized_texts = [tokenize(text) for text in self.texts]
        self.index = BM25Okapi(self.tokenized_texts)

    def retrieve(self, query: str, top_k: int = 5):
        scores = self.index.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
        return [
            make_result(self.documents[idx], float(scores[idx]), self.texts[idx])
            for idx in ranked
        ]


def retrieve(query: str, top_k: int = 5):
    return BM25Retriever().retrieve(query, top_k)
