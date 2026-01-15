from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from sentence_transformers import CrossEncoder


@dataclass
class Reranked:
    chunk_id: str
    text: str
    retr_score: float
    rerank_score: float


DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: List[Tuple[str, str, float]]) -> List[Reranked]:
        """
        hits: List of (chunk_id, text, retr_score)
        Returns sorted by rerank_score desc.
        """
        pairs = [(query, text) for _, text, _ in hits]
        scores = self.model.predict(pairs)

        out: List[Reranked] = []
        for (chunk_id, text, retr_score), s in zip(hits, scores):
            out.append(Reranked(chunk_id=chunk_id, text=text, retr_score=float(retr_score), rerank_score=float(s)))

        out.sort(key=lambda x: x.rerank_score, reverse=True)
        return out
