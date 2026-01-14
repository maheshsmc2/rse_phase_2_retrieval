import numpy as np
from sentence_transformers import SentenceTransformer
from ragcore_v2.indexing.load_index import load_faiss_bundle

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

_index, _meta = load_faiss_bundle()
_model = SentenceTransformer(MODEL_NAME)


def embed_query(query: str) -> np.ndarray:
    vec = _model.encode([query], normalize_embeddings=True)
    return np.asarray(vec, dtype="float32")


def dense_search(query: str, top_k: int = TOP_K):
    qvec = embed_query(query)
    scores, indices = _index.search(qvec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "id": _meta["ids"][idx],
            "score": float(score),
            "text": _meta["texts"][idx]
        })

    return results
