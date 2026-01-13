# ragcore_v2/src/retriever_v2.py

from __future__ import annotations

import json
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from indexing.load_index import load_faiss_bundle

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5

# ---------------------------------------------------------------------
# Load FAISS bundle ONCE (module-level)
# ---------------------------------------------------------------------
_FAISS_INDEX, _FAISS_META, _DOCSTORE_PATH = load_faiss_bundle()


def _load_docstore() -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    with open(_DOCSTORE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


_DOCSTORE = _load_docstore()

# Load embedding model ONCE (module-level)
_MODEL = SentenceTransformer(MODEL_NAME)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def embed_query(query: str) -> np.ndarray:
    """
    Returns a (1, d) float32 numpy array normalized for cosine similarity.
    """
    vec = _MODEL.encode([query], normalize_embeddings=True)
    return np.asarray(vec, dtype="float32")


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def dense_search(query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """
    Dense retrieval over FAISS (inner product with L2-normalized vectors).
    Uses docstore.jsonl alignment: FAISS id == docstore row index.
    """
    query_vec = embed_query(query)

    # (Safety) normalize again; harmless if already normalized
    faiss.normalize_L2(query_vec)

    scores, ids = _FAISS_INDEX.search(query_vec, top_k)

    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        doc = _DOCSTORE[int(idx)]
        results.append(
            {
                "id": int(idx),  # stable: FAISS id == docstore row
                "score": float(score),
                "text": doc.get("text"),
                "source": doc.get("source"),
                "chunk_i": doc.get("chunk_i"),
            }
        )

    return results


def index_info() -> Dict[str, Any]:
    """
    Quick debugging info.
    """
    return {
        "ntotal": int(_FAISS_INDEX.ntotal),
        "dim": int(_FAISS_INDEX.d),
        "docstore_rows": len(_DOCSTORE),
        "corpus_hash": _FAISS_META.get("corpus_hash"),
        "docstore_path": str(_DOCSTORE_PATH),
        "model": MODEL_NAME,
    }
