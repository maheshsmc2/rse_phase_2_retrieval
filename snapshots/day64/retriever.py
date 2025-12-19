# retriever.py – Day 56 version (dense + lexical + hybrid + reranker)

from typing import List, Dict
from pathlib import Path
import json
import hashlib

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

import time
from trace_helpers import init_trace, add_timing, save_trace, clip_text

# ---------------------------------------------------------
# 1. Models, paths & corpus (Day 45 + 46)
# ---------------------------------------------------------

ARTIFACT_DIR = Path("data")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Day 46: index versioning
INDEX_VERSION = "v1"  # bump to "v2", "v3", ... when corpus/model changes

EMB_PATH = ARTIFACT_DIR / f"doc_embeddings_{INDEX_VERSION}.npy"
INDEX_PATH = ARTIFACT_DIR / f"faiss_index_{INDEX_VERSION}.bin"
META_PATH = ARTIFACT_DIR / f"index_meta_{INDEX_VERSION}.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Dense model
model = SentenceTransformer(MODEL_NAME)

# Day 56: Reranker model
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Lazy-loaded reranker instance
_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Day 56: Lazy-load and cache the CrossEncoder reranker."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + eps)


# Load corpus (list of {"id", "text"} dicts)
CORPUS_PATH = ARTIFACT_DIR / "corpus_chunks.json"

with CORPUS_PATH.open(encoding="utf-8") as f:
    corpus = json.load(f)

DOCUMENTS: List[str] = [item["text"] for item in corpus]


def compute_corpus_hash(documents: List[str]) -> str:
    text = "\n".join(documents)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


CORPUS_HASH = compute_corpus_hash(DOCUMENTS)


# ---------------------------------------------------------
# 1.1 Build/load FAISS artifacts (Day 45 + 46)
# ---------------------------------------------------------

def _build_and_save_index():
    """Build FAISS index + embeddings from scratch and save artifacts."""
    print("[Day 45] Building embeddings and FAISS index from scratch...")

    # 1. Compute embeddings
    emb = model.encode(DOCUMENTS, convert_to_numpy=True)  # (N_docs, d)
    emb = l2_normalize(emb).astype("float32")

    # 2. Build index
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    # 3. Save artifacts
    np.save(EMB_PATH, emb)
    faiss.write_index(index, str(INDEX_PATH))

    meta = {
        "model_name": MODEL_NAME,
        "dim": int(dim),
        "n_docs": int(len(DOCUMENTS)),
        "normalized": True,
        "corpus_hash": CORPUS_HASH,
        "index_version": INDEX_VERSION,
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[Day 45] Saved embeddings, FAISS index, and metadata.")
    return emb, index, meta


def _load_artifacts():
    """Load FAISS index + embeddings if possible; otherwise build them."""
    if not (EMB_PATH.exists() and INDEX_PATH.exists() and META_PATH.exists()):
        return _build_and_save_index()

    print("[Day 45] Loading FAISS artifacts from disk...")

    try:
        emb = np.load(EMB_PATH)
        index = faiss.read_index(str(INDEX_PATH))
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[ERROR][Day 46] Failed to load artifacts: {e}")
        print("[Day 46] Rebuilding index...")
        return _build_and_save_index()

    # Self-healing checks
    if meta.get("index_version") != INDEX_VERSION:
        print(f"[WARN][Day 46] Index version mismatch. Rebuilding...")
        return _build_and_save_index()

    if meta.get("model_name") != MODEL_NAME:
        print(f"[WARN][Day 46] Model changed. Rebuilding...")
        return _build_and_save_index()

    if meta.get("corpus_hash") != CORPUS_HASH or meta.get("n_docs") != len(DOCUMENTS):
        print("[WARN][Day 46] Corpus changed. Rebuilding index...")
        return _build_and_save_index()

    if meta.get("dim") != emb.shape[1]:
        print("[WARN][Day 46] Dimension mismatch. Rebuilding index...")
        return _build_and_save_index()

    return emb, index, meta


DOC_EMBEDDINGS, faiss_index, META = _load_artifacts()

_dummy = model.encode(["hello world"], convert_to_numpy=True)
MODEL_DIM = _dummy.shape[1]

print(f"[Day 44] Model embedding dimension: {MODEL_DIM}")
print(f"[Day 44] Corpus embedding dimension: {DOC_EMBEDDINGS.shape[1]}")
print(f"[Day 44] FAISS index dimension: {faiss_index.d}")
print(f"[Day 45] Loaded meta: {META}")

assert MODEL_DIM == DOC_EMBEDDINGS.shape[1] == faiss_index.d == META["dim"], (
    f"[ERROR][Day 44/45] Dimension mismatch: "
    f"model={MODEL_DIM}, doc={DOC_EMBEDDINGS.shape[1]}, "
    f"index={faiss_index.d}, meta={META['dim']}"
)

assert META["n_docs"] == len(DOCUMENTS), (
    f"[ERROR][Day 45] Document count mismatch: meta={META['n_docs']}, corpus={len(DOCUMENTS)}"
)

_doc_norms = np.linalg.norm(DOC_EMBEDDINGS, axis=1)
assert np.allclose(_doc_norms.mean(), 1.0, atol=1e-2), (
    "[ERROR][Day 44] Corpus embeddings are not approximately L2-normalized."
)


# ---------------------------------------------------------
# 1.2 Lexical index
# ---------------------------------------------------------

BM25_VECTORIZER = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
BM25_MATRIX = BM25_VECTORIZER.fit_transform(DOCUMENTS)  # (N_docs, V)


# ---------------------------------------------------------
# 2. Utility: min–max normalization
# ---------------------------------------------------------

def min_max_norm(x: np.ndarray) -> np.ndarray:
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max - x_min < 1e-8:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


# ---------------------------------------------------------
# 3. Dense helpers (FAISS)
# ---------------------------------------------------------

def _encode_query_dense(query: str) -> np.ndarray:
    """Encode and normalize query for dense search."""
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = l2_normalize(q_emb).astype("float32")

    q_norms = np.linalg.norm(q_emb, axis=1)
    assert np.allclose(q_norms, 1.0, atol=1e-3), "[ERROR][Day 44] Query embedding not L2-normalized."
    return q_emb


def dense_search(query: str, top_k: int = 5) -> List[Dict]:
    """
    Return top_k documents by dense similarity using FAISS.
    Output format: list of {"id", "text", "score_dense", "score"} dicts.
    """
    q_emb = _encode_query_dense(query)
    top_k = min(top_k, len(DOCUMENTS))
    scores, indices = faiss_index.search(q_emb, top_k)

    results: List[Dict] = []
    for idx, score in zip(indices[0], scores[0]):
        doc = corpus[idx]
        s = float(score)
        results.append(
            {
                "id": doc.get("id", str(idx)),
                "text": doc["text"],
                "score_dense": s,
                "score": s,  # ✅ universal key
            }
        )
    return results


# ---------------------------------------------------------
# 3.1 Lexical retrieval
# ---------------------------------------------------------

def bm25_search(query: str, top_k: int = 5) -> List[Dict]:
    """
    Return top_k documents by BM25-like lexical similarity.
    Output format: list of {"id", "text", "score_bm25", "score"} dicts.
    """
    q_vec = BM25_VECTORIZER.transform([query])  # (1, V)
    scores = (BM25_MATRIX @ q_vec.T).toarray().ravel()  # (N_docs,)

    top_k = min(top_k, len(DOCUMENTS))
    top_idx = np.argsort(-scores)[:top_k]

    results: List[Dict] = []
    for idx in top_idx:
        doc = corpus[idx]
        s = float(scores[idx])
        results.append(
            {
                "id": doc.get("id", str(idx)),
                "text": doc["text"],
                "score_bm25": s,
                "score": s,  # ✅ universal key (lexical)
            }
        )
    return results


# ---------------------------------------------------------
# 3.2 Hybrid retrieval (dense + lexical)
# ---------------------------------------------------------

DEFAULT_ALPHA = 0.1  # tuned weight for lexical score in hybrid (Day 55)


def compute_hybrid_vector(
    dense_scores: np.ndarray,
    lexical_scores: np.ndarray,
    alpha: float,
) -> np.ndarray:
    dense_norm = min_max_norm(dense_scores)
    lex_norm = min_max_norm(lexical_scores)
    return alpha * lex_norm + (1.0 - alpha) * dense_norm


def hybrid_search(
    query: str,
    top_k: int = 5,
    alpha: float = DEFAULT_ALPHA,
) -> List[Dict]:
    """
    Output format: list of dicts with id/text and scores:
      {"id","text","score_hybrid","score_dense","score_bm25","score"}
    """
    dense_results = dense_search(query, top_k=top_k * 3)
    bm25_results = bm25_search(query, top_k=top_k * 3)

    merged: Dict[str, Dict] = {}

    for r in dense_results:
        merged[r["id"]] = {
            "id": r["id"],
            "text": r["text"],
            "score_dense": r["score_dense"],
            "score_bm25": 0.0,
        }

    for r in bm25_results:
        if r["id"] in merged:
            merged[r["id"]]["score_bm25"] = r["score_bm25"]
        else:
            merged[r["id"]] = {
                "id": r["id"],
                "text": r["text"],
                "score_dense": 0.0,
                "score_bm25": r["score_bm25"],
            }

    ids = list(merged.keys())
    dense_scores = np.array([merged[i]["score_dense"] for i in ids], dtype=np.float32)
    bm25_scores = np.array([merged[i]["score_bm25"] for i in ids], dtype=np.float32)

    hybrid_scores = compute_hybrid_vector(dense_scores=dense_scores, lexical_scores=bm25_scores, alpha=alpha)

    top_k = min(top_k, len(ids))
    top_idx = np.argsort(-hybrid_scores)[:top_k]

    results: List[Dict] = []
    for i in top_idx:
        id_ = ids[i]
        entry = merged[id_]
        s_h = float(hybrid_scores[i])
        results.append(
            {
                "id": id_,
                "text": entry["text"],
                "score_hybrid": s_h,
                "score_dense": float(entry["score_dense"]),
                "score_bm25": float(entry["score_bm25"]),
                "score": s_h,  # ✅ universal key (hybrid)
            }
        )

    return results


# ---------------------------------------------------------
# 3.3 Reranker (Day 56)
# ---------------------------------------------------------

def rerank_with_cross_encoder(
    query: str,
    candidates: List[Dict],
    top_k: int = 5,
) -> List[Dict]:
    """
    Attach:
      - score_rerank
      - score (universal) == score_rerank
    """
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)

    for cand, s in zip(candidates, scores):
        sr = float(s)
        cand["score_rerank"] = sr
        cand["score"] = sr  # ✅ universal key (rerank)

    candidates_sorted = sorted(candidates, key=lambda x: x["score_rerank"], reverse=True)
    return candidates_sorted[:top_k]


def hybrid_then_rerank(
    query: str,
    retrieve_k: int = 20,
    final_k: int = 5,
    alpha: float = DEFAULT_ALPHA,
    top_k: int | None = None,  # ✅ backward-compatible alias
) -> List[Dict]:
    """
    Day 56: Full retrieval pipeline.

    Backward compat:
      If caller passes top_k=..., we treat that as final_k.
    """
    if top_k is not None:
        final_k = int(top_k)

    candidates = hybrid_search(query, top_k=retrieve_k, alpha=alpha)
    reranked = rerank_with_cross_encoder(query, candidates, top_k=final_k)
    return reranked


# ---------------------------------------------------------
# 4. Score computation helper (Day 54 debug path)
# ---------------------------------------------------------

def dense_scores_all(query: str) -> np.ndarray:
    q_emb = _encode_query_dense(query)
    scores, indices = faiss_index.search(q_emb, len(DOCUMENTS))
    dense_vec = np.zeros(len(DOCUMENTS), dtype=np.float32)
    dense_vec[indices[0]] = scores[0]
    return dense_vec


def compute_scores(query: str, alpha_override: float | None = None) -> Dict[str, np.ndarray]:
    query = query.strip()
    if not query:
        raise ValueError("Query must be non-empty")

    dense_raw = dense_scores_all(query)
    dense_norm = min_max_norm(dense_raw)

    q_vec = BM25_VECTORIZER.transform([query])
    bm25_raw = (BM25_MATRIX @ q_vec.T).toarray().ravel()
    bm25_norm = min_max_norm(bm25_raw)

    if alpha_override is not None:
        alpha = float(alpha_override)
    else:
        num_words = len(query.split())
        alpha = 0.8 if num_words <= 3 else 0.2

    hybrid_scores = compute_hybrid_vector(dense_scores=dense_raw, lexical_scores=bm25_raw, alpha=alpha)

    print("\n[HYBRID DEBUG][Day 54]")
    print("alpha:", alpha)
    print("BM25 (raw)  :", bm25_raw[:5])
    print("Dense (raw) :", dense_raw[:5])
    print("Hybrid (top):", np.sort(hybrid_scores)[-5:])

    return {
        "query": query,
        "alpha": alpha,
        "dense_raw": dense_raw,
        "dense_norm": dense_norm,
        "bm25_raw": bm25_raw,
        "bm25_norm": bm25_norm,
        "hybrid_scores": hybrid_scores,
    }


# ---------------------------------------------------------
# 5. Main RAG-style helpers (older debug API)
# ---------------------------------------------------------

def _select_best_doc(hybrid_scores: np.ndarray) -> int:
    mean = float(hybrid_scores.mean())
    std = float(hybrid_scores.std())
    thr = max(0.0, min(mean - std, 1.0))

    mask = hybrid_scores >= thr
    candidate_indices = np.where(mask)[0]

    if candidate_indices.size == 0:
        return int(np.argmax(hybrid_scores))

    return int(candidate_indices[np.argmax(hybrid_scores[candidate_indices])])


def answer_query(query: str) -> str:
    txt = query.strip()
    if not txt:
        return "Please provide a non-empty query."

    scores = compute_scores(txt)
    dense_raw = scores["dense_raw"]
    bm25_raw = scores["bm25_raw"]
    hybrid_scores = scores["hybrid_scores"]
    alpha = scores["alpha"]

    mean = float(hybrid_scores.mean())
    std = float(hybrid_scores.std())
    thr = max(0.0, min(mean - std, 1.0))

    best_idx = _select_best_doc(hybrid_scores)

    best_doc = DOCUMENTS[best_idx]
    best_dense = float(dense_raw[best_idx])
    best_bm25 = float(bm25_raw[best_idx])
    best_hybrid = float(hybrid_scores[best_idx])

    return (
        f"Answer: {best_doc}\n\n"
        f"[debug] alpha={alpha:.2f}, dense={best_dense:.3f}, bm25={best_bm25:.3f}, "
        f"hybrid={best_hybrid:.3f}, threshold={thr:.3f}"
    )


def retrieve_top_k(query: str, k: int = 3) -> Dict[str, object]:
    scores = compute_scores(query)
    dense_raw = scores["dense_raw"]
    bm25_raw = scores["bm25_raw"]
    hybrid_scores = scores["hybrid_scores"]
    alpha = scores["alpha"]

    n_docs = len(DOCUMENTS)
    k = min(k, n_docs)
    top_indices = np.argsort(hybrid_scores)[::-1][:k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        idx = int(idx)
        results.append(
            {
                "rank": rank,
                "doc_index": idx,
                "text": DOCUMENTS[idx],
                "dense": float(dense_raw[idx]),
                "bm25": float(bm25_raw[idx]),
                "hybrid": float(hybrid_scores[idx]),
            }
        )

    return {"query": scores["query"], "alpha": float(alpha), "results": results}


def retrieve_with_trace(
    query: str,
    *,
    retrieve_k: int = 20,
    final_k: int = 5,
    alpha: float = DEFAULT_ALPHA,
    use_reranker: bool = True,
) -> Dict[str, object]:
    """
    Day 61: Observability wrapper around your existing pipeline (Day 56).
    """
    q = query.strip()
    if not q:
        raise ValueError("Query must be non-empty")

    trace = init_trace(q, meta={
        "retrieve_k": retrieve_k,
        "final_k": final_k,
        "alpha": float(alpha),
        "use_reranker": bool(use_reranker),
        "index_version": INDEX_VERSION,
        "model_name": MODEL_NAME,
        "reranker_model": RERANKER_MODEL_NAME,
    })

    # 1) Dense
    t0 = time.perf_counter()
    dense_results = dense_search(q, top_k=retrieve_k)
    add_timing(trace, "dense", (time.perf_counter() - t0) * 1000)

    trace["stages"]["dense"] = [
        {"rank": i + 1, "id": r["id"], "score_dense": float(r["score_dense"]), "text": clip_text(r["text"])}
        for i, r in enumerate(dense_results)
    ]

    # 2) BM25
    t0 = time.perf_counter()
    bm25_results = bm25_search(q, top_k=retrieve_k)
    add_timing(trace, "bm25", (time.perf_counter() - t0) * 1000)

    trace["stages"]["bm25"] = [
        {"rank": i + 1, "id": r["id"], "score_bm25": float(r["score_bm25"]), "text": clip_text(r["text"])}
        for i, r in enumerate(bm25_results)
    ]

    # 3) Hybrid
    t0 = time.perf_counter()
    hybrid_candidates = hybrid_search(q, top_k=retrieve_k, alpha=alpha)
    add_timing(trace, "hybrid", (time.perf_counter() - t0) * 1000)

    trace["stages"]["hybrid"] = [
        {
            "rank": i + 1,
            "id": r["id"],
            "score_hybrid": float(r["score_hybrid"]),
            "score_dense": float(r["score_dense"]),
            "score_bm25": float(r["score_bm25"]),
            "text": clip_text(r["text"]),
        }
        for i, r in enumerate(hybrid_candidates)
    ]

    # 4) Rerank (optional)
    if use_reranker:
        t0 = time.perf_counter()
        reranked = rerank_with_cross_encoder(q, hybrid_candidates, top_k=final_k)
        add_timing(trace, "rerank", (time.perf_counter() - t0) * 1000)

        trace["stages"]["rerank"] = [
            {
                "rank": i + 1,
                "id": r["id"],
                "score_rerank": float(r.get("score_rerank", 0.0)),
                "score_hybrid": float(r.get("score_hybrid", 0.0)),
                "score_dense": float(r.get("score_dense", 0.0)),
                "score_bm25": float(r.get("score_bm25", 0.0)),
                "text": clip_text(r["text"]),
            }
            for i, r in enumerate(reranked)
        ]
        final_selected = reranked
    else:
        final_selected = hybrid_candidates[:final_k]

    trace["final"]["selected"] = [
        {
            "rank": i + 1,
            "id": r["id"],
            "score_rerank": float(r.get("score_rerank", 0.0)),
            "score_hybrid": float(r.get("score_hybrid", 0.0)),
            "score_dense": float(r.get("score_dense", 0.0)),
            "score_bm25": float(r.get("score_bm25", 0.0)),
            "text": clip_text(r["text"], n=300),
        }
        for i, r in enumerate(final_selected)
    ]

    path = save_trace(trace)
    trace["meta"]["trace_path"] = str(path)

    return {"final": final_selected, "trace": trace}
