# run_query.py

from retriever import dense_search, hybrid_search, hybrid_then_rerank


# ----------------------------
# Day 64: Safety + Debug knobs
# ----------------------------
DEFAULT_MIN_SCORE = 0.0  # set to 0.35 later once you know your score scale


def classify_query(query: str) -> str:
    q = query.lower().strip()

    # simple intent rules (fast baseline)
    if any(p in q for p in ["what is", "define", "meaning of", "explain"]):
        return "definition"

    if any(p in q for p in ["policy", "rule", "leave", "probation", "attendance", "salary"]):
        return "policy"

    return "general"


def _extract_top_score(result):
    """
    Robustly extract a 'best score' from retriever outputs.

    Supports:
    - list[dict] where dict has "score"
    - dict with keys like {"hits": [...]} or {"results": [...]}
    - otherwise returns None
    """
    if result is None:
        return None

    # case 1: list of hits
    if isinstance(result, list):
        if not result:
            return None
        first = result[0]
        if isinstance(first, dict):
            return first.get("score", None)
        return None

    # case 2: dict wrapper
    if isinstance(result, dict):
        for k in ("hits", "results", "documents", "docs"):
            v = result.get(k)
            if isinstance(v, list) and v:
                first = v[0]
                if isinstance(first, dict):
                    return first.get("score", None)
        # sometimes score is top-level
        return result.get("score", None)

    return None


def _gate_if_low_confidence(result, min_score: float, debug: bool, route_name: str):
    """
    If best score < min_score, return a standard "no answer" payload.
    Otherwise return original result.
    """
    if min_score is None or min_score <= 0:
        return result

    best_score = _extract_top_score(result)

    if best_score is None:
        # Can't confidently gate if we don't see a score.
        # Still print debug so you notice this.
        if debug:
            print(f"[DAY64][{route_name}] No score found in result; gating skipped.")
        return result

    try:
        best_score_f = float(best_score)
    except Exception:
        if debug:
            print(f"[DAY64][{route_name}] Score not numeric ({best_score}); gating skipped.")
        return result

    if best_score_f < float(min_score):
        if debug:
            print(f"[DAY64][{route_name}] NO_ANSWER triggered. best_score={best_score_f} < min_score={min_score}")
        return {
            "answer": None,
            "reason": "Not enough information found",
            "best_score": best_score_f,
            "min_score": float(min_score),
            "raw": result,  # keep for debugging
        }

    if debug:
        print(f"[DAY64][{route_name}] PASS gate. best_score={best_score_f} >= min_score={min_score}")

    return result


def run_query(
    query: str,
    top_k: int = 5,
    alpha: float = 0.2,
    use_reranker: bool = True,
    min_score: float = DEFAULT_MIN_SCORE,  # ✅ Day 64 gate knob
    debug: bool = False,                   # ✅ Day 64 visibility
):
    q_type = classify_query(query)

    if debug:
        print("\n" + "=" * 60)
        print(f"[DAY64] query='{query}' | q_type='{q_type}' | top_k={top_k} | alpha={alpha} | rerank={use_reranker} | min_score={min_score}")

    # Route A: definition → dense only
    if q_type == "definition":
        out = dense_search(query, top_k=top_k)
        return _gate_if_low_confidence(out, min_score=min_score, debug=debug, route_name="definition:dense")

    # Route B: policy → hybrid (+ optional rerank)
    if q_type == "policy":
        if use_reranker:
            out = hybrid_then_rerank(query, top_k=top_k, alpha=alpha)
            return _gate_if_low_confidence(out, min_score=min_score, debug=debug, route_name="policy:hybrid_then_rerank")

        out = hybrid_search(query, top_k=top_k, alpha=alpha)
        return _gate_if_low_confidence(out, min_score=min_score, debug=debug, route_name="policy:hybrid")

    # Route C: general → dense only
    out = dense_search(query, top_k=top_k)
    return _gate_if_low_confidence(out, min_score=min_score, debug=debug, route_name="general:dense")


if __name__ == "__main__":
    tests = [
        "What is FAISS?",
        "probation leave policy",
        "tell me about RAG",
        "gibberish query that should not match anything xyz123",
    ]

    for q in tests:
        print(q, "=>", classify_query(q))
        out = run_query(q, top_k=3, alpha=0.2, use_reranker=False, min_score=0.35, debug=True)
        print("OUTPUT:", out)
        print("-" * 40)
